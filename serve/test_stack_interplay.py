"""serve/test_stack_interplay.py - what only the three changes together can get wrong: the fair queue (FairLock with a
preempt rank), the loop close (a wrap-up pass inside one request) and the VRAM preempt (a cut request, retried).  CPU only,
a fake elastic engine, no model, no GPU.

    python -m unittest serve.test_stack_interplay -v
"""
from __future__ import annotations

import io
import sys
import threading
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, Service, serve  # noqa: E402
from serve.test_vram_preempt import (ROOT, SHORT, ElasticEngine, PreemptCase, anthropic, load_helper,  # noqa: E402
                                     openai)

LOOP = "Let me think. " + "0" * 600          # one token for ever while thinking: the loop close wraps it up
ANSWER = "The answer is 42."
PLAIN = "hm\n</think>\n\n" + ANSWER


def thinking(content="T-request", **extra):
    return {"stream": True, "max_tokens": 3000, "chat_template_kwargs": {"enable_thinking": True},
            "messages": [{"role": "user", "content": content}], **extra}


class RecordingEngine(ElasticEngine):
    """ElasticEngine that also keeps the sampling of every pass and the most generations inside it at once."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.samplings, self.max_active = [], 0

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        with self.lock:
            self.samplings.append(dict(sampling or {}))
        gen = super().generate(ids, max_new, sampling, cancel, embeddings)
        for t in gen:
            self.max_active = max(self.max_active, self.active)
            yield t
        self.max_active = max(self.max_active, self.active)


class InterplayCase(PreemptCase):
    def setUp(self):
        self.tok = ByteTokenizer()
        self.engine = RecordingEngine(self.tok, self.plans, self.elastic)
        self.svc = Service(self.engine, self.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.svc.answer_room = False                 # the cases are about the repeated-token guard alone
        self.svc.model_arch = getattr(self, "arch", None)
        self.httpd = serve(self.svc, port=0)
        self.port = self.httpd.server_address[1]
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)
        self.calls = []

    def starts(self):
        return [e[1] for e in self.engine.log if e[0] == "start"]


class PreemptDuringALoopCloseWrapUp(InterplayCase):
    # pass 1 loops (256 zeros, closed by the wrap-up); pass 2 is still reading its prompt when the preempt comes; the
    # two requests behind it answer at once
    plans = [{"text": LOOP, "delay": 0.0}, {"lead": 5.0, "text": ANSWER}, {"text": SHORT, "delay": 0.0}]

    def test_the_turn_is_released_once_and_the_cut_request_gets_the_retryable_error(self):
        out = io.StringIO()
        with mock.patch.object(sys, "stdout", out):
            a = self.start("/v1/chat/completions", thinking("A-request"))
            self.wait_until(lambda: self.engine.starts == 2 and self.engine.active == 1,
                            "the wrap-up pass never started")
            b = self.start("/v1/chat/completions", openai("B-request", stream=False))
            self.wait_until(lambda: self.svc.status["queued"] == 1, "B never queued")
            c = self.start("/v1/chat/completions", openai("C-request", stream=False))
            self.wait_until(lambda: self.svc.status["queued"] == 2, "C never queued")
            code, _, body = self.vram({"reserve_mib": 4000, "preempt": True})
            for call in (a, b, c):
                self.assertTrue(call.finished.wait(10))
        self.assertEqual((code, body["preempt"]["outcome"], body["preempt"]["cut"]), (200, "cut", 1))
        self.assertIn("wrapping up the thinking", out.getvalue())
        self.assertEqual(out.getvalue().count("wrapping up the thinking"), 1)
        # the thinking had been streamed, so the only honest ending is the error: no finish_reason, no normal stop
        events = [d for _, d in a.sse()]
        self.assertIn("vram_preempted", a.text)
        self.assertEqual(events[-1], "[DONE]")
        self.assertTrue(any(isinstance(d, dict) and d.get("error", {}).get("code") == "vram_preempted" for d in events))
        self.assertFalse(any(isinstance(d, dict) and any(ch.get("finish_reason")
                                                         for ch in d.get("choices", [])) for d in events))
        # the resize ran with the engine idle, then B and C ran one at a time: the turn was handed on exactly once
        self.assertEqual([(r["reserve"], r["active"]) for r in self.engine.resizes], [(4000, 0)])
        self.assertEqual(self.engine.max_active, 1)
        for call in (b, c):
            self.assertEqual(call.status, 200, call.text)
            self.assertEqual(call.json()["choices"][0]["finish_reason"], "stop")
        self.assertEqual([e[0] for e in self.engine.log if e[0] in ("start", "resize")],
                         ["start", "start", "resize", "start", "start"])        # A, A's wrap-up pass, resize, B, C
        self.assertFalse(self.svc.fifo.locked())
        self.assertEqual(self.svc.fifo.waiting(), 0)
        self.assertEqual(self.svc.turn_holders, {})
        self.assertFalse(self.svc.status.get("busy"))
        self.assertEqual(self.svc.preempt_cuts, 1)


class PresencePenaltyAcrossACut(InterplayCase):
    arch = "qwen4exp"                                # the thinking default has presence_penalty 1.5

    def penalties(self):
        return [s.get("presence_penalty") for s in self.engine.samplings]

    def test_a_request_the_server_runs_again_keeps_its_default_sampling(self):
        self.engine.plans[:] = [{"lead": 5.0}, {"text": PLAIN, "delay": 0.0}]    # cut while reading its prompt
        a = self.start("/v1/chat/completions", thinking())
        self.wait_until(lambda: self.engine.active == 1, "A never started")
        code, _, body = self.vram({"reserve_mib": 4000, "preempt": True})
        self.assertEqual((code, body["preempt"]["cut"]), (200, 1))
        self.assertTrue(a.finished.wait(10))
        self.assertEqual(self.svc.preempt_requeues, 1)
        self.assertEqual(self.penalties(), [1.5, 1.5])
        self.assertEqual(a.sse()[-1][1], "[DONE]")
        self.assertIn('"finish_reason": "stop"', a.text)        # a normal ending, no error chunk
        self.assertNotIn("vram_preempted", a.text)

    def test_a_client_retry_after_the_error_keeps_its_default_sampling(self):
        self.engine.plans[:] = [{"text": "thinking " * 80, "delay": 0.01}, {"text": PLAIN, "delay": 0.0}]
        helper = load_helper()
        result = {}

        def go():
            result["reply"] = helper.post_with_retry(
                f"http://127.0.0.1:{self.port}/v1/chat/completions", thinking(), stream=True, sleep=lambda s: None)
        t = threading.Thread(target=go)
        t.start()
        self.wait_until(lambda: self.engine.yielded > 5, "the first attempt never streamed")
        code, _, body = self.vram({"reserve_mib": 4000, "preempt": True})
        self.assertEqual((code, body["preempt"]["cut"]), (200, 1))
        t.join(15)
        self.assertFalse(t.is_alive())
        self.assertTrue(result["reply"].ok, result["reply"])
        self.assertEqual(result["reply"].attempts, 2)
        self.assertEqual(self.penalties(), [1.5, 1.5])

    def test_an_explicit_value_survives_the_cut_and_the_retry_too(self):
        self.engine.plans[:] = [{"text": "thinking " * 80, "delay": 0.01}, {"text": PLAIN, "delay": 0.0}]
        helper = load_helper()
        result = {}

        def go():
            result["reply"] = helper.post_with_retry(
                f"http://127.0.0.1:{self.port}/v1/chat/completions", thinking(presence_penalty=0), stream=True,
                sleep=lambda s: None)
        t = threading.Thread(target=go)
        t.start()
        self.wait_until(lambda: self.engine.yielded > 5, "the first attempt never streamed")
        self.vram({"reserve_mib": 4000, "preempt": True})
        t.join(15)
        self.assertTrue(result["reply"].ok, result["reply"])
        self.assertEqual(self.penalties(), [0, 0])                 # the request's own 0 beats the card's 1.5, twice


class PreemptAndTheLine(InterplayCase):
    """A preempt cuts the request that holds the turn and never one waiting in the fair line; the line keeps its order."""

    def test_a_cut_request_that_sent_nothing_is_run_again_ahead_of_the_waiting_ones_in_their_order(self):
        self.engine.plans[:] = [{"lead": 5.0}, {"text": SHORT, "delay": 0.0}]
        a = self.start("/v1/messages", anthropic("A-request"))
        self.wait_until(lambda: self.engine.active == 1, "A never started")
        b = self.start("/v1/chat/completions", openai("B-request", stream=False))
        self.wait_until(lambda: self.svc.status["queued"] == 1, "B never queued")
        c = self.start("/v1/chat/completions", openai("C-request", stream=False))
        self.wait_until(lambda: self.svc.status["queued"] == 2, "C never queued")
        code, _, body = self.vram({"reserve_mib": 4000, "preempt": True})
        self.assertEqual((code, body["preempt"]["cut"]), (200, 1))      # A only: B and C were waiting, not holding
        for call in (a, b, c):
            self.assertTrue(call.finished.wait(10))
        self.assertEqual([e[0] for e in self.engine.log if e[0] in ("start", "resize")],
                         ["start", "resize", "start", "start", "start"])
        s = self.starts()
        for got, want in zip(s, ("A-request", "A-request", "B-request", "C-request")):
            self.assertIn(want, got)
        self.assertEqual(self.engine.max_active, 1)
        self.assertEqual(self.svc.fifo.waiting(), 0)

    def test_a_cut_request_that_sent_content_goes_to_the_back_when_the_client_sends_it_again(self):
        # A streams, B (slow to start) and C wait; the preempt cuts A only; A's retry arrives after B and C queued
        self.engine.plans[:] = [{"delay": 0.01}, {"lead": 0.5, "text": SHORT, "delay": 0.0},
                                {"text": SHORT, "delay": 0.0}, {"text": SHORT, "delay": 0.0}]
        a = self.start("/v1/messages", anthropic("A-request"))
        a.wait_content(3)
        b = self.start("/v1/chat/completions", openai("B-request", stream=False))
        self.wait_until(lambda: self.svc.status["queued"] == 1, "B never queued")
        c = self.start("/v1/chat/completions", openai("C-request", stream=False))
        self.wait_until(lambda: self.svc.status["queued"] == 2, "C never queued")
        code, _, body = self.vram({"reserve_mib": 4000, "preempt": True})
        self.assertEqual((code, body["preempt"]["cut"]), (200, 1))
        self.assertTrue(a.finished.wait(10))
        self.assertEqual(a.sse()[-1][0], "error")
        self.wait_until(lambda: self.engine.starts >= 2, "B never started")        # B reads its prompt (0.5 s) ...
        a2 = self.start("/v1/messages", anthropic("A-request"))                  # ... while A's retry arrives
        for call in (b, c, a2):
            self.assertTrue(call.finished.wait(10))
        self.assertEqual([e[0] for e in self.engine.log if e[0] in ("start", "resize")],
                         ["start", "resize", "start", "start", "start"])
        for got, want in zip(self.starts(), ("A-request", "B-request", "C-request", "A-request")):
            self.assertIn(want, got)                                             # arrival order: A, B, C, then A again
        self.assertEqual(a2.sse()[-1][0], "message_stop")
        self.assertEqual(self.engine.max_active, 1)
        self.assertEqual(self.svc.preempt_cuts, 1)


if __name__ == "__main__":
    unittest.main()
