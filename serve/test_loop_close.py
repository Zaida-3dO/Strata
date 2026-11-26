"""serve/test_loop_close.py - a loop of one token inside the thinking is closed with the wrap-up, not ended empty (mock engine).

Flash-Next sometimes writes one token (a "0") for ever while it thinks; the 256-in-a-row guard (#606) then ended the reply as
"length" with no answer.  Now the thinking is closed once the way a thinking budget closes it (REASONING_WRAP_UP) and the
model answers.  A loop in the answer, or a second loop in the thinking, still ends the reply as before.  CPU only:

    python -m unittest serve.test_loop_close -v
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import threading
import unittest
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve import server as _server  # noqa: E402
from serve.server import REASONING_WRAP_UP, REPEAT_STOP_TOKENS, ByteTokenizer, MockEngine, Service, serve  # noqa: E402

REPEAT_CLOSE_KEEP = getattr(_server, "REPEAT_CLOSE_KEEP", 8)     # (absent before the fix, so the control run still runs)

ROOT = Path(__file__).resolve().parents[1]
ANSWER = "The answer is 42."


class Recording(MockEngine):
    """Keeps every prompt the server sent, one per pass."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.prompts = []

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.prompts.append(list(ids))
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)


class LoopClose(unittest.TestCase):
    def setUp(self):
        self.tok = ByteTokenizer()
        self.ids = self.tok.encode("hi")

    def run_reply(self, scripts, thinking=True, max_new=3000, budget=None):
        engine = Recording(self.tok, scripts, max_context=100000)
        svc = Service(engine, self.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        svc.answer_room = False                     # the cases here are about the repeated-token guard alone
        sampling = {"reasoning_budget_tokens": budget} if budget else {}
        with contextlib.redirect_stdout(io.StringIO()) as out:
            events = list(svc.run(self.ids, thinking, None, max_new, sampling, threading.Event()))
        evs = [e for kind, e in events if kind == "event"]
        text = lambda k: "".join(e.text or "" for e in evs if e.kind == k)
        done = [x for kind, x in events if kind == "done"][0]
        return engine, done, text("reasoning"), text("content"), out.getvalue()

    def test_a_loop_in_the_thinking_is_closed_and_the_model_answers(self):
        engine, done, reasoning, content, log = self.run_reply(
            ["Let me think. " + "0" * 600, ANSWER])
        self.assertEqual(content, ANSWER)
        self.assertEqual(done["finish"], "stop")                    # a normal finish: not "length", not an error
        self.assertEqual(len(engine.prompts), 2)                    # one pass to the loop, one to the answer
        self.assertIn("wrapping up the thinking", log)
        self.assertNotIn("ended as \"length\"", log)
        self.assertTrue(reasoning.endswith(REASONING_WRAP_UP.split("</think>")[0]))

    def test_the_run_is_trimmed_before_it_goes_back_into_the_context(self):
        engine, done, reasoning, content, log = self.run_reply(["Let me think. " + "0" * 600, ANSWER])
        first, second = engine.prompts
        lead = self.tok.encode("Let me think. ")
        wrap = self.tok.encode(REASONING_WRAP_UP, parse_special=True)
        self.assertEqual(second, first + lead + self.tok.encode("0") * REPEAT_CLOSE_KEEP + wrap)
        self.assertEqual(second.count(self.tok.encode("0")[0]), REPEAT_CLOSE_KEEP)   # not the 256 the engine wrote
        # the tokens the engine wrote still count against max_tokens
        self.assertEqual(done["completion_tokens"], len(lead) + REPEAT_STOP_TOKENS + len(wrap) + len(ANSWER) + 1)

    def test_a_loop_in_the_answer_still_ends_the_reply(self):
        engine, done, reasoning, content, log = self.run_reply(
            ["Short thought.\n</think>\n\n" + "!" * 600])
        self.assertEqual(done["finish"], "length")
        self.assertEqual(len(engine.prompts), 1)                    # no wrap-up pass
        self.assertIn("repeated one token ('!') 256 times", log)
        self.assertNotIn("wrapping up the thinking", log)

    def test_a_loop_in_the_answer_after_the_wrap_up_still_ends_the_reply(self):
        engine, done, reasoning, content, log = self.run_reply(["Let me think. " + "0" * 600, "ok " + "!" * 600])
        self.assertEqual(done["finish"], "length")
        self.assertEqual(len(engine.prompts), 2)
        self.assertEqual(log.count("wrapping up the thinking"), 1)
        self.assertIn("repeated one token ('!') 256 times", log)
        self.assertTrue(content.startswith("ok !"))                 # what the model wrote before the loop reached the client

    def test_a_normal_long_thinking_is_untouched(self):
        thought = "".join(f"step {i} " for i in range(200)) + "=" * 255        # long, and a run one short of the limit
        engine, done, reasoning, content, log = self.run_reply([thought + "\n</think>\n\n" + ANSWER])
        self.assertEqual((done["finish"], content), ("stop", ANSWER))
        self.assertEqual(len(engine.prompts), 1)
        self.assertEqual(reasoning.strip(), thought)
        self.assertNotIn("wrapping up", log)

    def test_a_second_loop_in_the_thinking_after_the_wrap_up_stops_cleanly(self):
        # the wrap-up closed the thinking, but the model opens another thinking block and loops in it again
        engine, done, reasoning, content, log = self.run_reply(
            ["Let me think. " + "0" * 600, "hm<think>\n" + "1" * 600])
        self.assertEqual(len(engine.prompts), 2)                    # never a third pass
        self.assertEqual(log.count("wrapping up the thinking"), 1)
        self.assertEqual(done["finish"], "length")
        self.assertIn("repeated one token ('1') 256 times", log)

    def test_the_budget_wrap_up_and_the_loop_close_each_fire_once(self):
        # a thinking budget that is reached first closes the thinking; a later run is then in the answer
        engine, done, reasoning, content, log = self.run_reply(
            ["x" * 40 + "0" * 600, "fine " + "0" * 300], budget=20)
        self.assertEqual(log.count("thinking budget reached"), 1)
        self.assertNotIn("the thinking repeated one token", log)
        self.assertEqual(done["finish"], "length")                  # the answer's own loop ends as before

    def test_off_when_repeat_stop_tokens_is_zero(self):
        engine = Recording(self.tok, ["Let me think. " + "0" * 600], max_context=100000)
        svc = Service(engine, self.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        svc.answer_room = False
        svc.repeat_stop_tokens = 0
        with contextlib.redirect_stdout(io.StringIO()):
            done = [x for kind, x in svc.run(self.ids, True, None, 3000, {}, threading.Event()) if kind == "done"][0]
        self.assertEqual(len(engine.prompts), 1)

    def test_no_room_to_answer_ends_as_before(self):
        # the cap is so close that the wrap-up does not fit: "length", and the usual message
        engine, done, reasoning, content, log = self.run_reply(["Let me think. " + "0" * 600, ANSWER], max_new=270)
        self.assertEqual(done["finish"], "length")
        self.assertEqual(len(engine.prompts), 1)
        self.assertIn("repeated one token ('0') 256 times", log)

    def test_a_stream_consumer_sees_a_normal_finish(self):
        engine = Recording(self.tok, ["<think>\nLet me think. " + "0" * 600 + "", ANSWER], max_context=100000)
        svc = Service(engine, self.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        svc.answer_room = False
        httpd = serve(svc, port=0)
        try:
            body = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 3000, "stream": True}
            req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_address[1]}/v1/chat/completions",
                                         data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            with contextlib.redirect_stdout(io.StringIO()):
                raw = urllib.request.urlopen(req, timeout=30).read().decode()
        finally:
            httpd.shutdown()
            httpd.server_close()
        chunks = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: ") and "[DONE]" not in line]
        self.assertFalse(any("error" in c for c in chunks), raw[-400:])
        self.assertEqual([c["choices"][0]["finish_reason"] for c in chunks if c["choices"] and c["choices"][0].get("finish_reason")],
                         ["stop"])
        answer = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks if c["choices"])
        self.assertEqual(answer, ANSWER)


if __name__ == "__main__":
    unittest.main()
