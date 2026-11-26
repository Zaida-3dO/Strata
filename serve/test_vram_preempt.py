"""POST /v1/vram with "preempt": true - the resize does not wait for the running request: it takes the front of the line,
cuts the request that holds the engine (which ends with an error a client retries, never as a normal stop) and resizes at
once; waiting requests keep their places.  CPU only: a fake engine with a slow token stream, no model, no GPU.

    python -m unittest serve.test_vram_preempt -v
"""
from __future__ import annotations

import http.client
import importlib.util
import json
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import serve.server as server  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, ClientGone, MockEngine, Service, serve  # noqa: E402
from serve.test_lifecycle import ResidentEngine  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LONG = "word " * 80            # 400 tokens: 4 s at 10 ms a token
SHORT = "ok " * 4


def load_helper():
    spec = importlib.util.spec_from_file_location("strata_retry", ROOT / "tools" / "strata_retry.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["strata_retry"] = module
    spec.loader.exec_module(module)
    return module


class ElasticEngine(ResidentEngine):
    """A resident engine with an elastic cache and a scripted, slow answer.  `plans[i]` is what the i-th generate() call
    does (the last one repeats): {"lead": seconds spent reading the prompt before the first token, "delay": seconds per
    token, "text": the answer}.  It stops at once when the request's cancel Event is set, like the real one after a STOP.
    `log` is the order of things: ("start", prompt), ("end",), ("resize", reserve); `active` counts the generations that
    are inside the engine right now (a resize must never see one)."""

    def __init__(self, tok, plans=None, elastic=True):
        super().__init__(tok)
        self.tok = tok
        self.loaded, self.unloaded, self.info = True, False, {"vram_elastic": elastic}
        self.plans = list(plans or [{}])
        self.active, self.starts, self.yielded, self.fail = 0, 0, 0, None
        self.log, self.resizes = [], []
        self.lock = threading.Lock()

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        with self.lock:
            plan = self.plans[min(self.starts, len(self.plans) - 1)]
            self.starts += 1
            self.active += 1
            self.log.append(("start", self.tok.decode(list(ids))))
        try:
            end = time.monotonic() + plan.get("lead", 0.0)
            while time.monotonic() < end:
                if cancel.is_set():
                    return
                time.sleep(0.005)
            script = MockEngine(self.tok, plan.get("text", LONG), 4096, plan.get("delay", 0.01))
            for t in script.generate(ids, max_new, sampling, cancel):
                self.yielded += 1
                yield t
        finally:
            with self.lock:
                self.active -= 1
                self.log.append(("end",))

    def vram(self, reserve_mib, timeout=120.0):
        with self.lock:
            self.resizes.append({"reserve": reserve_mib, "active": self.active, "yielded": self.yielded})
            self.log.append(("resize", reserve_mib))
        if self.fail:
            raise ValueError(self.fail)
        r = 700 if reserve_mib is None else reserve_mib
        out = {"reserve_mib": r, "expert_slots": 100, "expert_cache_mib": max(500, 6000 - r),
               "expert_cache_full_mib": 6000, "vram_free_mib": r, "prompt_chunk": 4096}
        self.info["vram"] = out
        return out


class Call:
    """One HTTP request on its own thread, recording what arrives as it arrives."""

    def __init__(self, port, path, body, headers=None):
        self.lines, self.status, self.headers = [], None, {}
        self.started, self.finished = threading.Event(), threading.Event()
        self.finished_at = None
        self.thread = threading.Thread(target=self._run, args=(port, path, body, headers or {}), daemon=True)
        self.thread.start()

    def _run(self, port, path, body, headers):
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        try:
            c.request("POST", path, json.dumps(body), {"Content-Type": "application/json", **headers})
            r = c.getresponse()
            self.status, self.headers = r.status, {k.lower(): v for k, v in r.getheaders()}
            self.started.set()
            while True:
                line = r.fp.readline()
                if not line:
                    break
                self.lines.append(line.decode("utf-8"))
        finally:
            c.close()
            self.finished_at = time.monotonic()
            self.finished.set()

    @property
    def text(self):
        return "".join(self.lines)

    def content_events(self):
        return sum(1 for line in list(self.lines) if '"text_delta"' in line or '"content": "' in line and "role" not in line)

    def wait_content(self, n=3, limit=5.0):
        end = time.monotonic() + limit
        while self.content_events() < n and time.monotonic() < end:
            time.sleep(0.005)
        assert self.content_events() >= n, f"only {self.content_events()} content events: {self.text!r}"

    def sse(self):
        """The events of a stream: [(event name or None, parsed data)]."""
        out, name = [], None
        for line in self.text.splitlines():
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                raw = line[5:].strip()
                out.append((name, json.loads(raw) if raw.startswith("{") else raw))
                name = None
        return out

    def json(self):
        return json.loads(self.text)


def anthropic(content="A-request", stream=True, **extra):
    return {"max_tokens": 2000, "stream": stream, "thinking": {"type": "disabled"},
            "messages": [{"role": "user", "content": content}], **extra}


def openai(content="A-request", stream=True, **extra):
    return {"stream": stream, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": content}], **extra}


class PreemptCase(unittest.TestCase):
    plans = None
    elastic = True

    def setUp(self):
        self.tok = ByteTokenizer()
        self.engine = ElasticEngine(self.tok, self.plans, self.elastic)
        self.svc = Service(self.engine, self.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.port = self.httpd.server_address[1]
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)
        self.calls = []

    def start(self, path, body, headers=None):
        call = Call(self.port, path, body, headers)
        self.calls.append(call)
        return call

    def tearDown(self):
        for c in self.calls:                     # nothing may keep running into the next test
            c.finished.wait(10)

    def vram(self, body, timeout=15):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        try:
            c.request("POST", "/v1/vram", json.dumps(body), {"Content-Type": "application/json"})
            r = c.getresponse()
            return r.status, {k.lower(): v for k, v in r.getheaders()}, json.loads(r.read().decode())
        finally:
            c.close()

    def wait_until(self, cond, what, limit=5.0):
        end = time.monotonic() + limit
        while not cond() and time.monotonic() < end:
            time.sleep(0.005)
        self.assertTrue(cond(), what)


class CutRunning(PreemptCase):
    def test_a_running_anthropic_stream_is_cut_and_the_resize_is_at_once(self):
        a = self.start("/v1/messages", anthropic())
        a.wait_content(3)
        t0 = time.monotonic()
        code, _, body = self.vram({"reserve_mib": 4000, "preempt": True})
        took = time.monotonic() - t0
        self.assertEqual(code, 200, body)
        self.assertLess(took, 1.5, "the 4 s answer must not have been waited for")
        self.assertEqual(body["status"], "ok")
        p = body["preempt"]
        self.assertEqual((p["outcome"], p["cut"], p["grace_s"]), ("cut", 1, 0.0))
        self.assertEqual(len(p["cut_ids"]), 1)
        self.assertLessEqual(p["took_ms"], 1500)
        self.assertIn("resize_ms", p)
        # the resize ran with the engine idle: the turn changes hands only between engine steps
        self.assertEqual([(r["reserve"], r["active"]) for r in self.engine.resizes], [(4000, 0)])
        self.assertTrue(a.finished.wait(5))
        events = a.sse()
        name, data = events[-1]
        self.assertEqual(name, "error")                         # the stream ends with Anthropic's error event ...
        self.assertEqual(data["type"], "error")
        err = data["error"]
        self.assertEqual((err["type"], err["code"], err["retry_after_s"]), ("overloaded_error", "vram_preempted", 1))
        self.assertIn("resized by a vram preempt", err["message"])
        self.assertIn("send the same request again after 1 s", err["message"])
        self.assertNotIn("message_stop", [n for n, _ in events])     # ... and never reads as a finished turn
        self.assertNotIn("message_delta", [n for n, _ in events])
        self.assertNotIn("end_turn", a.text)
        self.assertEqual(self.engine.starts, 1, "it had streamed content, so the server must not run it again")
        self.assertEqual(self.svc.preempt_cuts, 1)
        self.assertFalse(self.svc.fifo.locked())
        self.assertFalse(self.svc.status.get("busy"))

    def test_a_running_chat_completions_stream_ends_with_an_error_chunk_not_a_stop(self):
        a = self.start("/v1/chat/completions", openai())
        a.wait_content(3)
        code, _, body = self.vram({"reserve_mib": 4000, "preempt": True})
        self.assertEqual((code, body["preempt"]["cut"]), (200, 1))
        self.assertTrue(a.finished.wait(5))
        events = a.sse()
        self.assertEqual(events[-1][1], "[DONE]")
        err = events[-2][1]["error"]
        self.assertEqual((err["type"], err["code"]), ("server_error", "vram_preempted"))
        self.assertIn("resized by a vram preempt", err["message"])
        for _, chunk in events[:-2]:
            self.assertIsNone(chunk["choices"][0]["finish_reason"], "no chunk may carry a finish_reason")
        self.assertNotIn('"finish_reason": "stop"', a.text)

    def test_a_non_streamed_request_gets_the_retryable_status_and_headers(self):
        for path, body, status, kind in (("/v1/messages", anthropic(stream=False), 529, "overloaded_error"),
                                         ("/v1/chat/completions", openai(stream=False), 503, "server_error")):
            with self.subTest(path=path):
                self.engine.log.clear()
                a = self.start(path, body)
                self.wait_until(lambda: self.engine.active == 1 and self.engine.yielded > 3, "the request never ran")
                code, _, out = self.vram({"reserve_mib": 4000, "preempt": True})
                self.assertEqual((code, out["preempt"]["cut"]), (200, 1))
                self.assertTrue(a.finished.wait(5))
                self.assertEqual(a.status, status)
                self.assertEqual(a.headers.get("retry-after"), "1")
                self.assertEqual(a.headers.get("x-should-retry"), "true")
                err = a.json()["error"]
                self.assertEqual((err["type"], err["code"]), (kind, "vram_preempted"))
                if status == 529:
                    self.assertEqual(a.json()["type"], "error")        # Anthropic's envelope
                self.assertNotIn("stop", a.json().get("stop_reason") or "")
                self.engine.yielded = 0

    def test_the_responses_route_ends_with_response_failed_or_a_503(self):
        body = {"input": "A-request", "stream": True, "reasoning": {"effort": "none"}}
        a = self.start("/v1/responses", body)
        self.wait_until(lambda: self.engine.yielded > 10, "the response never streamed")
        code, _, out = self.vram({"reserve_mib": 4000, "preempt": True})
        self.assertEqual((code, out["preempt"]["cut"]), (200, 1))
        self.assertTrue(a.finished.wait(5))
        events = a.sse()
        name, data = events[-1]
        self.assertEqual(name, "response.failed")
        self.assertEqual(data["response"]["error"]["code"], "vram_preempted")
        self.assertNotIn("response.completed", [n for n, _ in events])
        self.engine.log.clear()
        self.engine.yielded = 0
        b = self.start("/v1/responses", {**body, "stream": False})
        self.wait_until(lambda: self.engine.active == 1 and self.engine.yielded > 5, "the response never ran")
        code, _, out = self.vram({"reserve_mib": 3000, "preempt": True})
        self.assertEqual((code, out["preempt"]["cut"]), (200, 1))
        self.assertTrue(b.finished.wait(5))
        self.assertEqual(b.status, 503)
        self.assertEqual(b.headers.get("retry-after"), "1")
        self.assertEqual(b.json()["error"]["code"], "vram_preempted")

    def test_nothing_running_resizes_at_once(self):
        code, _, body = self.vram({"reserve_mib": 3000, "preempt": True})
        self.assertEqual(code, 200)
        self.assertEqual((body["preempt"]["outcome"], body["preempt"]["cut"], body["preempt"]["cut_ids"]),
                         ("nothing running", 0, []))
        self.assertEqual([(r["reserve"], r["active"]) for r in self.engine.resizes], [(3000, 0)])
        self.assertFalse(self.svc.fifo.locked())
        self.assertEqual(self.svc.preempt_cuts, 0)

    def test_the_reserve_is_remembered_like_a_plain_resize(self):
        _, _, body = self.vram({"reserve_mib": 3000, "preempt": True})
        self.assertIn("preempt", body)
        self.assertEqual((self.svc.vram_reserve, self.svc.vram_reserve_asked), (3000, True))

    def test_an_unloaded_engine_only_records_the_reserve(self):
        self.engine.loaded = False
        code, _, body = self.vram({"reserve_mib": 3000, "preempt": True})
        self.assertEqual((code, body["status"], body["preempt"]["cut"]), (200, "not loaded", 0))
        self.assertEqual(self.engine.resizes, [])


class Refusals(PreemptCase):
    def test_a_non_elastic_engine_is_refused_and_nothing_is_cut(self):
        self.elastic = False
        self.engine.info["vram_elastic"] = False
        a = self.start("/v1/messages", anthropic(content="x", stream=True))
        a.wait_content(2)
        code, _, body = self.vram({"reserve_mib": 3000, "preempt": True})
        self.assertEqual(code, 400)
        self.assertEqual(body["error"]["code"], "vram_not_elastic")
        self.assertIn("--vram-elastic", body["error"]["message"])
        self.assertIn("nothing was cut", body["error"]["message"])
        self.assertEqual(self.engine.resizes, [])
        self.assertEqual(self.svc.preempt_cuts, 0)
        self.assertTrue(a.finished.wait(10))          # the request ran to its end
        names = [n for n, _ in a.sse()]
        self.assertEqual(names[-1], "message_stop")
        self.assertNotIn("error", names)

    def test_preempt_must_be_a_boolean_and_grace_a_number(self):
        a = self.start("/v1/messages", anthropic())
        a.wait_content(2)
        for bad in ({"preempt": "yes"}, {"preempt": 1}, {"preempt": True, "preempt_grace_s": -1},
                    {"preempt": True, "preempt_grace_s": "5"}, {"preempt": True, "preempt_grace_s": True},
                    {"preempt": True, "preempt_grace_s": 9999}, {"preempt_grace_s": 5}):
            with self.subTest(bad=bad):
                code, _, body = self.vram({"reserve_mib": 1000, **bad})
                self.assertEqual(code, 400, body)
        self.assertEqual(self.engine.resizes, [])
        self.assertEqual(self.svc.preempt_cuts, 0)
        self.assertEqual(a.finished.is_set(), False)                  # still streaming, untouched

    def test_a_turn_that_cannot_be_cut_times_out_with_409_and_leaves_the_line_clean(self):
        self.svc.vram_wait_s = 0.3
        self.svc.fifo.acquire()                  # a session save or an engine load holds the turn: not a request
        try:
            code, _, body = self.vram({"reserve_mib": 1000, "preempt": True})
        finally:
            self.svc.fifo.release()
        self.assertEqual(code, 409)
        self.assertEqual(body["error"]["type"], "model_busy")
        self.assertEqual(body["error"]["cut"], 0)
        self.assertEqual(self.svc.fifo.waiting(), 0)
        self.assertEqual(self.engine.resizes, [])
        self.assertFalse(self.svc.fifo.locked())


class ResizeFailsAfterTheCut(PreemptCase):
    def test_the_reserve_stays_and_the_server_keeps_working(self):
        a = self.start("/v1/messages", anthropic())
        a.wait_content(3)
        self.engine.fail = "ERR VRAM: the engine said no"
        code, _, body = self.vram({"reserve_mib": 4000, "preempt": True})
        self.assertEqual(code, 409)
        err = body["error"]
        self.assertEqual((err["type"], err["cut"], err["reserve_unchanged"]), ("vram_resize_failed", 1, True))
        self.assertIn("the engine said no", err["message"])
        self.assertEqual((self.svc.vram_reserve, self.svc.vram_reserve_asked), (None, False))
        self.assertTrue(a.finished.wait(5))
        self.assertEqual(a.sse()[-1][1]["error"]["code"], "vram_preempted")      # the cut request can retry
        self.assertFalse(self.svc.fifo.locked())
        self.assertFalse(self.svc.status.get("busy"))
        self.engine.fail = None
        self.engine.plans = [{"text": SHORT, "delay": 0.0}]
        self.engine.starts = 0
        b = self.start("/v1/messages", anthropic(content="B-request"))      # the next request runs normally
        self.assertTrue(b.finished.wait(5))
        self.assertEqual(b.sse()[-1][0], "message_stop")
        code, _, body = self.vram({"reserve_mib": 4000, "preempt": True})
        self.assertEqual(code, 200)
        self.assertEqual(self.svc.vram_reserve, 4000)


class WaitingRequestsKeepTheirPlace(PreemptCase):
    plans = [{}, {"text": SHORT, "delay": 0.0}]          # the first request is long, the ones behind it short

    def test_waiting_requests_run_after_the_resize_in_order_and_are_not_cut(self):
        a = self.start("/v1/messages", anthropic("A-request"))
        a.wait_content(3)
        b = self.start("/v1/chat/completions", openai("B-request", stream=False))
        self.wait_until(lambda: self.svc.status["queued"] == 1, "B never queued")
        c = self.start("/v1/chat/completions", openai("C-request", stream=False))
        self.wait_until(lambda: self.svc.status["queued"] == 2, "C never queued")
        code, _, body = self.vram({"reserve_mib": 4000, "preempt": True})
        self.assertEqual((code, body["preempt"]["cut"]), (200, 1))
        for call in (a, b, c):
            self.assertTrue(call.finished.wait(10))
        self.assertEqual(a.sse()[-1][1]["error"]["code"], "vram_preempted")
        for call in (b, c):                                   # they were never cut: normal answers, a normal stop
            self.assertEqual(call.status, 200, call.text)
            self.assertEqual(call.json()["choices"][0]["finish_reason"], "stop")
        order = [e[0] for e in self.engine.log if e[0] in ("start", "resize")]
        self.assertEqual(order, ["start", "resize", "start", "start"])                   # A, the resize, B, C
        starts = [e[1] for e in self.engine.log if e[0] == "start"]
        self.assertIn("A-request", starts[0])
        self.assertIn("B-request", starts[1])
        self.assertIn("C-request", starts[2])
        self.assertEqual(self.svc.preempt_cuts, 1)


class RunAgain(PreemptCase):
    plans = [{"lead": 5.0}, {"text": SHORT, "delay": 0.0}]     # the first try is still reading its prompt when cut

    def test_a_request_cut_before_it_sent_anything_is_run_again_by_the_server(self):
        a = self.start("/v1/messages", anthropic())
        self.wait_until(lambda: self.engine.active == 1, "A never started")
        b = self.start("/v1/chat/completions", openai("B-request", stream=False))
        self.wait_until(lambda: self.svc.status["queued"] == 1, "B never queued")
        code, _, body = self.vram({"reserve_mib": 4000, "preempt": True})
        self.assertEqual((code, body["preempt"]["cut"]), (200, 1))
        self.assertTrue(a.finished.wait(10))
        self.assertTrue(b.finished.wait(10))
        names = [n for n, _ in a.sse()]
        self.assertEqual(names[-1], "message_stop")                   # a complete answer; the client saw no error
        self.assertNotIn("error", names)
        self.assertEqual(a.status, 200)
        self.assertEqual(self.svc.preempt_requeues, 1)
        # A, the resize, A again (ahead of B, which had been waiting behind it), then B
        self.assertEqual([e[0] for e in self.engine.log if e[0] in ("start", "resize")],
                         ["start", "resize", "start", "start"])
        starts = [e[1] for e in self.engine.log if e[0] == "start"]
        self.assertIn("A-request", starts[0])
        self.assertIn("A-request", starts[1])
        self.assertIn("B-request", starts[2])
        self.assertEqual(b.json()["choices"][0]["finish_reason"], "stop")
        self.assertEqual([(r["reserve"], r["active"]) for r in self.engine.resizes], [(4000, 0)])

    def test_it_gets_the_error_when_the_server_may_not_run_it_again(self):
        with mock.patch.object(server, "PREEMPT_REQUEUE_MAX", 0):
            a = self.start("/v1/messages", anthropic())
            self.wait_until(lambda: self.engine.active == 1, "A never started")
            code, _, body = self.vram({"reserve_mib": 4000, "preempt": True})
            self.assertEqual((code, body["preempt"]["cut"]), (200, 1))
            self.assertTrue(a.finished.wait(10))
        events = a.sse()                                              # the stream had begun: an error event, no content
        self.assertEqual([n for n, _ in events], ["message_start", "error"])
        self.assertEqual(events[-1][1]["error"]["type"], "overloaded_error")
        self.assertEqual(self.svc.preempt_requeues, 0)
        self.assertEqual(self.engine.starts, 1)


class WithoutPreempt(PreemptCase):
    plans = [{"text": "word " * 30, "delay": 0.01}]          # 150 tokens: about 1.5 s

    def test_a_plain_resize_still_waits_for_the_running_request(self):
        a = self.start("/v1/messages", anthropic())
        a.wait_content(3)
        for body in ({"reserve_mib": 4000}, ):
            code, _, out = self.vram(body)
            done = time.monotonic()
            self.assertEqual(code, 200)
            self.assertNotIn("preempt", out)
            self.assertTrue(a.finished.wait(5))
            self.assertEqual(a.sse()[-1][0], "message_stop")           # a normal end: nothing was cut
            self.assertNotIn("error", [n for n, _ in a.sse()])
        # the engine had produced the whole answer (150 tokens + the end token) when the resize ran
        self.assertEqual([(r["reserve"], r["active"], r["yielded"]) for r in self.engine.resizes],
                         [(4000, 0, 151)])
        self.assertEqual([e[0] for e in self.engine.log], ["start", "end", "resize"])
        self.assertEqual(self.svc.preempt_cuts, 0)

    def test_preempt_false_is_the_plain_resize(self):
        a = self.start("/v1/messages", anthropic())
        a.wait_content(3)
        code, _, out = self.vram({"reserve_mib": 4000, "preempt": False})
        self.assertEqual(code, 200)
        self.assertNotIn("preempt", out)
        self.assertEqual([e[0] for e in self.engine.log], ["start", "end", "resize"])
        self.assertTrue(a.finished.wait(5))
        self.assertEqual(a.sse()[-1][0], "message_stop")

    def test_a_plain_resize_still_times_out_with_409(self):
        self.svc.vram_wait_s = 0.2
        with self.svc.fifo:
            code, _, body = self.vram({"reserve_mib": 1})
        self.assertEqual((code, body["error"]["type"]), (409, "model_busy"))
        self.assertNotIn("cut", body["error"])


class Grace(PreemptCase):
    plans = [{"text": "word " * 60, "delay": 0.01}]          # 300 tokens: about 3 s

    def test_a_streaming_request_that_ends_within_the_grace_is_not_cut(self):
        a = self.start("/v1/messages", anthropic())
        a.wait_content(3)
        code, _, body = self.vram({"reserve_mib": 4000, "preempt": True, "preempt_grace_s": 10})
        self.assertEqual(code, 200)
        p = body["preempt"]
        self.assertEqual((p["outcome"], p["cut"], p["grace_s"]), ("waited", 0, 10.0))
        self.assertGreater(p["turn_wait_ms"], 500)                    # it waited for the answer to end
        self.assertLess(p["turn_wait_ms"], 8000)
        self.assertTrue(a.finished.wait(5))
        self.assertEqual(a.sse()[-1][0], "message_stop")              # a clean end
        self.assertNotIn("error", [n for n, _ in a.sse()])
        self.assertEqual([e[0] for e in self.engine.log], ["start", "end", "resize"])
        self.assertEqual([(r["reserve"], r["active"]) for r in self.engine.resizes], [(4000, 0)])

    def test_a_streaming_request_still_running_after_the_grace_is_cut(self):
        a = self.start("/v1/messages", anthropic())
        a.wait_content(3)
        code, _, body = self.vram({"reserve_mib": 4000, "preempt": True, "preempt_grace_s": 0.3})
        self.assertEqual(code, 200)
        p = body["preempt"]
        self.assertEqual((p["outcome"], p["cut"]), ("cut", 1))
        self.assertGreaterEqual(p["turn_wait_ms"], 280)               # it did wait the grace first
        self.assertLess(p["turn_wait_ms"], 2000)
        self.assertTrue(a.finished.wait(5))
        self.assertEqual(a.sse()[-1][1]["error"]["code"], "vram_preempted")

    def test_a_request_that_has_only_thought_so_far_is_cut_at_once_despite_the_grace(self):
        self.engine.plans = [{"text": "thinking about it " * 30, "delay": 0.01}]     # no </think>: all reasoning
        a = self.start("/v1/messages", anthropic(thinking={"type": "enabled", "budget_tokens": 1000}))
        self.wait_until(lambda: self.engine.yielded > 10, "no thinking streamed")
        self.assertIn("thinking_delta", a.text)
        t0 = time.monotonic()
        code, _, body = self.vram({"reserve_mib": 4000, "preempt": True, "preempt_grace_s": 30})
        self.assertEqual(code, 200)
        self.assertLess(time.monotonic() - t0, 3, "only thinking had been sent: no reason to wait for the grace")
        self.assertEqual((body["preempt"]["outcome"], body["preempt"]["cut"]), ("cut", 1))
        self.assertTrue(a.finished.wait(5))
        self.assertEqual(a.sse()[-1][1]["error"]["type"], "overloaded_error")    # Claude Code retries this by itself
        self.assertEqual(self.engine.starts, 1)                  # thinking went out, so the server cannot run it again

    def test_a_request_that_has_sent_nothing_is_cut_at_once_despite_the_grace(self):
        self.engine.plans = [{"lead": 5.0}, {"text": SHORT, "delay": 0.0}]
        a = self.start("/v1/messages", anthropic())
        self.wait_until(lambda: self.engine.active == 1, "A never started")
        t0 = time.monotonic()
        code, _, body = self.vram({"reserve_mib": 4000, "preempt": True, "preempt_grace_s": 30})
        self.assertEqual(code, 200)
        self.assertLess(time.monotonic() - t0, 3, "nothing had been sent: no reason to wait for the grace")
        self.assertEqual((body["preempt"]["outcome"], body["preempt"]["cut"]), ("cut", 1))
        self.assertTrue(a.finished.wait(10))
        self.assertEqual(a.sse()[-1][0], "message_stop")                # run again by the server, answered normally
        self.assertEqual(self.svc.preempt_requeues, 1)


class TwoPreempts(PreemptCase):
    def test_two_preempts_behind_a_turn_nobody_can_cut_run_in_arrival_order_ahead_of_a_waiting_request(self):
        self.svc.fifo.acquire()                  # a session save holds the turn
        results = []
        a = self.start("/v1/chat/completions", openai("A-request", stream=False))   # waits first ...
        self.wait_until(lambda: self.svc.status["queued"] == 1, "a request never queued")

        def go(reserve):
            results.append((reserve, self.vram({"reserve_mib": reserve, "preempt": True})))
        t1 = threading.Thread(target=go, args=(1000,))
        t1.start()
        self.wait_until(lambda: self.svc.fifo.waiting() == 2, "the first preempt never queued")
        t2 = threading.Thread(target=go, args=(2000,))
        t2.start()
        self.wait_until(lambda: self.svc.fifo.waiting() == 3, "the second preempt never queued")    # ... they jump it
        self.plans = [{"text": SHORT, "delay": 0.0}]
        self.engine.plans = self.plans
        self.svc.fifo.release()
        t1.join(10)
        t2.join(10)
        self.assertTrue(a.finished.wait(10))
        self.assertEqual([r["reserve"] for r in self.engine.resizes], [1000, 2000])
        self.assertEqual(sorted(code for _, (code, _, _) in results), [200, 200])
        self.assertEqual(a.status, 200)                                  # it ran after both: not cut, not skipped
        self.assertEqual([e[0] for e in self.engine.log if e[0] in ("start", "resize")], ["resize", "resize", "start"])
        self.assertEqual(a.json()["choices"][0]["finish_reason"], "stop")

    def test_two_preempts_against_a_running_request_cut_it_once(self):
        a = self.start("/v1/messages", anthropic())
        a.wait_content(3)
        results = []

        def go(reserve):
            results.append(self.vram({"reserve_mib": reserve, "preempt": True}))
        threads = [threading.Thread(target=go, args=(r,)) for r in (1000, 2000)]
        for t in threads:
            t.start()
            time.sleep(0.01)
        for t in threads:
            t.join(10)
        self.assertEqual(sorted(code for code, _, _ in results), [200, 200])
        self.assertEqual(sum(body["preempt"]["cut"] for _, _, body in results), 1)
        self.assertEqual(len(self.engine.resizes), 2)
        self.assertTrue(all(r["active"] == 0 for r in self.engine.resizes))
        self.assertTrue(a.finished.wait(5))
        self.assertFalse(self.svc.fifo.locked())


class ImageEncodeHoldsTheTurn(PreemptCase):
    def test_an_encode_is_cut_and_the_route_answers_the_retryable_error(self):
        held = threading.Event()
        released = []

        def prepare(messages, tools, kwargs, max_new=None, force=None, req=None, cancel=None):
            with self.svc._turn(cancel, cuttable=True):          # what an image encode does
                held.set()
                while not cancel.is_set():
                    time.sleep(0.005)
                released.append(True)
                raise ClientGone()                               # what encode_all raises between pictures
        with mock.patch.object(self.svc, "prepare", prepare):
            a = self.start("/v1/messages", anthropic(stream=False))
            self.assertTrue(held.wait(5))
            self.wait_until(lambda: self.svc.fifo.locked(), "the encode never held the turn")
            code, _, body = self.vram({"reserve_mib": 4000, "preempt": True})
            self.assertTrue(a.finished.wait(5))
        self.assertEqual((code, body["preempt"]["cut"], body["preempt"]["cut_ids"]), (200, 1, ["image-encode"]))
        self.assertEqual([(r["reserve"], r["active"]) for r in self.engine.resizes], [(4000, 0)])
        self.assertEqual(a.status, 529)
        self.assertEqual(a.json()["error"]["code"], "vram_preempted")
        self.assertEqual(a.headers.get("retry-after"), "1")
        self.assertEqual(released, [True])

    def test_a_client_that_left_is_not_turned_into_an_error_for_nobody(self):
        cancel = threading.Event()
        who = server.Holder("r1")
        self.svc.hold(cancel, who)
        self.svc.note_client_disconnect(cancel)                  # the client left first ...
        self.assertTrue(cancel.is_set())
        self.assertEqual(self.svc.preempt_cut(), [])             # ... so the preempt does not claim the cut
        self.assertIsNone(server.preempt_reason(cancel))
        late = threading.Event()
        self.svc.hold(late, server.Holder("r2"))
        self.assertEqual(self.svc.preempt_cut(), ["r2"])         # a preempt first, then the client leaves:
        self.assertIsNotNone(server.preempt_reason(late))
        self.svc.note_client_disconnect(late)
        self.assertIsNone(server.preempt_reason(late))           # the goodbye wins: nothing to answer


class ParallelSlots(PreemptCase):
    """--parallel: requests do not hold the turn; the engine's control lines are taken and it refuses a resize while a
    slot still decodes.  The preempt cuts every running request and retries the resize for a moment."""

    def test_it_cuts_until_the_control_lines_are_free_and_retries_the_refusal(self):
        self.engine.batch, self.engine.ctl = 2, threading.Lock()
        refusals = ["VRAM: not while batch slots are decoding"] * 2
        real = self.engine.vram

        def vram(reserve_mib, timeout=120.0):
            if refusals:
                self.engine.resizes.append({"reserve": reserve_mib, "active": 0, "yielded": 0})
                raise ValueError(refusals.pop())
            return real(reserve_mib, timeout)
        self.engine.vram = vram
        running = threading.Event()
        self.svc.hold(running, server.Holder("slot-1"))          # a request running in a slot
        self.engine.ctl.acquire()                                  # a prompt is being admitted
        threading.Timer(0.15, self.engine.ctl.release).start()
        result = self.svc.vram(4000, preempt=True)
        self.assertEqual(result["preempt"]["cut"], 1)
        self.assertEqual(result["preempt"]["cut_ids"], ["slot-1"])
        self.assertTrue(running.is_set())
        self.assertIsNotNone(server.preempt_reason(running))
        self.assertEqual(len(self.engine.resizes), 3)             # refused twice, then done
        self.assertEqual(result["status"], "ok")
        self.assertFalse(self.engine.ctl.locked())
        self.assertFalse(self.svc.fifo.locked())

    def test_a_refusal_that_does_not_go_away_is_vram_resize_failed(self):
        self.engine.batch, self.engine.ctl = 2, threading.Lock()
        self.engine.fail = "VRAM: not while batch slots are decoding"
        with mock.patch.object(server, "PREEMPT_POLL_S", 0.001):
            with self.assertRaises(server.VramResizeFailed) as raised:
                self.svc.vram(4000, preempt=True)
        self.assertEqual(raised.exception.cut, 0)
        self.assertEqual(len(self.engine.resizes), 50)
        self.assertFalse(self.engine.ctl.locked())
        self.assertFalse(self.svc.fifo.locked())


class RetryHelperEndToEnd(PreemptCase):
    plans = [{}, {"text": SHORT, "delay": 0.0}]          # the first try is cut mid-answer, the retry is quick

    def test_the_helper_sends_the_cut_request_again_and_gets_the_answer(self):
        helper = load_helper()
        slept, retried, events = [], [], []
        result = {}

        def go():
            result["reply"] = helper.post_with_retry(
                f"http://127.0.0.1:{self.port}/v1/messages", anthropic(), stream=True, sleep=slept.append,
                on_retry=lambda attempt, why: retried.append((attempt, why)),
                on_event=lambda name, data: events.append(name))
        t = threading.Thread(target=go)
        t.start()
        self.wait_until(lambda: self.engine.yielded > 5, "the first attempt never streamed")
        code, _, body = self.vram({"reserve_mib": 4000, "preempt": True})
        self.assertEqual((code, body["preempt"]["cut"]), (200, 1))
        t.join(15)
        self.assertFalse(t.is_alive())
        reply = result["reply"]
        self.assertTrue(reply.ok, reply)
        self.assertEqual((reply.status, reply.attempts), (200, 2))
        self.assertEqual(slept, [1.0])                                     # the server's retry_after_s
        self.assertEqual(len(retried), 1)
        self.assertIn("vram preempt", retried[0][1])
        self.assertEqual(reply.events[-1][0], "message_stop")
        self.assertEqual([e[0] for e in self.engine.log if e[0] in ("start", "resize")],
                         ["start", "resize", "start"])


class RetryHelper(unittest.TestCase):
    """tools/strata_retry.py against a fake server that answers from a script (no Strata)."""

    def setUp(self):
        self.helper = load_helper()
        self.script, self.seen = [], []
        test = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *a):
                pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                test.seen.append(json.loads(body))
                status, headers, payload = test.script[min(len(test.seen), len(test.script)) - 1]
                self.send_response(status)
                for k, v in headers.items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(payload if isinstance(payload, bytes) else payload.encode())

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1/messages"

    CUT = (529, {"Content-Type": "application/json", "Retry-After": "2", "x-should-retry": "true"},
           json.dumps({"type": "error", "error": {"type": "overloaded_error", "code": "vram_preempted",
                                                  "message": "cut", "retry_after_s": 1}}))
    OK = (200, {"Content-Type": "application/json"}, json.dumps({"type": "message", "content": []}))

    def sse(self, *events):
        text = "".join(f"event: {n}\ndata: {json.dumps(d)}\n\n" for n, d in events)
        return (200, {"Content-Type": "text/event-stream"}, text)

    def test_a_cut_request_is_sent_again_after_the_retry_after_and_succeeds(self):
        self.script = [self.CUT, self.CUT, self.OK]
        slept = []
        reply = self.helper.post_with_retry(self.url, {"q": 1}, sleep=slept.append)
        self.assertEqual((reply.status, reply.attempts, reply.ok), (200, 3, True))
        self.assertEqual(slept, [2.0, 2.0])                        # the Retry-After header wins over the body
        self.assertEqual(self.seen, [{"q": 1}] * 3)                # the same request every time
        self.assertEqual(reply.waited_s, 4.0)

    def test_an_error_event_in_a_stream_is_retried_and_the_partial_is_dropped(self):
        cut = self.sse(("message_start", {"type": "message_start"}),
                       ("error", {"type": "error", "error": {"type": "overloaded_error", "code": "vram_preempted",
                                                             "message": "cut", "retry_after_s": 1}}))
        ok = self.sse(("message_start", {"type": "message_start"}), ("message_stop", {"type": "message_stop"}))
        self.script = [cut, ok]
        slept, retried = [], []
        reply = self.helper.post_with_retry(self.url, {"stream": True}, stream=True, sleep=slept.append,
                                            on_retry=lambda n, why: retried.append(n))
        self.assertEqual((reply.status, reply.attempts, reply.ok), (200, 2, True))
        self.assertEqual([e[0] for e in reply.events], ["message_start", "message_stop"])
        self.assertEqual((slept, retried), ([1.0], [1]))

    def test_an_openai_error_chunk_is_retried_too(self):
        err = {"error": {"type": "server_error", "code": "vram_preempted", "message": "cut", "retry_after_s": 1}}
        cut = (200, {"Content-Type": "text/event-stream"}, f"data: {json.dumps(err)}\n\ndata: [DONE]\n\n")
        ok = (200, {"Content-Type": "text/event-stream"}, 'data: {"choices": []}\n\ndata: [DONE]\n\n')
        self.script = [cut, ok]
        reply = self.helper.post_with_retry(self.url, {}, stream=True, sleep=lambda s: None)
        self.assertEqual((reply.attempts, reply.ok), (2, True))

    def test_other_errors_are_returned_not_retried(self):
        for status, body in ((400, {"error": {"type": "invalid_request_error", "message": "bad"}}),
                             (409, {"error": {"type": "model_busy", "message": "busy"}}),
                             (500, {"error": {"type": "server_error", "message": "boom"}}),
                             (503, {"error": {"type": "server_error", "message": "engine died"}})):
            with self.subTest(status=status):
                self.seen.clear()
                self.script = [(status, {"Content-Type": "application/json"}, json.dumps(body)), self.OK]
                slept = []
                reply = self.helper.post_with_retry(self.url, {}, sleep=slept.append)
                self.assertEqual((reply.status, reply.attempts, slept, reply.ok), (status, 1, [], False))
                self.assertEqual(len(self.seen), 1)

    def test_it_gives_up_after_the_attempts_and_says_so(self):
        self.script = [self.CUT]
        slept = []
        reply = self.helper.post_with_retry(self.url, {}, attempts=3, sleep=slept.append)
        self.assertEqual((reply.status, reply.attempts, reply.ok), (529, 3, False))
        self.assertEqual(len(slept), 2)
        self.assertEqual(self.helper.error_of(reply)["code"], "vram_preempted")

    def test_a_long_retry_after_is_capped(self):
        self.script = [(529, {"Content-Type": "application/json", "Retry-After": "600"}, self.CUT[2]), self.OK]
        slept = []
        self.helper.post_with_retry(self.url, {}, max_wait_s=5, sleep=slept.append)
        self.assertEqual(slept, [5])

    def test_the_command_line_prints_the_answer_and_returns_zero(self):
        self.script = [self.CUT, self.OK]
        out = []
        with mock.patch.object(self.helper.time, "sleep", lambda s: None), \
                mock.patch.object(sys, "stdout") as stdout:
            stdout.write.side_effect = out.append
            code = self.helper.main([self.url, "--body", '{"q": 2}', "-H", "x-api-key: k", "--max-wait", "0"])
        self.assertEqual(code, 0)
        self.assertEqual(self.seen, [{"q": 2}, {"q": 2}])
        self.assertIn('"type": "message"', "".join(out))


if __name__ == "__main__":
    unittest.main()
