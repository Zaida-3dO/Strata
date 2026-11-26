"""Reviewer's extra tests for the loop close (not on the builder's branch).  CPU only, mock engine."""
from __future__ import annotations
import contextlib, io, json, sys, threading, unittest, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate
from serve.server import REASONING_WRAP_UP, ByteTokenizer, MockEngine, Service, serve

ROOT = Path(__file__).resolve().parents[1]
ANSWER = "The answer is 42."
CALL = ("<tool_call>\n<function=write>\n<parameter=path>\na.md\n</parameter>\n<parameter=content>\nhi\n</parameter>\n"
        "</function>\n</tool_call>")
TOOLS = [{"name": "write", "parameters": {"type": "object", "properties": {"path": {"type": "string"},
                                                                          "content": {"type": "string"}}}}]


class Rec(MockEngine):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw); self.prompts = []
    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.prompts.append(list(ids)); self.maxes = getattr(self, "maxes", []) + [max_new]
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)


class Review(unittest.TestCase):
    def setUp(self):
        self.tok = ByteTokenizer(); self.ids = self.tok.encode("hi")

    def svc(self, scripts, answer_room=False):
        e = Rec(self.tok, scripts, max_context=100000)
        s = Service(e, self.tok, ChatTemplate(ROOT / "serve/chat_template.jinja")); s.answer_room = answer_room
        return e, s

    def run_it(self, s, thinking=True, tools=None, max_new=3000, sampling=None, **kw):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            ev = list(s.run(self.ids, thinking, tools, max_new, sampling or {}, threading.Event(), **kw))
        return ev, out.getvalue()

    def test_answer_room_on_loop_before_budget(self):
        e, s = self.svc(["Let me think. " + "0" * 600, ANSWER], answer_room=True)
        ev, log = self.run_it(s, max_new=3000)       # budget = 3000 - 750 = 2250; loop at ~270
        done = [x for k, x in ev if k == "done"][0]
        self.assertEqual(done["finish"], "stop"); self.assertEqual(len(e.prompts), 2)
        self.assertIn("wrapping up the thinking so the model answers", log)
        self.assertNotIn("thinking budget reached", log)
        self.assertEqual("".join(x.text or "" for k, x in ev if k == "event" and x.kind == "content"), ANSWER)

    def test_tool_call_after_wrap_up(self):
        e, s = self.svc(["Let me think. " + "0" * 600, CALL])
        ev, log = self.run_it(s, tools=TOOLS)
        calls = [x.call for k, x in ev if k == "event" and x.kind == "tool_call"]
        self.assertEqual(len(calls), 1); self.assertEqual(calls[0].arguments, {"path": "a.md", "content": "hi"})
        self.assertEqual([x for k, x in ev if k == "done"][0]["finish"], "stop")

    def test_disconnect_during_wrap_up_releases_turn_once(self):
        e, s = self.svc(["Let me think. " + "0" * 600, ANSWER])
        gen = s.run(self.ids, True, None, 3000, {}, threading.Event())
        with contextlib.redirect_stdout(io.StringIO()):
            seen = ""
            for k, x in gen:
                if k == "event" and x.kind == "reasoning":
                    seen += x.text or ""
                    if "give my answer" in seen:      # the wrap-up text is being streamed
                        break
            self.assertTrue(s.fifo.locked())
            gen.close()
        self.assertFalse(s.fifo.locked()); self.assertEqual(s.fifo.waiting(), 0)
        ev, _ = self.run_it(s)                        # next request is served
        self.assertEqual([x for k, x in ev if k == "done"][0]["finish"], "stop")

    def test_recover_mode_on(self):
        e, s = self.svc(["Let me think. " + "0" * 600, ANSWER]); s.reasoning_loop_recovery = "recover"
        ev, log = self.run_it(s)
        self.assertEqual([x for k, x in ev if k == "done"][0]["finish"], "stop"); self.assertEqual(len(e.prompts), 2)

    def test_two_concurrent_requests(self):
        e, s = self.svc(["Let me think. " + "0" * 600, ANSWER, "Let me think. " + "0" * 600, ANSWER], answer_room=True)
        e.cycle = True
        res = []
        def go():
            ev, _ = self.run_it(s); res.append([x for k, x in ev if k == "done"][0]["finish"])
        ts = [threading.Thread(target=go) for _ in range(2)]
        [t.start() for t in ts]; [t.join(30) for t in ts]
        self.assertEqual(res, ["stop", "stop"]); self.assertFalse(s.fifo.locked())

    def test_answer_that_starts_with_the_loop_token_is_not_cut_at_once(self):
        # the counter must restart after the wrap-up: the run's token can be the answer's first token ("0 apples")
        e, s = self.svc(["Let me think. " + "0" * 600, "0 apples"])
        ev, log = self.run_it(s)
        self.assertEqual("".join(x.text or "" for k, x in ev if k == "event" and x.kind == "content"), "0 apples")
        self.assertEqual([x for k, x in ev if k == "done"][0]["finish"], "stop")

    def test_legit_space_run_in_thinking_now_closes_thinking(self):
        # behaviour record: a legit 300-token run inside the thinking (before: reply ended 'length', no answer)
        e, s = self.svc(["Table:\n" + "-" * 300 + "\nmore thinking\n</think>\n\n" + ANSWER, ANSWER])
        ev, log = self.run_it(s)
        done = [x for k, x in ev if k == "done"][0]
        self.assertEqual(done["finish"], "stop")

    def test_anthropic_stream(self):
        e, s = self.svc(["<think>\nLet me think. " + "0" * 600, ANSWER])
        httpd = serve(s, port=0)
        try:
            body = {"model": "m", "max_tokens": 3000, "stream": True, "messages": [{"role": "user", "content": "hi"}],
                    "thinking": {"type": "enabled", "budget_tokens": 2000}}
            req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_address[1]}/v1/messages",
                                         data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            with contextlib.redirect_stdout(io.StringIO()):
                raw = urllib.request.urlopen(req, timeout=30).read().decode()
        finally:
            httpd.shutdown(); httpd.server_close()
        evs = [json.loads(l[6:]) for l in raw.splitlines() if l.startswith("data: ")]
        md = [x for x in evs if x["type"] == "message_delta"]
        self.assertEqual(md[-1]["delta"]["stop_reason"], "end_turn")
        text = "".join(x["delta"].get("text", "") for x in evs if x["type"] == "content_block_delta")
        self.assertEqual(text, ANSWER)
        self.assertFalse([x for x in evs if x["type"] == "error"])
        self.assertEqual(evs[-1]["type"], "message_stop")


if __name__ == "__main__":
    unittest.main()
