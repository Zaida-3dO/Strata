"""End to end over real HTTP: a long streamed request cut by a priority "now", retried with tools/strata_retry.py,
on all three chat routes (docs/PRIORITY_NOW.md).  Real handler and the real retry helper; only the engine is the fake
ElasticEngine - no GPU.  The client-side contract is what this pins: whatever a route sends when the cut lands in the
middle of a stream, strata_retry sees a vram_preempted cut, sends the same request again (the resume credit keeps its
place) and the caller ends up with a full answer that had error in it nowhere.

    python -m unittest serve.test_priority_now_e2e -v
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from serve.test_priority_now import LONGSTREAM, NOW_HEADER, Lane   # fixtures, not copies
from serve.test_vram_preempt import SHORT, anthropic, openai

TOOLS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "stack_live_check.py")


class CutAndRetryThroughTheHelper(Lane):
    """One long streamed request per route; a now cuts it after ~1 s of stream; post_with_retry sends it again and the
    caller only sees: attempts 2, answer complete."""

    def run_route(self, path, body_builder, last_event_check):
        from strata_retry import post_with_retry
        # plan order: the long first try (streamed, cut mid-way), the now's short answer, the long request's full retry
        self.engine.plans[:] = [LONGSTREAM, {"text": SHORT, "delay": 0.0}, {"text": "word " * 10, "delay": 0.0}]
        url = f"http://127.0.0.1:{self.port}{path}"
        result, retried, got = {}, [], []

        def go():
            result["reply"] = post_with_retry(url, body_builder("long-request"), stream=True, timeout=60,
                                              on_event=lambda n, d: got.append((n, d)),
                                              on_retry=lambda n, why: retried.append(why))
        t = threading.Thread(target=go)
        t.start()
        self.wait_until(lambda: len(got) >= 5, "the long request never streamed")
        n = self.start(path, body_builder("now-request", short=True), NOW_HEADER)
        self.assertTrue(n.finished.wait(10), "the now request never answered")
        self.assertEqual(n.status, 200, n.text)
        self.assertTrue(t.join(30) or not t.is_alive(), "the retried long request never finished")
        reply = result["reply"]
        self.assertEqual(len(retried), 1, f"expected exactly one retry, saw {retried}")
        self.assertIn("X-Strata-Priority", retried[0], f"the cut error should name the lane: {retried[0]}")
        self.assertTrue(reply.ok, f"the helper ended on a bad reply: status={reply.status} events={len(reply.events)}")
        self.assertEqual(reply.attempts, 2)
        last_event_check(reply)
        self.assertEqual(self.svc.now_cuts, 1)
        self.assertEqual(self.svc.resume_credit_hits, 1,   # the identical-prompt retry took its place in line
                         "the helper's retry should hit the resume credit")
        return reply

    def test_openai_chat_completions_stream_survives_a_now_cut(self):
        def check(reply):
            fins = [d.get("choices", [{}])[0].get("finish_reason") for _, d in reply.events
                    if isinstance(d, dict) and d.get("choices")]
            self.assertIn("stop", fins)
        self.run_route("/v1/chat/completions", lambda tag, short=False: openai(tag, stream=True), check)

    def test_anthropic_messages_stream_survives_a_now_cut(self):
        def check(reply):
            names = [name for name, _ in reply.events]
            self.assertEqual(names[-1], "message_stop")
            self.assertNotIn("error", names, "the cut of the first attempt must not appear in the final answer's events")
        self.run_route("/v1/messages", lambda tag, short=False: anthropic(tag), check)

    def test_responses_stream_survives_a_now_cut(self):
        def body(tag, short=False):
            return {"model": "m", "stream": True, "input": [{"role": "user", "content": f"{tag}"}]}

        def check(reply):
            self.assertTrue(any(isinstance(d, dict) and d.get("type") == "response.completed" for _, d in reply.events),
                            [n for n, _ in reply.events])
        self.run_route("/v1/responses", body, check)


class LiveCheckSendsNothingBare(unittest.TestCase):
    """`stack_live_check.py now` run bare: help + exit 2, nothing sent (same gate as the preempt check)."""

    def test_bare_now_check_sends_nothing(self):
        r = subprocess.run([sys.executable, TOOLS, "now"], capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 2)
        self.assertIn("Nothing was sent", r.stderr)

    def test_now_check_needs_the_cut_flag_even_with_a_base(self):
        # a base nobody listens on: if the script ever tried to send, this would not exit 2 with this message
        r = subprocess.run([sys.executable, TOOLS, "now", "--base", "http://127.0.0.1:1"],
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 2)
        self.assertIn("--yes-cut-requests", r.stderr)


class LiveCheckNowSubcommandExists(unittest.TestCase):
    def test_the_help_lists_the_now_check(self):
        r = subprocess.run([sys.executable, TOOLS, "--help"], capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0)
        self.assertIn("now", r.stdout)
