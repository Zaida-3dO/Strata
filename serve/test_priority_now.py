"""The priority "now" lane (docs/PRIORITY_NOW.md, config "priority_lane": true): a request with the header
X-Strata-Priority: now stands at rank 3 in the fair line (above waiters, above a requeued one, below an admin VRAM
resize) and cuts the running request with the same retryable error a preempt uses - but never another now.  A cut
request's own retry (identical prompt ids) keeps its place through the resume credit; after NOW_CUT_LIMIT cuts a
request is protected; after NOW_STREAK_MAX now turns in a row an ordinary waiter goes first.  CPU only: the fake
ElasticEngine of serve/test_vram_preempt.py, no model, no GPU - the fixtures are imported, not copied.

    python -m unittest serve.test_priority_now -v
"""
from __future__ import annotations

import http.client
import json
import threading
import time
import unittest

import serve.server as server
from serve.fairlock import FairLock
from serve.test_fairlock_streak import granted_place, take_and_release
from serve.test_vram_preempt import (SHORT, Call, ElasticEngine, PreemptCase, anthropic, load_helper, openai)

NOW_HEADER = {"X-Strata-Priority": "now"}
LONGSTREAM = {"text": "word " * 18, "delay": 0.02}       # the tokenizer is byte-level: ~90 tokens of streaming (~2 s)
NHOLD = {"text": "word " * 30, "delay": 0.02}             # ~150 tokens: about 3 s keeping the turn while the test sequences arrivals


def http_get(port, path):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        c.request("GET", path)
        r = c.getresponse()
        return r.status, json.loads(r.read().decode())
    finally:
        c.close()


class Lane(PreemptCase):
    """The lane as main() would build it before serve(): config key on, fifo swapped to the capped FairLock.  (The
    server reads svc.priority_lane per request, so switching it here after construction behaves like the config.)"""

    def setUp(self):
        super().setUp()
        self.svc.priority_lane = True
        self.svc.fifo = FairLock(server.NOW_STREAK_MAX, server.NOW_RANK)

    def now_events(self, event):
        return [e for e in self.svc.priority_events if e["event"] == event]

    def starts(self):
        return [e[1] for e in self.engine.log if len(e) == 2 and e[0] == "start"]


class ConstantsPinned(unittest.TestCase):
    """The rank numbers and caps named by the lane plan (docs/PRIORITY_NOW.md): a later change must fail loudly here."""

    def test_the_lane_constants_are_the_pinned_numbers(self):
        self.assertEqual(server.NOW_RANK, 3)
        self.assertEqual(server.PREEMPT_RANK, 4)          # was 2 before the lane: still ahead of everything
        self.assertEqual(server.REQUEUE_RANK, 1)
        self.assertEqual(server.NOW_CUT_LIMIT, 2)
        self.assertEqual(server.NOW_STREAK_MAX, 3)
        self.assertEqual(server.RESUME_CREDIT_S, 120)
        self.assertEqual(server.PRIORITY_HEADER, "X-Strata-Priority")


class LaneOffDefault(PreemptCase):
    """With the config off (every server that does not ask for it), the header changes nothing at all."""
    plans = [{"text": SHORT, "delay": 0.0}]

    def test_the_header_is_ignored_even_when_invalid(self):
        for headers in ({}, NOW_HEADER, {"X-Strata-Priority": "urgent"}):
            self.engine.starts = 0
            c = self.start("/v1/chat/completions", openai("A-request", stream=False), headers)
            self.assertTrue(c.finished.wait(10))
            self.assertEqual((c.status, self.engine.starts), (200, 1), headers)
        status = http_get(self.port, "/v1/status")[1]
        self.assertNotIn("priority", status)


class HeaderCheck(Lane):
    plans = [{"text": SHORT, "delay": 0.0}]

    def test_an_unknown_priority_value_gets_400_before_any_work(self):
        c = self.start("/v1/chat/completions", openai("A-request", stream=False), {"X-Strata-Priority": "urgent"})
        self.assertTrue(c.finished.wait(5))
        self.assertEqual(c.status, 400)
        self.assertIn("X-Strata-Priority", c.json()["error"]["message"])
        self.assertEqual(self.engine.starts, 0, "nothing may reach the engine before the header is accepted")

    def test_normal_is_an_ordinary_request(self):
        c = self.start("/v1/chat/completions", openai("A-request", stream=False), {"X-Strata-Priority": "normal"})
        self.assertTrue(c.finished.wait(10))
        self.assertEqual(c.status, 200)


class CutUnanswered(Lane):
    """Acceptance 2: A is queued/reading its prompt (nothing sent); a now cuts it within one engine step, answers,
    and the server runs A again - its client only sees a longer wait."""
    plans = [{"lead": 5.0}, {"text": SHORT, "delay": 0.0}]      # A's first try is stuck reading; every later call is fast

    def test_a_now_cut_a_request_that_had_sent_nothing_and_it_is_run_again(self):
        a = self.start("/v1/messages", anthropic("A-request"))
        self.wait_until(lambda: self.engine.active == 1, "A never started")
        t0 = time.monotonic()
        n = self.start("/v1/chat/completions", openai("N-request", stream=False), NOW_HEADER)
        self.assertTrue(n.finished.wait(8))                     # it cut A and answered: the lead of 5 s was not waited for
        self.assertLess(time.monotonic() - t0, 3.0)
        self.assertEqual(self.svc.now_cuts, 1)
        self.assertEqual(self.svc.preempt_cuts, 0, "a now cut counts as a now, not as a vram preempt")
        cut = [e for e in self.now_events("cut_issued")]
        self.assertEqual(len(cut), 1)
        self.assertNotEqual(cut[0]["by"], "vram_preempt", "the cutting request's own id names the by-field")
        self.assertEqual([e["event"] for e in self.now_events("requeued_by_server")], ["requeued_by_server"])
        self.assertTrue(a.finished.wait(10))                    # run again by the server: one long wait, then the answer
        self.assertEqual([name for name, _ in a.sse()][-1], "message_stop")
        self.assertNotIn("error", [name for name, _ in a.sse()])
        starts = self.starts()
        self.assertIn("A-request", starts[0])
        self.assertIn("N-request", starts[1])                   # N went first
        self.assertIn("A-request", starts[2])                   # ... and A again behind it
        status = http_get(self.port, "/v1/status")[1]
        self.assertEqual((status["priority"]["now_cuts"], status["priority"]["server_requeues"]), (1, 1))
        self.assertFalse(self.svc.fifo.locked())


class CutAnswered(Lane):
    """Acceptance 3: A already streamed content, so it gets the retryable error - with cause priority_now and no
    normal stop; its helper retry then takes rank 1 through the resume credit and goes before an ordinary request
    that queued while N ran (the control: without the credit C arrived first and would have gone first)."""
    plans = [LONGSTREAM, NHOLD, {"text": SHORT, "delay": 0.0}]

    def test_the_cut_answer_gets_an_error_then_the_retry_outranks_a_queued_ordinary_request(self):
        a = self.start("/v1/messages", anthropic("A-request"))
        a.wait_content(3)
        n = self.start("/v1/chat/completions", openai("N-request", stream=False), NOW_HEADER)
        self.assertTrue(a.finished.wait(8))                     # A ends with the error, not an answer
        events = a.sse()
        name, data = events[-1]
        self.assertEqual(name, "error")
        err = data["error"]
        self.assertEqual((err["type"], err["code"]), ("overloaded_error", "vram_preempted"))
        self.assertEqual(err["cause"], "priority_now")          # the only thing that changes for a client
        self.assertIn("urgent request", err["message"])
        self.assertNotIn("message_stop", [x for x, _ in events])
        self.assertEqual(self.svc.now_cuts, 1)

        self.wait_until(lambda: self.engine.active == 1 and any("N-request" in p for p in self.starts()),
                        "the now never got into the engine")     # it holds the turn ~3 s: the window to queue behind it
        c = self.start("/v1/chat/completions", openai("C-request", stream=False))   # queued while N still answers
        self.wait_until(lambda: self.svc.status["queued"] == 1, "C never queued")
        helper = load_helper()
        result, retried = {}, []

        def go():
            result["reply"] = helper.post_with_retry(
                f"http://127.0.0.1:{self.port}/v1/messages", anthropic("A-request"), stream=True,
                sleep=lambda s: None, on_retry=lambda attempt, why: retried.append(attempt))
        t = threading.Thread(target=go)
        t.start()
        self.assertTrue(n.finished.wait(8))
        t.join(30)
        self.assertFalse(t.is_alive())
        reply = result["reply"]
        self.assertTrue(reply.ok, reply)
        self.assertEqual(reply.attempts, 1)                     # one fresh send: the retry IS this attempt (rank 1 via credit)
        starts = self.starts()
        i_a2 = next(i for i, p in enumerate(starts) if "A-request" in p and i > 0)
        i_c = next(i for i, p in enumerate(starts) if "C-request" in p)
        self.assertLess(i_a2, i_c, "the resumed request (rank 1 via its credit) must run before the queued C")
        self.assertEqual(self.svc.resume_credit_hits, 1)
        self.assertIn("resume_credit_hit", [e["event"] for e in self.now_events("resume_credit_hit")])


class TwoNowRequests(Lane):
    """Acceptance 4: a now never cuts another now (it waits); an admin VRAM preempt still cuts everything."""
    plans = [LONGSTREAM, NHOLD, {"text": SHORT, "delay": 0.0}]

    def test_a_second_now_waits_and_never_cuts_the_first(self):
        a = self.start("/v1/messages", anthropic("A-request"))
        a.wait_content(3)
        n1 = self.start("/v1/messages", anthropic("N-one"), NOW_HEADER)
        self.wait_until(lambda: any("N-one" in p for p in self.starts()), "the first now never started")
        n2 = self.start("/v1/chat/completions", openai("N-two-request", stream=False), NOW_HEADER)
        self.wait_until(lambda: self.svc.status["queued"] >= 1, "the second now never queued")
        self.assertEqual(self.svc.now_cuts, 1, "the second now must not have cut the first")
        for call in (n1, n2):
            self.assertTrue(call.finished.wait(10))
        refused = self.now_events("cut_refused")
        self.assertTrue(refused, "the refused cut of a running now is in the event log")
        self.assertEqual(refused[0]["why"], "now")
        starts = self.starts()
        self.assertIn("N-one", starts[1])                       # N1 after cutting A ...
        self.assertIn("N-two-request", starts[2])               # ... and N2 only after N1 finished

class AdminPreemptsNow(Lane):
    """Split from the case above: while a now sits reading its prompt, an admin preempt cuts it (nothing sent yet,
    so the server runs it again) and the resize still happens."""
    plans = [LONGSTREAM, {"lead": 2.0}, {"text": SHORT, "delay": 0.0}]

    def test_a_vram_preempt_cuts_it_and_the_line_stays_clean(self):
        a = self.start("/v1/messages", anthropic("A-request"))
        a.wait_content(3)                                       # A: cut by the now, ends with the error event
        n = self.start("/v1/chat/completions", openai("N-request", stream=False), NOW_HEADER)
        self.wait_until(lambda: any("N-request" in p for p in self.starts()) and self.engine.active == 1,
                        "the now never got into the engine")
        code, _, body = self.vram({"reserve_mib": 4000, "preempt": True})
        self.assertEqual((code, body["preempt"]["cut"]), (200, 1))
        self.assertTrue(n.finished.wait(10))                    # nothing sent: run again by the server after the resize
        self.assertEqual(n.status, 200)
        vram_cuts = [e for e in self.now_events("cut_issued") if e["by"] == "vram_preempt"]
        self.assertEqual(len(vram_cuts), 1)


class ProtectedAfterTwoCuts(Lane):
    """Acceptance 5: two now cuts across a helper retry, then the third now does not cut it - it waits - and the
    event log says so (cut_refused with why protected)."""
    plans = [LONGSTREAM, {"text": SHORT, "delay": 0.0}, LONGSTREAM, {"text": SHORT, "delay": 0.0}, LONGSTREAM,
             {"text": SHORT, "delay": 0.0}]

    def test_the_third_now_waits_for_a_protected_request(self):
        a = self.start("/v1/messages", anthropic("A-request"))
        a.wait_content(3)
        n1 = self.start("/v1/chat/completions", openai("N-one-request", stream=False), NOW_HEADER)
        self.assertTrue(a.finished.wait(5))                     # cut 1 (content had been sent)
        self.assertTrue(n1.finished.wait(5))

        helper = load_helper()
        events, result = [], {}

        def go():
            result["reply"] = helper.post_with_retry(
                f"http://127.0.0.1:{self.port}/v1/messages", anthropic("A-request"), stream=True,
                sleep=lambda s: None, on_event=lambda name, data: events.append(name), attempts=5)
        t = threading.Thread(target=go)
        t.start()
        self.wait_until(lambda: sum(1 for e in events if e == "content_block_delta") >= 3,
                        "the retried request never streamed content")
        n2 = self.start("/v1/chat/completions", openai("N-two-request", stream=False), NOW_HEADER)
        self.wait_until(lambda: self.svc.now_cuts == 2, "the second now did not cut the retry")
        self.assertTrue(n2.finished.wait(5))
        self.wait_until(lambda: any(who.protected for who in list(self.svc.turn_holders.values())),
                        "the protected third attempt never held the turn")
        n3 = self.start("/v1/chat/completions", openai("N-three-request", stream=False), NOW_HEADER)
        # give N3's first cut attempt time to notice the protected holder, then pin the counters
        self.wait_until(lambda: any(e["why"] == "protected" for e in self.now_events("cut_refused")),
                        "the event log never named the protected refusal")
        self.assertEqual(self.svc.now_cuts, 2, "a protected request is not cut a third time")
        refused = [e for e in self.now_events("cut_refused") if e["why"] == "protected"]
        self.assertTrue(refused, "the event log names the protected refusal")
        self.assertEqual(self.svc.now_cuts_refused_protected, 1)   # one refusal per protected request, not one per poll
        t.join(20)
        self.assertFalse(t.is_alive())
        self.assertTrue(result["reply"].ok, result["reply"])    # the protected request answered fully (attempt 3)
        self.assertEqual(result["reply"].attempts, 2)          # try1 was cut (cut 2), the protected try2 answered
        self.assertTrue(n3.finished.wait(10))                   # and the third now was served after it
        self.assertEqual(n3.status, 200)
        self.assertEqual(self.engine.starts, 6)                 # A, N1, retry1, N2, retry2(protected), N3
        status = http_get(self.port, "/v1/status")[1]["priority"]
        self.assertEqual(status["now_cuts"], 2)
        self.assertEqual(status["resume_credit_hits"], 2)       # both client retries used their credit


class StreakWiring(Lane):
    """Acceptance 6: the capped FairLock main() installs for the lane redirects a chain of nows exactly once per
    cap (N1 N2 N3 A N4 N5 B, streak_skips = 1); the mechanics themselves are pinned in serve/test_fairlock_streak.py."""

    def test_five_nows_a_resumed_and_an_ordinary_go_in_the_expected_order(self):
        lock = self.svc.fifo
        lock.acquire()                                          # H holds the turn
        nows = [lock.enter(server.NOW_RANK) for _ in range(5)]
        b, a = lock.enter(), lock.enter(server.REQUEUE_RANK)    # B ordinary arrived, A resumed via its credit
        places = nows + [b, a]
        order = []
        lock.release()
        for _ in range(len(places)):
            nxt = granted_place(places)
            order.append(nxt.rank)
            take_and_release(self, lock, nxt)
        self.assertEqual(order, [3, 3, 3, 1, 3, 3, 0])          # N1 N2 N3 A N4 N5 B
        self.assertEqual(lock.streak_skips, 1)


class WaitsForUncuttable(Lane):
    """Acceptance 7: the turn is held by a session save / engine load (not in turn_holders): the now simply waits."""

    def test_a_now_waits_for_a_turn_that_cannot_be_cut(self):
        self.svc.fifo.acquire()
        try:
            n = self.start("/v1/chat/completions", openai("N-request", stream=False), NOW_HEADER)
            self.wait_until(lambda: self.svc.status["queued"] == 1, "the now never queued")
            time.sleep(0.3)                                     # a few PREEMPT_POLL slices of cut attempts
            self.assertEqual(self.svc.now_cuts, 0)
        finally:
            self.svc.fifo.release()
        self.assertTrue(n.finished.wait(10))
        self.assertEqual((n.status, self.engine.starts), (200, 1))
        self.assertEqual(self.svc.fifo.waiting(), 0, "the line is clean afterwards")


class ClientLeftWins(Lane):
    """Acceptance 9: a request whose client already left is not cut into an error for nobody (existing rule)."""

    def test_a_gone_holder_is_never_claimed_by_a_now_cut(self):
        cancel = threading.Event()
        who = server.Holder("r1", now=False)
        self.svc.hold(cancel, who)
        self.svc.note_client_disconnect(cancel)
        self.assertEqual(self.svc.preempt_cut(by_now=True, by="nx"), [])
        self.assertEqual(self.svc.now_cuts, 0)
        self.assertFalse(self.svc.resume_credit)


class ParallelIgnoresLane(Lane):
    """Acceptance 8: with parallel slots the header is ignored (a request never holds the solo turn there)."""

    def test_with_batch_slots_a_now_is_just_an_ordinary_request(self):
        self.engine.batch = 2                                   # what --parallel sets
        for name in ("A-request", "N-request"):
            c = self.start("/v1/chat/completions", openai(name, stream=False), NOW_HEADER)
            self.assertTrue(c.finished.wait(15))
            self.assertEqual(c.status, 200)
        self.assertEqual(self.svc.now_cuts, 0)
        self.assertFalse(self.now_events("cut_issued"))


class EventRingAndStatus(Lane):
    """Section E of the lane plan: GET /v1/priority/events returns the ring with epoch-ms stamps and a since filter;
    /v1/status only carries the "priority" block when the lane is on."""

    def test_the_ring_lists_the_story_of_a_cut(self):
        self.plans = [{"lead": 5.0}, {"text": SHORT, "delay": 0.0}]
        self.engine.plans = self.plans
        a = self.start("/v1/messages", anthropic("A-request"))
        self.wait_until(lambda: self.engine.active == 1, "A never started")
        n = self.start("/v1/chat/completions", openai("N-request", stream=False), NOW_HEADER)
        self.assertTrue(n.finished.wait(5))
        self.assertTrue(a.finished.wait(10))
        code, body = http_get(self.port, "/v1/priority/events")
        self.assertEqual(code, 200)
        rows = body["events"]
        names = [r["event"] for r in rows]
        for want in ("arrived", "cut_issued", "admitted", "finished"):
            self.assertIn(want, names)
        self.assertEqual([r["seq"] for r in rows], sorted(r["seq"] for r in rows))
        first = rows[0]["seq"]
        _, trimmed = http_get(self.port, f"/v1/priority/events?since={first}")
        self.assertTrue(trimmed["events"])
        self.assertLess(first, [r["seq"] for r in trimmed["events"]][0])
        st_ms = {r["event"]: r["t_ms"] for r in rows}
        self.assertIsInstance(st_ms["arrived"], int)
        arrived = next(r for r in rows if r["event"] == "arrived")
        self.assertEqual((arrived["prompt_tokens"] > 0, isinstance(arrived["ihash"], str)), (True, True))
        status = http_get(self.port, "/v1/status")[1]["priority"]
        self.assertTrue(status["enabled"])
        for key in ("now_served", "now_cuts", "now_cuts_refused_protected", "vram_cuts", "server_requeues",
                    "resume_credit_hits", "resume_credit_misses", "streak_skips", "longest_now_wait_ms"):
            self.assertIn(key, status)


if __name__ == "__main__":
    unittest.main()
