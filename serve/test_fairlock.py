"""The engine's single turn (Service.fifo) is a FairLock: waiters are admitted in the order they arrived, also when they
poll with short timeouts (a keep-alive, a client that may have left), and a waiter that leaves does not lose the turn
for the one behind it.  CPU only: no engine, no model.

    python -m unittest serve.test_fairlock -v
"""
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import serve.server as server
from serve.fairlock import FairLock
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service

ROOT = Path(__file__).resolve().parents[1]


def wait_until(test, cond, what, limit=5.0):
    end = time.monotonic() + limit
    while not cond() and time.monotonic() < end:
        time.sleep(0.002)
    test.assertTrue(cond(), what)


class FairLockOrder(unittest.TestCase):
    def start_waiters(self, lock, n, how):
        """n threads that start one after the other (each is in the line before the next starts); -> (threads, order)."""
        order, threads = [], []
        for i in range(n):
            def run(i=i):
                if how == "block":
                    got = lock.acquire()
                else:                                       # polls the way the server's queued requests do
                    place = lock.enter()
                    while not place.wait(0.001):
                        pass
                    got = True
                if got:
                    order.append(i)
                    time.sleep(0.001)
                    lock.release()
            t = threading.Thread(target=run)
            t.start()
            threads.append(t)
            wait_until(self, lambda: lock.waiting() == i + 1, f"waiter {i} never queued")
        return threads, order

    def test_a_slow_holder_and_three_waiters_run_in_arrival_order(self):
        for how in ("block", "poll"):
            for _ in range(40):
                lock = FairLock()
                lock.acquire()                              # the slow holder
                threads, order = self.start_waiters(lock, 3, how)
                time.sleep(0.01 if how == "poll" else 0)    # the pollers time out and look again meanwhile
                lock.release()
                for t in threads:
                    t.join(5)
                    self.assertFalse(t.is_alive())
                self.assertEqual(order, [0, 1, 2], how)
                self.assertFalse(lock.locked())

    def test_a_late_arrival_does_not_pass_a_waiter(self):
        lock = FairLock()
        lock.acquire()
        threads, order = self.start_waiters(lock, 2, "block")
        lock.release()                                      # the turn goes to waiter 0 at once ...
        self.assertFalse(lock.acquire(blocking=False))      # ... so a newcomer finds it taken
        for t in threads:
            t.join(5)
        self.assertEqual(order, [0, 1])

    def test_a_waiter_that_times_out_leaves_the_order_intact(self):
        for _ in range(20):
            lock = FairLock()
            lock.acquire()
            threads, order = self.start_waiters(lock, 1, "block")
            self.assertFalse(lock.acquire(timeout=0.02))    # the second in line gives up
            self.assertEqual(lock.waiting(), 1)
            more = []
            t = threading.Thread(target=lambda: (lock.acquire(), more.append("c"), lock.release()))
            t.start()
            wait_until(self, lambda: lock.waiting() == 2, "the third never queued")
            lock.release()
            for th in threads + [t]:
                th.join(5)
                self.assertFalse(th.is_alive())
            self.assertEqual((order, more), ([0], ["c"]))
            self.assertFalse(lock.locked())

    def test_a_cancelled_place_does_not_deadlock_the_next(self):
        lock = FairLock()
        lock.acquire()
        gone = lock.enter()
        stays = lock.enter()
        self.assertFalse(gone.wait(0.01))
        self.assertEqual(stays.ahead(), 2)                  # the holder and the one in front
        gone.cancel()
        self.assertEqual(stays.ahead(), 1)
        lock.release()
        self.assertTrue(stays.wait(2))
        stays.cancel()                                      # after wait() returned True the turn is the caller's
        self.assertTrue(lock.locked())
        lock.release()
        self.assertFalse(lock.locked())

    def test_a_turn_handed_to_a_cancelling_place_moves_on(self):
        lock = FairLock()
        lock.acquire()
        first, second = lock.enter(), lock.enter()
        self.assertFalse(first.wait(0.01))
        lock.release()                                      # handed to `first` after its last wait() timed out
        first.cancel()                                      # it gives up: the turn must reach `second`
        self.assertTrue(second.wait(2))
        lock.release()
        self.assertFalse(lock.locked())

    def test_non_blocking_acquire_fails_while_others_queue(self):
        lock = FairLock()
        self.assertTrue(lock.acquire(blocking=False))
        self.assertFalse(lock.acquire(blocking=False))      # held
        place = lock.enter()
        lock.release()                                      # free for an instant, but the turn is the waiter's
        self.assertTrue(lock.locked())
        self.assertFalse(lock.acquire(blocking=False))
        self.assertTrue(place.wait(1))
        lock.release()
        self.assertTrue(lock.acquire(blocking=False))       # nobody queued and nobody holding: it succeeds
        lock.release()

    def test_timeout_acquire_returns_false_and_leaves_the_lock_usable(self):
        lock = FairLock()
        lock.acquire()
        t0 = time.monotonic()
        self.assertFalse(lock.acquire(timeout=0.05))
        self.assertGreaterEqual(time.monotonic() - t0, 0.04)
        self.assertEqual(lock.waiting(), 0)
        lock.release()
        self.assertTrue(lock.acquire(timeout=0.05))
        lock.release()
        with self.assertRaises(RuntimeError):
            lock.release()                                  # like threading.Lock

    def test_the_context_manager_releases_on_an_exception(self):
        lock = FairLock()
        with self.assertRaises(ValueError):
            with lock:
                self.assertTrue(lock.locked())
                raise ValueError("boom")
        self.assertFalse(lock.locked())
        self.assertTrue(lock.acquire(blocking=False))
        lock.release()


class ServiceQueueOrder(unittest.TestCase):
    def test_queued_requests_are_admitted_in_arrival_order_while_pinging(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "ok"), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.assertIsInstance(svc.fifo, FairLock)
        p = mock.patch.object(server, "QUEUE_BEAT_S", 0.02)    # every queued request polls often, as under load
        p.start()
        self.addCleanup(p.stop)
        ids = svc.tok.encode("hi")
        svc.fifo.acquire()                                  # a long request holds the turn
        finished, pings, threads = [], [], []
        for i in range(4):
            def go(i=i):
                for ev in svc.run(ids, False, None, 8, {}, threading.Event()):
                    if ev[0] == "ping":
                        pings.append(i)
                    elif ev[0] == "done":
                        finished.append(i)
            t = threading.Thread(target=go)
            t.start()
            threads.append(t)
            wait_until(self, lambda: svc.status["queued"] == i + 1, f"request {i} never queued")
        time.sleep(0.2)                                     # many polls: none of them may cost a place
        self.assertTrue(pings)
        svc.fifo.release()
        for t in threads:
            t.join(10)
            self.assertFalse(t.is_alive())
        self.assertEqual(finished, [0, 1, 2, 3])
        self.assertFalse(svc.fifo.locked())


if __name__ == "__main__":
    unittest.main()
