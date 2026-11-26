"""Reviewer's extra tests for the fair queue (9e88160e): a waiter that leaves must never leave a place behind that the
turn is later handed to.  CPU only."""
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import serve.server as server
from serve.fairlock import FairLock
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, ClientGone, MockEngine, QueueCancelled, QueueWaitExpired, Service

ROOT = Path(__file__).resolve().parents[1]


class LeftPlacesDoNotStallTheLine(unittest.TestCase):
    def setUp(self):
        tok = ByteTokenizer()
        self.svc = Service(MockEngine(tok, "ok"), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        p = mock.patch.object(server, "QUEUE_BEAT_S", 0.02)
        p.start()
        self.addCleanup(p.stop)
        self.ids = self.svc.tok.encode("hi")

    def assert_line_moves_on(self):
        self.svc.fifo.release()                              # the holder finishes
        self.assertFalse(self.svc.fifo.locked(), "the turn went to a place nobody waits on")
        self.assertEqual(self.svc.fifo.waiting(), 0)
        done = [k for k, _ in self.svc.run(self.ids, False, None, 8, {}, threading.Event())]
        self.assertEqual(done[-1], "done")

    def test_a_queued_request_cancelled_by_its_client(self):
        self.svc.fifo.acquire()
        cancel = threading.Event()
        run = self.svc.run(self.ids, False, None, 8, {}, cancel)
        self.assertEqual(next(run), ("ping", None))
        cancel.set()
        list(run)
        self.assert_line_moves_on()

    def test_a_queued_request_whose_generator_is_closed(self):
        self.svc.fifo.acquire()
        run = self.svc.run(self.ids, False, None, 8, {}, threading.Event())
        next(run)
        run.close()
        self.assert_line_moves_on()

    def test_a_queued_request_behind_a_left_one_is_admitted(self):
        self.svc.fifo.acquire()
        gone = threading.Event()
        first = self.svc.run(self.ids, False, None, 8, {}, gone)
        next(first)                                          # first in line
        result = []
        t = threading.Thread(target=lambda: result.extend(k for k, _ in self.svc.run(self.ids, False, None, 8, {},
                                                                                       threading.Event())))
        t.start()
        end = time.monotonic() + 5
        while self.svc.fifo.waiting() < 2 and time.monotonic() < end:
            time.sleep(0.005)
        self.assertEqual(self.svc.fifo.waiting(), 2)
        gone.set()
        list(first)                                          # the first leaves its place
        self.svc.fifo.release()
        t.join(5)
        self.assertFalse(t.is_alive(), "the second request never got the turn")
        self.assertEqual(result[-1], "done")
        self.assertFalse(self.svc.fifo.locked())

    def test_an_image_turn_given_up_by_its_client(self):
        self.svc.fifo.acquire()
        cancel = threading.Event()
        err = []

        def wait_turn():
            try:
                with self.svc._turn(cancel):
                    pass
            except ClientGone as e:
                err.append(e)
        t = threading.Thread(target=wait_turn)
        t.start()
        end = time.monotonic() + 5
        while self.svc.fifo.waiting() < 1 and time.monotonic() < end:
            time.sleep(0.005)
        cancel.set()
        t.join(5)
        self.assertTrue(err)
        self.svc.fifo.release()
        self.assertFalse(self.svc.fifo.locked(), "the turn went to the image request that left")

    def test_the_parallel_start_wait_gives_its_place_up(self):
        for exc, cancel, deadline in ((QueueCancelled, threading.Event(), time.monotonic() + 5),
                                      (QueueWaitExpired, None, time.monotonic() + 0.05)):
            self.svc.fifo.acquire()
            if cancel is not None:
                threading.Timer(0.05, cancel.set).start()
            with self.assertRaises(exc):
                with self.svc._inference_fifo(deadline, cancel):
                    pass
            self.svc.fifo.release()
            self.assertFalse(self.svc.fifo.locked(), f"{exc.__name__}: the turn went to a place nobody waits on")


class AFailingLogLineNeverWedgesTheQueue(unittest.TestCase):
    """A queue line that cannot be written must not cost the turn (review finding 1: the 'admitted' line used to run
    between registering the release and entering the with, so a write that raised left the engine's turn held forever)."""

    def setUp(self):
        tok = ByteTokenizer()
        self.svc = Service(MockEngine(tok, "ok"), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        for name, value in (("QUEUE_LOG_S", 0.05), ("QUEUE_BEAT_S", 0.02)):
            p = mock.patch.object(server, name, value)
            p.start()
            self.addCleanup(p.stop)
        self.ids = self.svc.tok.encode("hi")

    def _wait_behind_a_holder(self):
        """Hold the turn, release it a moment after the request has waited long enough to be logged."""
        self.svc.fifo.acquire()
        threading.Timer(0.2, self.svc.fifo.release).start()

    def _run_to_the_end(self):
        return [k for k, _ in self.svc.run(self.ids, False, None, 8, {}, threading.Event())]

    def test_the_turn_is_released_when_the_log_function_raises(self):
        def log(message):
            if "admitted" in message:
                raise OSError(22, "Invalid argument (stderr gone)")
        self._wait_behind_a_holder()
        with mock.patch.object(server, "_queue_log", log):
            with self.assertRaises(OSError):
                self._run_to_the_end()
        self.assertFalse(self.svc.fifo.locked(), "the turn stayed held after the request ended")
        self.assertEqual(self._run_to_the_end()[-1], "done", "the next request never got the turn")

    def test_a_stderr_that_cannot_be_written_is_not_an_error(self):
        class Dead:
            def write(self, _):
                raise OSError(22, "Invalid argument")

            def flush(self):
                raise OSError(22, "Invalid argument")
        self._wait_behind_a_holder()
        with mock.patch.object(server.sys, "stderr", Dead()):
            self.assertEqual(self._run_to_the_end()[-1], "done")
        self.assertFalse(self.svc.fifo.locked())
        with mock.patch.object(server.sys, "stderr", mock.Mock(write=mock.Mock(side_effect=ValueError("closed")))):
            server._queue_log("still fine")                  # a closed stream raises ValueError: also swallowed


class VramResizeOrder(unittest.TestCase):
    def test_a_resize_waits_behind_queued_requests(self):
        """Pins what POST /v1/vram does today: its acquire (with a timeout) takes a place at the back of the line, behind
        the requests already waiting.  (A later change adds an opt-in front-of-line entry for it.)"""
        lock = FairLock()
        lock.acquire()
        order = []
        a = lock.enter()
        t = threading.Thread(target=lambda: (lock.acquire(timeout=5), order.append("resize"), lock.release()))
        t.start()
        end = time.monotonic() + 5
        while lock.waiting() < 2 and time.monotonic() < end:
            time.sleep(0.005)
        lock.release()
        self.assertTrue(a.wait(5))
        order.append("request")
        lock.release()
        t.join(5)
        self.assertEqual(order, ["request", "resize"])


if __name__ == "__main__":
    unittest.main()
