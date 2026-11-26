"""The FairLock streak cap (max_streak, capped_rank): a chain of high-rank requests cannot hold off the ordinary line.

A grant to the capped rank while a lower rank waits advances a streak; once the streak reaches max_streak, the next
grant goes to the first waiting lower-rank place instead (streak_skips counts those redirects).  A head above the
capped rank (the VRAM resize) is never delayed and leaves the streak alone.  CPU only: no engine, no model.

    python -m unittest serve.test_fairlock_streak -v
"""
import unittest

from serve.fairlock import FairLock


def granted_place(places):
    """The one place that was handed the turn and not taken or given up yet (exactly one in these walks)."""
    got = [p for p in places if p.granted and not p.taken and not p.left]
    assert len(got) == 1, [p.rank for p in places]
    return got[0]


def take_and_release(test, lock, place):
    test.assertTrue(place.wait(0))              # the turn was already handed over under the lock
    lock.release()


class StreakOrder(unittest.TestCase):
    def test_six_now_two_ordinary_one_resumed_come_out_in_the_pinned_order(self):
        # FairLock(max_streak=3): holder H, then 6 rank-3 places, 2 rank-0 and 1 rank-1 in that arrival order.
        lock = FairLock(max_streak=3)
        lock.acquire()                          # the slow holder
        nows = [lock.enter(3) for _ in range(6)]
        ordinary = [lock.enter() for _ in range(2)]
        resumed = lock.enter(1)                 # queues ahead of rank 0, behind every rank 3
        places = nows + ordinary + [resumed]
        lock.release()                          # H's turn ends; from here each grant is driven by its own place
        order = []
        for _ in range(len(places)):
            nxt = granted_place(places)
            order.append(nxt.rank)
            take_and_release(self, lock, nxt)
        # three now turns, then the cap sends the resumed rank-1 request first, three more now turns, the pair of 0s last
        self.assertEqual(order, [3, 3, 3, 1, 3, 3, 3, 0, 0])
        self.assertFalse(lock.locked())

    def test_a_grant_without_a_lower_waiter_does_not_build_or_keep_a_streak(self):
        lock = FairLock(max_streak=2)
        lock.acquire()
        nows = [lock.enter(3) for _ in range(4)]     # nothing below the capped rank ever waits
        lock.release()
        order = []
        for _ in range(len(nows)):
            nxt = granted_place(nows)
            order.append(nxt.rank)
            take_and_release(self, lock, nxt)
        self.assertEqual(order, [3, 3, 3, 3])
        self.assertEqual(lock.streak_skips, 0)

    def test_one_now_turn_in_a_row_then_the_ordinary_line(self):
        lock = FairLock(max_streak=1)
        lock.acquire()
        a, b, c = lock.enter(3), lock.enter(3), lock.enter(3)
        z = lock.enter()
        places = [a, b, c, z]
        lock.release()
        order = []
        for _ in range(len(places)):
            nxt = granted_place(places)
            order.append(nxt.rank)
            take_and_release(self, lock, nxt)
        self.assertEqual(order, [3, 0, 3, 3])       # each now turn is followed by a lower-rank grant once the cap hits
        self.assertEqual(lock.streak_skips, 1)      # only one redirect: after z no lower rank waits any more

    def test_a_resize_head_goes_first_and_does_not_reset_the_streak(self):
        lock = FairLock(max_streak=1)
        lock.acquire()
        a, b, c = lock.enter(3), lock.enter(3), lock.enter(3)
        z = lock.enter()
        places = [a, b, c, z]
        lock.release()                              # a goes first (streak 1: z waits behind it)
        a.wait(0)                                   # a holds the turn; nothing is passed on yet
        order = [3]
        x = lock.enter(4)                           # an admin VRAM resize arrives while the streak runs
        places.append(x)
        lock.release()                              # a ends: the head is now the rank-4 place
        for _ in range(len(places) - 1):
            nxt = granted_place(places)
            order.append(nxt.rank)
            take_and_release(self, lock, nxt)
        # x (rank 4) outranks the whole line and is never counted; b is then capped out, so z goes before it
        self.assertEqual(order, [3, 4, 0, 3, 3])
        self.assertEqual(lock.streak_skips, 1)

    def test_a_cancelled_grant_passes_the_turn_on_through_the_same_rule(self):
        lock = FairLock(max_streak=2)
        lock.acquire()
        a, b, c = lock.enter(3), lock.enter(3), lock.enter(3)
        z = lock.enter()
        places = [a, b, c, z]
        lock.release()                              # a is handed the turn ...
        first = granted_place(places)
        self.assertIs(first, a)
        a.cancel()                                  # ... and gives it up before taking it: the pass-on sees the streak
        order = []
        while any(p.granted and not p.taken and not p.left for p in places):
            nxt = granted_place(places)
            order.append(nxt.rank)
            take_and_release(self, lock, nxt)
        # b is granted (streak 2), then the cap sends z before c; a left with nothing owed to it
        self.assertEqual(order, [3, 0, 3])
        self.assertTrue(a.left)
        self.assertFalse(lock.locked())


class StreakDefaults(unittest.TestCase):
    def test_a_plain_FairLock_behaves_exactly_as_before(self):
        # the serve/test_fairlock.py rank walk: ranks order the line, nothing is ever redirected
        lock = FairLock()
        lock.acquire()
        a, b = lock.enter(), lock.enter()
        again = lock.enter(1)
        first, second = lock.enter(2), lock.enter(2)      # preempts used rank 2 before the lane existed
        places = [a, b, again, first, second]
        order = []
        lock.release()
        for _ in range(len(places)):
            nxt = granted_place(places)
            order.append("preempt1" if nxt is first else "preempt2" if nxt is second
                         else "again" if nxt is again else "a" if nxt is a else "b")
            take_and_release(self, lock, nxt)
        self.assertEqual(order, ["preempt1", "preempt2", "again", "a", "b"])
        self.assertFalse(lock.locked())

    def test_the_default_lock_never_counts_or_skips(self):
        lock = FairLock()                                  # max_streak=0: the cap is off whatever the ranks
        lock.acquire()
        nows = [lock.enter(3) for _ in range(4)]
        z = lock.enter()
        places = nows + [z]
        order = []
        lock.release()
        for _ in range(len(places)):
            nxt = granted_place(places)
            order.append(nxt.rank)
            take_and_release(self, lock, nxt)
        self.assertEqual(order, [3, 3, 3, 3, 0])           # arrival order: rank 3 in front, no redirect
        self.assertEqual(lock.streak_skips, 0)


if __name__ == "__main__":
    unittest.main()
