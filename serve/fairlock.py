"""A lock that hands the turn to its waiters in the order they arrived.

The server runs one request at a time through one lock.  A plain threading.Lock is not fair: a woken waiter has to
win the lock against every thread that arrives meanwhile, and a waiter that polls (acquire with a short timeout, so
it can send a keep-alive or notice that its client left) is not queued at all, so a request could wait behind
hundreds of later ones.  FairLock keeps a line: release gives the turn to the first waiter, and a new arrival
never takes it past a waiter.

It has the surface of threading.Lock (acquire with blocking and timeout, release, locked, the context manager).
A caller that polls takes its place once with enter() and then waits on that place again and again, so a poll that
times out does not cost it its position:

    place = lock.enter()
    try:
        while not place.wait(0.1):
            ...                     # keep-alive, client-gone and deadline checks
    except BaseException:
        place.cancel()              # leave the line; a turn handed over meanwhile moves on to the next waiter
        raise
    ...                             # it holds the turn: lock.release() when done

Known limit: the order is per turn, not per request.  One request can take the turn more than once (load() when the
model was unloaded, each image encode, then the generation), and each time it takes a new place at the back of the
line, so a request that needs a second turn waits again behind whoever arrived in between.  That is no worse than the
old random order.  Carrying a request's first place across its turns would fix it.
"""
from __future__ import annotations

import collections
import threading


class Place:
    """One waiter's place in a FairLock's line."""

    __slots__ = ("lock", "event", "granted", "taken", "left", "rank")

    def __init__(self, lock: "FairLock", rank: int = 0):
        self.lock = lock
        self.rank = rank            # 0: an ordinary arrival; higher ranks stand in front of lower ones (FairLock.enter)
        self.event = threading.Event()
        self.granted = False        # the turn was handed to this place (set under the lock's mutex)
        self.taken = False          # wait() reported True: the caller holds the turn and must release it
        self.left = False           # the place left the line without the turn

    def wait(self, timeout: float | None = None) -> bool:
        """True when the turn is this place's (the caller must release the lock); False after timeout seconds."""
        if not self.granted:
            self.event.wait(timeout)
        with self.lock._mutex:
            if self.granted:
                self.taken = True
            return self.taken

    def cancel(self) -> None:
        """Give up the place.  Nothing to do once wait() has returned True (the caller then holds the turn).  A turn
        handed over after the last wait() timed out is passed on, never lost."""
        lock = self.lock
        with lock._mutex:
            if self.taken or self.left:
                return
            self.left = True
            if self.granted:
                lock._pass_on_locked()
            else:
                try:
                    lock._line.remove(self)
                except ValueError:
                    pass

    def ahead(self) -> int:
        """How many requests are before this one: the one running plus the waiters in front of it."""
        lock = self.lock
        with lock._mutex:
            if self.granted or self.left:
                return 0
            try:
                return lock._line.index(self) + (1 if lock._held else 0)
            except ValueError:
                return 0


class FairLock:
    def __init__(self):
        self._mutex = threading.Lock()
        self._held = False
        self._line: collections.deque[Place] = collections.deque()

    def enter(self, rank: int = 0) -> Place:
        """Take a place in the line.  If the lock is free and nobody waits, the place already holds the turn.

        `rank` (0 for everyone, the default): a place with a higher rank stands in front of every waiter with a lower
        one, and behind the waiters of its own rank that came before it.  It never takes the turn from the holder: the
        turn changes hands only when the holder releases it, so a high rank is the next to be served, not an
        interruption.  (The VRAM preempt of POST /v1/vram, and a request the server sends back into the line.)"""
        place = Place(self, rank)
        with self._mutex:
            if not self._held and not self._line:
                self._held = True
                place.granted = True
                place.event.set()
            elif rank <= 0 or not self._line or self._line[-1].rank >= rank:
                self._line.append(place)
            else:
                at = next(i for i, other in enumerate(self._line) if other.rank < rank)
                self._line.insert(at, place)
        return place

    def acquire(self, blocking: bool = True, timeout: float = -1, rank: int = 0) -> bool:
        if not blocking:
            with self._mutex:
                if self._held or self._line:
                    return False
                self._held = True
                return True
        place = self.enter(rank)
        try:
            return place.wait(None if timeout is None or timeout < 0 else timeout)
        except BaseException:
            place.cancel()
            raise
        finally:
            if not place.taken:
                place.cancel()

    def release(self) -> None:
        with self._mutex:
            if not self._held:
                raise RuntimeError("release unlocked lock")
            self._pass_on_locked()

    def _pass_on_locked(self) -> None:
        if self._line:
            nxt = self._line.popleft()
            nxt.granted = True              # _held stays True: the turn goes straight to the first waiter
            nxt.event.set()
        else:
            self._held = False

    def locked(self) -> bool:
        return self._held

    def waiting(self) -> int:
        return len(self._line)

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()
