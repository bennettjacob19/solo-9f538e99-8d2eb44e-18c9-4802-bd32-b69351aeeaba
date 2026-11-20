"""Compute-slot scheduling: one inference at a time, with urgent preemption.

The 256 MiB container budget only leaves room for the buffers of a single
tiled inference.  Requests therefore share one compute slot.  Two kinds of
jobs use it:

* **normal** jobs (full-roll inspection) run tile by tile over their whole
  image.  They may be preempted at a tile boundary, but at most **once** in
  their lifetime.
* **urgent** jobs (re-shot small images) are served in acceptance (FIFO)
  order.  When a validated urgent job is pending and the running normal job
  has not yielded yet, the normal job must hand the slot over after it has
  written its current tile and dropped that tile's compute buffers.  It then
  waits until the urgent jobs that were already accepted at that moment have
  finished and resumes exactly where it left off (its already-written tiles
  stay on disk).

Only the thread owning the slot computes; everybody else is parked on a
condition variable, so at most one image is ever being processed.

The public surface is intentionally small: :meth:`ComputeScheduler.ticket`,
:meth:`acquire`, :meth:`checkpoint` and :meth:`release`.
"""

from __future__ import annotations

import bisect
import itertools
import threading
from dataclasses import dataclass, field

_SEQ = itertools.count()


@dataclass
class Ticket:
    """A pending or running inference job."""

    urgent: bool
    # Monotonic acceptance sequence: urgent jobs are served in this order.
    seq: int = field(default_factory=lambda: next(_SEQ))
    # True once this normal job has handed the slot over (at most once).
    yielded: bool = False

    def __lt__(self, other: "Ticket") -> bool:  # ordering for bisect
        return self.seq < other.seq


class ComputeScheduler:
    """Monitor arbitrating the single inference slot."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._owner: Ticket | None = None
        # Urgent tickets in acceptance order (sorted by seq).
        self._urgent: list[Ticket] = []
        # Normal tickets waiting for their first compute slot, in arrival order.
        self._normal: list[Ticket] = []
        # A normal job that yielded and must resume before fresh work.
        self._resume: Ticket | None = None
        # Urgent jobs accepted before the yield that still have to finish
        # before the yielded normal job may resume.
        self._resume_pending: list[Ticket] = []

    # -- tickets -----------------------------------------------------------

    def ticket(self, urgent: bool = False) -> Ticket:
        return Ticket(urgent=urgent)

    # -- slot ownership ----------------------------------------------------

    def acquire(self, ticket: Ticket) -> None:
        """Queue ``ticket`` and block until it owns the compute slot."""
        with self._cond:
            if ticket.urgent:
                bisect.insort(self._urgent, ticket)
            else:
                self._normal.append(ticket)
            self._dispatch()
            self._cond.notify_all()
            while self._owner is not ticket:
                self._cond.wait()

    def release(self, ticket: Ticket) -> None:
        """Give up the slot (idempotent; safe from a ``finally`` block)."""
        with self._cond:
            if self._owner is ticket:
                self._owner = None
            if ticket.urgent and self._resume_pending:
                try:
                    self._resume_pending.remove(ticket)
                except ValueError:
                    pass
            self._dispatch()
            self._cond.notify_all()

    def checkpoint(self, ticket: Ticket) -> None:
        """Tile-boundary hook for a running normal job.

        Must only be called after the current tile has been fully written and
        its temporary buffers released.  If a validated urgent job is pending
        and this job has not yielded yet, this blocks until the urgent work
        accepted up to now has been served in order, then resumes.
        """
        with self._cond:
            if ticket.urgent or ticket.yielded or not self._urgent:
                return
            ticket.yielded = True
            self._resume = ticket
            self._resume_pending = list(self._urgent)
            self._owner = None
            self._dispatch()
            self._cond.notify_all()
            while self._owner is not ticket:
                self._cond.wait()

    # -- dispatching -------------------------------------------------------

    def _dispatch(self) -> None:
        """Hand the free slot to the next job (call with the lock held)."""
        if self._owner is not None:
            return
        if self._resume is not None:
            if self._resume_pending:
                # Urgent work accepted before the hand-over goes first.
                self._owner = self._urgent.pop(0)
                return
            # The snapshot is done: resume the preempted normal job ahead of
            # later urgent arrivals and of normals that have not started.
            self._owner = self._resume
            self._resume = None
        elif self._urgent:
            self._owner = self._urgent.pop(0)
        elif self._normal:
            self._owner = self._normal.pop(0)
