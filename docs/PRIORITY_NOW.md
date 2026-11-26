# The priority "now" lane: one request that must be answered right away

One house, one GPU, one engine. A speaker command or a phone message should not wait behind a 60-second agent turn.
The lane lets one marked request take the engine now: it stands ahead of everyone waiting, and it cuts the request that
is running - with exactly the retryable error a VRAM preempt already uses ([VRAM_PREEMPT.md](VRAM_PREEMPT.md)).

It is opt-in and off by default. With it off the lane does nothing at all (no header read, no ranks, no cutting); with
the config key absent a server still carries three additions that need nothing of clients: a `"cause"` field in the
preempt error body (`"vram_preempt"`), the log and ring lines a VRAM preempt writes (`cut_issued ...`), and
`GET /v1/priority/events` listing them **(test)**. Marks:
**(measured)** was run on real hardware; **(code)** is read from the code; **(test)** is pinned by
`serve/test_priority_now.py` / `serve/test_fairlock_streak.py` (CPU only, fake engine - no number here is a GPU
measurement).

## Turning it on and using it

Two config keys in `strata-<model>.json`:

```
"priority_lane": true          off by default; turns the lane on for this server
"priority_now_grace_s": 0      optional: an answering request gets these many seconds before a now cuts it (default 0)
```

The header marks a request:

```
X-Strata-Priority: now         this request may cut the running one
X-Strata-Priority: normal      ordinary (the default; saying it changes nothing)
```

With the lane on and any other value: `400` before any work **(test)**. With the lane off the header does nothing at all,
even an invalid value **(test)**. The header is honoured for anyone who passes the existing API-key check - same trust
domain as `POST /v1/vram`; a "now" costs someone else a retry, it does not leak anything. A lane that asked for itself
on an engine with parallel slots (`--parallel`) is switched off at start with one log line: there requests never hold the
solo turn and cutting runs through the control lines **(code)**.

At start the server prints `[strata] priority lane: on (X-Strata-Priority: now cuts the running request; max 2 cuts
per request, 3 urgent turns in a row)` **(code)**.

## Who may cut whom

In the fair line (the order the engine hands its one turn over) there are four ranks **(code)**:

| rank | who | can cut the running request? |
|---|---|---|
| 4 | an admin VRAM resize (`POST /v1/vram` with `preempt`) | yes - everything, a now included |
| 3 | a priority "now" request | only a request that is not itself a now |
| 1 | a cut request the server runs again, or the client's own retry of one (the resume credit) | no |
| 0 | ordinary requests | no |

- A `now` never cuts another `now`. Two urgent requests are equally urgent: they run first-come-first-served at rank 3.
- An admin preempt outranks everything: it is the controller taking memory back for another program.
- The cut of a running request uses the existing machinery: its cancel Event is marked, its generator stops between
  engine steps and gives the turn up **(code, test)**. A holder that cannot be cut (a session save, an engine load) is
  waited for, exactly as the resize waits **(test)**.
- `priority_now_grace_s` works like `preempt_grace_s`: within it only requests that have sent nothing or only thinking
  are cut; at 0 (the default) everything running is cut at once **(test)**.

## What a client sees, and must do

The error is **unchanged** where it can be: `529` with `overloaded_error` on `/v1/messages`, `503` on the other routes,
an SSE error event once a stream began, all with `code "vram_preempted"` and `Retry-After: 1`. New: a `"cause"` field
inside the error object - `"priority_now"` for a lane cut, `"vram_preempt"` for a resize **(test)**. Clients that follow
the three rules of [VRAM_PREEMPT.md](VRAM_PREEMPT.md#what-a-client-must-do) need no change at all; `tools/strata_retry.py`
retries this code already **(test)**. The `/v1/responses` error body keeps its own shape (its message text names the
cause instead).

## Not cut forever: the two caps

A long agent turn should not be sacrificed again and again for a chain of urgent requests **(code)**:

- **Two cuts per request.** A request's id-hash lives in a resume credit; after it has been cut twice by nows it is
  **protected**: even a `now` waits for it. The third attempt answers fully **(test)**. What the first cut already lost
  (the decode done) is not paid back; for a 60 s turn that is up to ~35 s per cut (**measured** at 6000 tokens in the
  resize-elastic run of 24 Nov).
- **Three now turns in a row.** `FairLock(max_streak=3)`: after three grants to rank 3 while an ordinary request waits,
  the next grant goes to the first waiting lower-rank place in line order (a resumed one, then an ordinary one), then
  the lane runs again. A resize (rank 4) is never delayed by this and never counted; when the waiter that justified a
  streak leaves the line, or the line empties, the count restarts - one lone `now` is never made to wait for nothing
  **(test: serve/test_fairlock_streak.py)**.
- **A completed tool call ends the cutting.** Once a whole tool-call has been streamed out, an agent may already have
  run it; cutting would make its retry run that tool a second time (a `git commit`, a file write, an HA service call).
  Even a `now` waits for such a turn: the refusal is logged as `cut_refused why="tool_done"` and counted **(test)**.

## The resume credit: a cut request's retry does not queue at the back

A now that must end an already-answering request cuts it with the error above; the client sends the same request again
(Claude Code does this by itself, `tools/strata_retry.py` for the rest). That retry arrives as a new request and would
stand behind everyone who queued meanwhile. So on a cut the server stores `blake2b(prompt token ids) -> (expires in
120 s, cuts so far)`; one use **(code)**. A later request with the same hash runs at rank 1 - ahead of ordinary waiters,
behind any now - and its prompt is found in the parked conversation cache without re-reading it **(test for the place
in line; the cache behaviour itself is under "not measured" below)**.

A retry that changed the body (a client that appends a message before resending) hashes to something else: it takes rank
0 and no credit, which is functionally fine, just slower. The event log shows `resume_credit_miss` when the old credit
expires unused **(code)**. One ordinary server does this bookkeeping only when the lane is on; the event ring (below) is
always small (500 rows).

## Reading what happened: `/v1/priority/events` and the status counters

`GET /v1/priority/events?since=<seq>` (auth as `/v1/status`) returns the last 500 lane events as JSON rows **(test)**:
`seq`, `t_ms` (epoch milliseconds, this PC's wall clock), `event`, `request_id`, `route`, `priority`
(`now`/`normal`), `busy_at_arrival`, `queue_ahead`, plus per event - `arrived` (`prompt_tokens`, first 12 hex of the id
hash), `cut_issued` (`by`: the cutting now's id, or `"vram_preempt"`; `victim_id`, `victim_generated_tokens` = what the
cut lost, `victim_sent`, `victim_answered`, `victim_phase`), `cut_refused` (`why: now|protected`), `admitted`
(`waited_ms`, `streak_skips_so_far`), `finished` (`finish`, `completion_tokens`, `total_ms`, `ttft_ms`, `cache_n`: what
the engine reused of this prompt - for a resumed request that is the parked-conversation answer), `requeued_by_server`
(`attempt`), `resume_credit_hit/miss` (`ihash`, `cuts_so_far`, `age_ms`), `resumed_admitted`. The same events also print
to the server window (kept in `runtime\launch-solo.server.log`); routine rows (arrived/admitted/finished) go to the ring
only, or a busy server would drown **(code)**. The VRAM preempt writes the same `cut_issued` rows with
`by: "vram_preempt"`, so a benchmark reads both kinds of victim from one log **(test)**.

`GET /v1/status` carries a `"priority"` block **only while the lane is on**: `enabled`, `now_served`, `now_cuts`,
`now_cuts_refused_protected`, `now_cuts_refused_tool_done`, `vram_cuts`, `server_requeues`, `resume_credit_hits`,
`resume_credit_misses`, `streak_skips`, `longest_now_wait_ms` **(test)**.

`python tools/stack_live_check.py now --live --yes-cut-requests` runs one end-to-end check against a live server:
a long streamed generation, one short `now` request in the middle, then it prints the event-log timeline and checks
"the long one finished (after its retry), the now answered first, cuts = 1".

## Limits - not measured yet

- Does the engine's conversation cache keep a cut prompt while a `now` prompt is read, **with one slot**? The code says
  that is what it is for; the resize-elastic measurement (32k-prompt retry: first token 0.19 to 0.22 s against 5.7 s
  cold) used the same prompt retried with nothing in between **(measured, but not with an interleaved now)**. If a
  resumed request's `cache_n` comes back 0 on real hardware, the cost is one cold read of its prompt (5.8 s per 32k
  tokens, measured). The event log records `cache_n` for exactly this question.
- Real Claude Code **interactive mode** under a now-cut, on a real model: not tested. Print mode with a fake engine is
  what VRAM_PREEMPT.md measures (Claude Code 2.1.289 retried in all four cut stages).
- Tool loops: cutting once a whole tool-call has streamed out is refused (the cap above), so a double-run should not
  happen from this lane. What the server cannot see: a client that reorders or re-runs tools on its own, and calls cut
  mid-stream before any whole call existed - a harness still counts tool executions from its own side **(code)**.
- A `now` request with pictures: its encode stands in line at rank 3 and no second now may cut it, but while another
  request runs the now waits for it to finish or be cut normally - the cutting loop lives in generation, not in an
  image encode **(test: serve/test_priority_now.py)**.
