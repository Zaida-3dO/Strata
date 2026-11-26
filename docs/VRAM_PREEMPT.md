# Resize the VRAM at once: `preempt`, and what a client does when its request is cut

`POST /v1/vram` normally waits for the request that is running to finish (up to 300 s, then 409). That is fine for a
small answer and bad for a 60-second agent turn when another program needs the VRAM now. Two opt-in fields change it.
Without them nothing changes: the same call waits, answers and fails exactly as in [VRAM_ELASTIC.md](VRAM_ELASTIC.md).

Needs an engine started with `--vram-elastic` (the config's `"vram_elastic": true`). Marks: **(measured)** was run and
the number is from that run; **(code)** is read from the code; **(test)** is pinned by `serve/test_vram_preempt.py`.

## The request

```
POST /v1/vram
{"reserve_mib": 8000, "preempt": true}                          cut the running request now
{"reserve_mib": 8000, "preempt": true, "preempt_grace_s": 20}   wait up to 20 s for it to end on its own first
```

- `preempt` must be `true` or `false` (anything else: 400). `preempt_grace_s` is a number from 0 to 300 (default 0 =
  cut at once) and needs `preempt: true`.
- The resize goes to the **front of the line**: ahead of every request that is waiting, behind nobody. The request that
  is running is cut. The turn is only ever handed over between two engine steps, after the cut request has stopped the
  engine and read its last line, so a resize never lands in the middle of one **(code, test)**.
- Requests that are **waiting** are never cut. They keep their order and run after the resize **(test)**.
- A request that holds the turn for a **picture encode** is cut too: it stops between pictures (a picture that is being
  encoded finishes first) and gets the same retryable error before any answer began **(test)**. A session save/restore, an
  engine load and an image *generation* are not requests and are not cut: the preempt waits behind them (409 after
  `vram_wait_s`, 300 s, with `cut` in the body) **(test)**.
- Two preempts at once run one after the other in the order they came; the second finds nothing running **(test)**.

### The grace period

A cut after the answer has started is lost work. With `preempt_grace_s: N`:

- a request that has sent **nothing, or only thinking**, is cut **at once** (the cheapest to cut: the server runs one
  that sent nothing again by itself, and Claude Code retries one that sent only thinking);
- a request that is **streaming an answer or a tool call** gets N seconds to end on its own, and is cut only if it is
  still running after that **(test)**.

Pick N by the longest turn you can bear to wait for. A coder turn is typically 10 to 60 s. 0 is the right value when the
speaker or another program needs the VRAM now.

### The answer

```
200 {"status": "ok", ...the engine's figures as before...,
     "preempt": {"outcome": "cut" | "waited" | "nothing running",
                 "cut": 1, "cut_ids": ["458e8092ebf1"], "grace_s": 0.0,
                 "turn_wait_ms": 19, "resize_ms": 0, "took_ms": 19}}
```

`outcome` says what happened: `nothing running` (the turn was free), `waited` (a request was running and ended within
the grace, nothing was cut) or `cut`. `turn_wait_ms` is how long the resize waited for the turn, `resize_ms` the resize
itself (the engine's own, 10 to 36 ms for ~3.5 GiB on an RTX 5090 **(measured, 24 Nov 2026, resize-elastic REPORT)**).

| Case | Status | Body |
|---|---|---|
| A preempt on an engine without `--vram-elastic` | 400 | `vram_not_elastic`; **nothing was cut** |
| The turn was held by something that cannot be cut for `vram_wait_s` | 409 | `model_busy` with `cut` |
| The engine refused the resize after the cut | 409 | `vram_resize_failed` with `cut` and `reserve_unchanged: true`; the reserve is as it was and the cut requests simply retry |
| An unloaded model | 200 | as before (`"status": "not loaded"`), `cut: 0` |

## What a cut request sees: a retryable error, never a normal ending

The cut request ends with an error that says it is safe to send the **same request again**. It never ends as a normal
`stop` or `end_turn`, so a client can not mistake a half answer for a finished one **(test)**.

| Route | Before any answer began | After the stream began |
|---|---|---|
| `/v1/messages` (Claude Code) | HTTP **529** `{"type":"error","error":{"type":"overloaded_error","code":"vram_preempted","message":...,"retry_after_s":1}}` | an SSE `event: error` with that same body; no `message_delta`, no `message_stop` |
| `/v1/chat/completions` | HTTP **503** `{"error":{"type":"server_error","code":"vram_preempted",...}}` | `data: {"error": {...}}` then `data: [DONE]`; no chunk has a `finish_reason` |
| `/v1/responses` | HTTP 503 (Responses error body) | a `response.failed` event with code `vram_preempted` |

The HTTP errors carry `Retry-After: 1` and `x-should-retry: true`. The message names the cause: *This request was cut
because its VRAM was resized by a vram preempt (POST /v1/vram with preempt). Nothing is wrong with it: send the same
request again after 1 s (Retry-After).* Why 529 on `/v1/messages`: it is the status Anthropic's own API uses for
"overloaded", the one Claude Code's retry logic is written for (read from the installed Claude Code 2.1.289 binary).

### The server runs some requests again by itself

A request that was cut **before anything of it was sent** (still reading its prompt) is put back in the line by the
server, ahead of the requests that were waiting (the resize still goes first), and answered normally. The client sees
only a longer wait **(test)**. Up to 3 times for one request; then it gets the error. Not done when the request carries
pictures, or when the client has left.

A request cut after events were sent is not run again by the server: part of it is already with the client.

## What a client must do

1. On `529` / `503` with `error.code == "vram_preempted"`, or an error event with that code in a stream: **wait
   `Retry-After` seconds (1), then send the identical request again.** Throw away what the cut attempt delivered; the
   new attempt starts from the beginning.
2. Do not retry anything else because of this. A `409`, `400`, or a 5xx without that code is a different problem.
3. Give up after a few tries (`preempt` is rare; three cuts in a row means something keeps calling it).

`tools/strata_retry.py` does this (Python 3.8+, no dependencies; **(test)**):

```python
from strata_retry import post_with_retry
reply = post_with_retry("http://127.0.0.1:8080/v1/messages", body, headers={"x-api-key": key}, stream=True,
                        on_retry=lambda attempt, why: print("cut, sending again", attempt, why))
reply.ok, reply.status, reply.attempts, reply.events
```

or `python tools/strata_retry.py URL --body-file request.json --stream`. The same error and the same three rules apply
when a priority "now" request cuts the running one ([PRIORITY_NOW.md](PRIORITY_NOW.md)); the only addition is an
optional `"cause"` field in the error object (`"vram_preempt"` or `"priority_now"`), which a client that follows these
rules never has to read - `strata_retry` retries both the same way **(test)**.

### What happens to Claude Code

Measured on this PC (24 to 26 Nov 2026), with the installed Claude Code 2.1.289 in print mode (`claude -p`), against
Strata's real request handler with a fake CPU engine (no model, no GPU), one preempt in the middle of one request:

| Cut at | What Claude Code did |
|---|---|
| before the first token (the server ran it again by itself) | nothing visible: one answer, `Done.` |
| while thinking only | sent the request again by itself; final answer `Done.` |
| in the middle of the text | sent the request again by itself (a streamed request again); final answer `Done.` |
| in the middle of a tool call it was writing | sent the request again by itself; final answer `Done.` |

So for this version Claude Code needs nothing from Helm for a cut. What is **not** covered: interactive mode, any other
version (Helm's own worker may use another), a real model (the answers here were scripted), and clients that are not
Claude Code (Codex, the web app, scripts: use the helper above). Claude Code's code (read from the binary) retries a
`529` and an `overloaded_error` event. The same binary also has a path that keeps a partial answer and appends "Server
error mid-response. The response above may be incomplete."; I did not find out which errors take it and did not trigger
it, so **keep the error type `overloaded_error`** on `/v1/messages`: it is the one that was tested.

## What to build on (the lease and release controller)

- Shrink with `{"reserve_mib": R, "preempt": true}` when the other program needs the VRAM **now**. Add `preempt_grace_s`
  (say 20 to 30) when a short wait is acceptable: a turn that ends on its own is not lost.
- Shrink first, then start the other program; stop it, then grow with `{"reserve_mib": null}` (a plain call: nothing is
  running to cut, and it does not need `preempt`).
- Check `preempt.cut` and `preempt.outcome` in the answer; log them. `took_ms` is the time the program had to wait.
- `GET /v1/status` is unchanged. The server's window prints `VRAM preempt (cut): cut N request(s), the turn came after
  X ms, the resize took Y ms`.
- Not covered by a test or a measurement: a real Strata under a real Claude Code turn on the GPU; see the board item.
