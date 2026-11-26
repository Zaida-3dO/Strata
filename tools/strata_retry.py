#!/usr/bin/env python3
"""POST one request to Strata and send it again when a VRAM preempt cut it (no dependencies, Python 3.8+).

POST /v1/vram with "preempt": true cuts the request that is running so the resize happens at once.  The cut request
ends with an error that says it is safe to send it again (docs/VRAM_PREEMPT.md):

    HTTP 529 {"type": "error", "error": {"type": "overloaded_error", "code": "vram_preempted", ...}}   (/v1/messages)
    HTTP 503 {"error": {"type": "server_error", "code": "vram_preempted", ...}}                       (the other routes)
    both with Retry-After and x-should-retry: true; or, once a stream has started, the same body as an SSE event
    ("event: error" on /v1/messages, "data: {"error": ...}" on /v1/chat/completions).

This helper sends the same request again after the wait the server asked for.  It retries only that error (and
Anthropic's own `overloaded_error`): a 400, a 401, a 409 model_busy or a 5xx without it is returned to the caller as it is.

    from strata_retry import post_with_retry
    reply = post_with_retry("http://127.0.0.1:8080/v1/messages", body, headers={"x-api-key": key}, stream=True)
    reply.status, reply.attempts, reply.events          # events: [(event name or None, parsed data), ...]

    python tools/strata_retry.py http://127.0.0.1:8080/v1/chat/completions --body-file request.json [--stream]

A streamed answer that was cut is discarded and asked for again from the start: `on_retry(attempt, reason)` tells the
caller to throw away what it showed (`on_event` is called for every event as it arrives, including those of an attempt
that is cut).
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request

RETRY_CODE = "vram_preempted"          # the error's `code`
DEFAULT_WAIT_S = 1.0                   # when the server gave no Retry-After


class Reply:
    """What the last attempt returned: HTTP status, lower-case headers, the body (bytes), the SSE events of a streamed
    answer, how many attempts were made and how long the helper waited between them."""

    def __init__(self, status, headers, body, events, attempts, waited_s):
        self.status, self.headers, self.body, self.events = status, headers, body, events
        self.attempts, self.waited_s = attempts, waited_s

    @property
    def ok(self):
        return 200 <= self.status < 300 and error_of(self) is None

    def json(self):
        return json.loads(self.body.decode("utf-8"))

    def __repr__(self):
        return f"Reply(status={self.status}, attempts={self.attempts}, events={len(self.events)})"


class Gave_up(Exception):
    pass


def error_of(reply):
    """The error object a reply carries, or None: from the JSON body of an error status, or from an `error` event or
    chunk of a stream that began with 200.  (Both Anthropic's {"type": "error", "error": {...}} and OpenAI's
    {"error": {...}} keep it under "error".)"""
    for data in [e[1] for e in reply.events if isinstance(e[1], dict)] or [_body_json(reply.body)]:
        err = data.get("error") if isinstance(data, dict) else None
        if isinstance(err, dict):
            return err
    return None


def _body_json(body):
    try:
        return json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError, AttributeError):
        return None


def retry_wait(reply):
    """None when the reply is not a VRAM-preempt cut (do not retry); else the seconds to wait before the next try."""
    err = error_of(reply)
    cut = isinstance(err, dict) and (err.get("code") == RETRY_CODE or (
        err.get("type") == "overloaded_error" and reply.status in (0, 200, 529)))
    if reply.status >= 500 and reply.headers.get("x-should-retry") == "true" and isinstance(err, dict):
        cut = cut or err.get("code") == RETRY_CODE
    if not cut:
        return None
    for value in (reply.headers.get("retry-after"), err.get("retry_after_s")):
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            continue
    return DEFAULT_WAIT_S


def _parse_sse(lines_iter, on_event):
    """(event name or None, parsed data) for every event of an SSE body, read line by line as it arrives."""
    events, name, data = [], None, []
    for raw in lines_iter:
        line = raw.decode("utf-8", "replace").rstrip("\r\n")
        if line == "":
            if data:
                text = "\n".join(data)
                try:
                    parsed = json.loads(text)
                except ValueError:
                    parsed = text                      # "[DONE]"
                events.append((name, parsed))
                if on_event is not None:
                    on_event(name, parsed)
            name, data = None, []
        elif line.startswith(":"):
            continue                                   # keep-alive comment
        elif line.startswith("event:"):
            name = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())
    return events


def post_once(url, body, headers=None, stream=False, timeout=600.0, on_event=None):
    """One attempt.  -> Reply (attempts=1).  An HTTP error status is a Reply, not an exception."""
    data = body if isinstance(body, (bytes, bytearray)) else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, method="POST",
                                     headers={"Content-Type": "application/json", **(headers or {})})
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as e:
        response = e
    with response:
        status = response.status if hasattr(response, "status") else response.code
        hdrs = {k.lower(): v for k, v in response.headers.items()}
        if stream and status == 200 and "text/event-stream" in hdrs.get("content-type", ""):
            events = _parse_sse(iter(response.readline, b""), on_event)
            return Reply(status, hdrs, b"", events, 1, 0.0)
        return Reply(status, hdrs, response.read(), [], 1, 0.0)


def post_with_retry(url, body, headers=None, stream=False, attempts=6, max_wait_s=30.0, timeout=600.0,
                    on_event=None, on_retry=None, sleep=time.sleep):
    """POST `body` (a dict, or bytes already JSON) and send it again while the server says a VRAM preempt cut it.
    At most `attempts` tries; each wait is the server's Retry-After, at most `max_wait_s`.  -> the last Reply: when it
    is still the cut error after the last try, the caller sees that error (check `reply.ok`)."""
    waited = 0.0
    for attempt in range(1, max(1, attempts) + 1):
        reply = post_once(url, body, headers, stream, timeout, on_event)
        reply.attempts, reply.waited_s = attempt, waited
        wait = retry_wait(reply)
        if wait is None or attempt >= attempts:
            return reply
        wait = min(wait, max_wait_s)
        if on_retry is not None:
            on_retry(attempt, (error_of(reply) or {}).get("message", "cut by a VRAM preempt"))
        sleep(wait)
        waited += wait
    return reply                                       # unreachable: the loop returns at the last attempt


def main(argv=None):
    p = argparse.ArgumentParser(description="POST a request to Strata and retry it when a VRAM preempt cut it.")
    p.add_argument("url", help="for example http://127.0.0.1:8080/v1/messages")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--body", help="the JSON request body")
    group.add_argument("--body-file", help="a file with the JSON request body")
    p.add_argument("-H", "--header", action="append", default=[], help='"Name: value" (an API key, for example)')
    p.add_argument("--stream", action="store_true", help="the request streams: read it as server-sent events")
    p.add_argument("--attempts", type=int, default=6)
    p.add_argument("--max-wait", type=float, default=30.0, help="the longest single wait between tries, in seconds")
    a = p.parse_args(argv)
    raw = a.body if a.body is not None else open(a.body_file, "rb").read()
    headers = dict(h.split(":", 1) for h in a.header)
    headers = {k.strip(): v.strip() for k, v in headers.items()}

    def say_retry(attempt, why):
        print(f"[strata_retry] attempt {attempt} was cut ({why}); sending it again", file=sys.stderr)

    reply = post_with_retry(a.url, raw.encode() if isinstance(raw, str) else raw, headers, a.stream, a.attempts,
                            a.max_wait, on_retry=say_retry)
    if reply.events:
        for name, data in reply.events:
            print(("event: %s\n" % name if name else "") + "data: " + (data if isinstance(data, str) else json.dumps(data)))
    else:
        sys.stdout.write(reply.body.decode("utf-8", "replace") + "\n")
    print(f"[strata_retry] status {reply.status} after {reply.attempts} attempt(s)", file=sys.stderr)
    return 0 if reply.ok else 1


if __name__ == "__main__":
    sys.exit(main())
