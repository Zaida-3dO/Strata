#!/usr/bin/env python3
"""Live checks for the fair queue, the loop close and the VRAM preempt, against a running Strata (no dependencies).

    python tools/stack_live_check.py fair                    one slow request + three waiters: they must answer in arrival order
    python tools/stack_live_check.py messages                the /v1/messages route end to end (thinking on)
    python tools/stack_live_check.py loop [--tries 6] [--presence-penalty 0] [--prompt-file P]
                                                             send thinking requests until one loops; watch the server log
    python tools/stack_live_check.py penalty [--runs 3]      a coding task and a tool call, presence_penalty default vs 0
    python tools/stack_live_check.py preempt [--reserve 5800] [--grace 0]
                                                             cut a long generation with POST /v1/vram preempt, retry it

--base (default http://127.0.0.1:18731), --key (or the STRATA_KEY variable, or the "key" in launch/solo.choices.json
beside the install: never printed).  Every check prints PASS or FAIL for what it can see from outside; the server log
(runtime/launch-solo.server.log) says the rest.  Nothing here changes a setting: preempt resizes the expert cache and
grows it back at the end.
"""
import argparse
import http.client
import json
import os
import sys
import threading
import time
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CHOICES = os.path.expandvars(r"%LOCALAPPDATA%\Strata\launch\solo.choices.json")


class Api:
    def __init__(self, base, key=None):
        u = urllib.parse.urlparse(base)
        self.host, self.port = u.hostname, u.port or 80
        if not key:
            key = os.environ.get("STRATA_KEY")
        if not key and os.path.exists(CHOICES):
            key = json.load(open(CHOICES)).get("key")
        self.headers = {"Content-Type": "application/json"}
        if key:
            self.headers.update({"Authorization": "Bearer " + key, "x-api-key": key})
        self.base = base.rstrip("/")
        self._model = None

    def call(self, method, path, body=None, timeout=600):
        c = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
        c.request(method, path, json.dumps(body) if body is not None else None, self.headers)
        return c, c.getresponse()

    def json(self, method, path, body=None, timeout=600):
        c, r = self.call(method, path, body, timeout)
        try:
            raw = r.read().decode("utf-8", "replace")
            return r.status, (json.loads(raw) if raw.startswith(("{", "[")) else raw)
        finally:
            c.close()

    def model(self):
        if self._model is None:
            self._model = self.json("GET", "/v1/models")[1]["data"][0]["id"]
        return self._model

    def stream(self, path, body, on_event=None):
        """POST a streamed request; -> (status, [(event name or None, data)]).  on_event is called as each one arrives."""
        c, r = self.call("POST", path, body)
        events, name = [], None
        try:
            if r.status != 200:
                return r.status, [(None, r.read().decode("utf-8", "replace"))]
            while True:
                line = r.fp.readline()
                if not line:
                    break
                line = line.decode("utf-8", "replace").rstrip("\n")
                if line.startswith("event:"):
                    name = line[6:].strip()
                elif line.startswith("data:"):
                    raw = line[5:].strip()
                    data = json.loads(raw) if raw.startswith("{") else raw
                    events.append((name, data))
                    if on_event:
                        on_event(name, data)
                    name = None
        finally:
            c.close()
        return r.status, events


def chat(api, prompt, max_tokens=64, thinking=False, **extra):
    return {"model": api.model(), "stream": True, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": thinking},
            "messages": [{"role": "user", "content": prompt}], **extra}


def text_of(events):
    out = ""
    for _, d in events:
        if isinstance(d, dict):
            for ch in d.get("choices", []):
                out += ch.get("delta", {}).get("content") or ""
            if d.get("type") == "content_block_delta":
                out += d["delta"].get("text", "")
    return out


def verdict(ok, what):
    print(("PASS  " if ok else "FAIL  ") + what)
    return 0 if ok else 1


def check_fair(api, a):
    seen, t0 = {}, time.monotonic()
    first = threading.Event()

    def run(tag, prompt, max_tokens, gate=None):
        def note(name, data):
            if not text_of([(name, data)]):                    # the role chunk and keep-alive pings are not an answer
                return
            now = time.monotonic() - t0
            seen.setdefault(tag, {"first": now})["last"] = now
            if gate:
                gate.set()
        st, ev = api.stream("/v1/chat/completions", chat(api, prompt, max_tokens), note)
        seen.setdefault(tag, {"first": time.monotonic() - t0, "last": time.monotonic() - t0}).update(
            status=st, text=text_of(ev))
    slow = threading.Thread(target=run, args=("R0", "Count from 1 to 600, one number per line, nothing else.", 1500, first))
    slow.start()
    if not first.wait(120):
        return verdict(False, "the slow request never started streaming")
    waiters = []
    for i in (1, 2, 3):
        w = threading.Thread(target=run, args=(f"W{i}", f"Reply with exactly: W{i}", 16))
        w.start()
        waiters.append(w)
        time.sleep(0.4)                              # arrival order is W1, W2, W3
    for t in [slow] + waiters:
        t.join(600)
    for tag in ("R0", "W1", "W2", "W3"):
        d = seen[tag]
        print(f"  {tag}: first text {d['first']:6.1f} s, last {d['last']:6.1f} s, status {d.get('status')}  {d.get('text', '')[:30]!r}")
    print("  now look in runtime/launch-solo.server.log for 'queue: request ... has waited 5s for the engine, K ahead of it'")
    firsts = [seen[f"W{i}"]["first"] for i in (1, 2, 3)]
    in_order = firsts == sorted(firsts) and len(set(firsts)) == 3
    behind = firsts[0] >= seen["R0"]["last"] - 1.0          # no waiter was answered while the slow one still ran
    return verdict(in_order and behind and all(seen[k].get("status") == 200 for k in seen),
                   "the waiters were answered after the slow request, in arrival order (W1, W2, W3)")


def check_messages(api, a):
    body = {"model": api.model(), "max_tokens": 4000, "stream": True,
            "thinking": {"type": "enabled", "budget_tokens": 2000},
            "messages": [{"role": "user", "content": "What is 17 * 23? Answer in one sentence."}]}
    st, ev = api.stream("/v1/messages", body)
    names = [n for n, _ in ev]
    stop = [d["delta"].get("stop_reason") for n, d in ev if n == "message_delta"]
    text = text_of(ev)
    print(f"  status {st}, events {len(ev)}, stop_reason {stop}, text {text[:80]!r}")
    return verdict(st == 200 and stop[-1:] == ["end_turn"] and text.strip() and "error" not in names
                   and names[-1] == "message_stop", "/v1/messages: a thinking request ends end_turn with an answer")


def check_loop(api, a):
    prompt = (open(a.prompt_file, encoding="utf-8").read() if a.prompt_file else
              "Count the digits of the number 2**4000 one digit at a time, writing each digit on its own line while you think, "
              "then say how many there are.")
    extra = {} if a.presence_penalty is None else {"presence_penalty": a.presence_penalty}
    bad = 0
    for i in range(a.tries):
        t0 = time.monotonic()
        st, ev = api.stream("/v1/chat/completions", chat(api, prompt, a.max_tokens, True, **extra))
        reasoning = "".join((ch.get("delta", {}).get("reasoning_content") or "") for _, d in ev if isinstance(d, dict)
                            for ch in d.get("choices", []))
        text = text_of(ev)
        fin = [ch.get("finish_reason") for _, d in ev if isinstance(d, dict) for ch in d.get("choices", [])
               if ch.get("finish_reason")]
        run = best = 0
        prev = None
        for ch in reasoning:
            run = run + 1 if ch == prev else 1
            prev = ch
            best = max(best, run)
        print(f"  try {i + 1}: {time.monotonic() - t0:5.0f} s  finish {fin[-1:]}  reasoning {len(reasoning)} chars "
              f"(longest single-character run {best})  answer {len(text)} chars")
        if best >= 200:
            print("    -> this run looped; the server log must show 'the thinking repeated one token ... wrapping up' and "
                  "the engine log the second pass's prompt_read / reused")
            bad += verdict(fin[-1:] == ["stop"] and len(text) > 0, "the looping run still ended with an answer")
            return bad
    print("  no try looped; repeat with --tries more, --presence-penalty 0 (the old default), or a prompt that looped before")
    return 0


def check_penalty(api, a):
    code = ("Write a Python function parse_duration(s) that turns strings like '1h30m', '45s' and '2d' into seconds, "
            "with a few unit tests. Reply with code only.")
    tools = [{"type": "function", "function": {"name": "write_file", "description": "Write a file",
              "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                             "required": ["path", "content"]}}}]
    bad = 0
    for arm, extra in (("default (1.5 for thinking)", {}), ("presence_penalty 0", {"presence_penalty": 0})):
        ok_code = ok_call = 0
        for i in range(a.runs):
            t0 = time.monotonic()
            _, ev = api.stream("/v1/chat/completions", chat(api, code, 3000, True, **extra))
            t = text_of(ev)
            ok_code += "def parse_duration" in t
            body = chat(api, "Create hello.py containing print('hi') using the tool.", 2000, True, tools=tools,
                        tool_choice="auto", **extra)
            _, ev2 = api.stream("/v1/chat/completions", body)
            args = ""
            for _, d in ev2:
                if isinstance(d, dict):
                    for ch in d.get("choices", []):
                        for tc in ch.get("delta", {}).get("tool_calls") or []:
                            args += (tc.get("function") or {}).get("arguments") or ""
            try:
                ok_call += bool(json.loads(args).get("content"))
            except ValueError:
                pass
            print(f"  [{arm}] run {i + 1}: {time.monotonic() - t0:4.0f} s, code answer {'ok' if 'def parse_duration' in t else 'MISSING'}, "
                  f"tool call {'ok' if ok_call else 'not valid yet'}")
        print(f"  [{arm}] code ok {ok_code}/{a.runs}, tool call ok {ok_call}/{a.runs}")
        bad += 0 if ok_code == a.runs and ok_call == a.runs else 1
    return verdict(bad == 0, "both arms answered the coding task and made a valid tool call every time "
                             "(a difference between arms is Ope's call, not a failure of the code)")


def check_preempt(api, a):
    from strata_retry import post_with_retry
    st, status = api.json("GET", "/v1/status")
    elastic = status.get("vram", {}).get("elastic") if isinstance(status, dict) else None
    print(f"  /v1/status vram: {status.get('vram') if isinstance(status, dict) else status}")
    if not elastic:
        return verdict(False, "the engine is not elastic (vram_elastic is not on): a preempt would be refused")
    body = chat(api, "Write a very long story about a lighthouse keeper. Keep going for many pages.", 4000)
    body["model"] = api.model()
    got, retried, result = [], [], {}
    headers = dict(api.headers)

    def go():
        result["reply"] = post_with_retry(api.base + "/v1/chat/completions", body, headers=headers, stream=True,
                                          on_retry=lambda n, why: retried.append((time.monotonic(), why)),
                                          on_event=lambda n, d: got.append(time.monotonic()), timeout=900)
    t = threading.Thread(target=go)
    t.start()
    end = time.monotonic() + 180
    while len(got) < 20 and time.monotonic() < end:
        time.sleep(0.05)
    if len(got) < 20:
        return verdict(False, "the long request never streamed")
    req = {"reserve_mib": a.reserve, "preempt": True}
    if a.grace:
        req["preempt_grace_s"] = a.grace
    t0 = time.monotonic()
    code, out = api.json("POST", "/v1/vram", req)
    took = time.monotonic() - t0
    print(f"  POST /v1/vram {req} -> {code} in {took * 1000:.0f} ms: {json.dumps(out.get('preempt') if isinstance(out, dict) else out)}")
    print(f"  expert cache after: {out.get('expert_cache_mib') if isinstance(out, dict) else '?'} of "
          f"{out.get('expert_cache_full_mib') if isinstance(out, dict) else '?'} MiB, vram_free_mib "
          f"{out.get('vram_free_mib') if isinstance(out, dict) else '?'}")
    t.join(900)
    reply = result.get("reply")
    print(f"  the request: {len(retried)} retry(ies); final status {getattr(reply, 'status', None)}, attempts "
          f"{getattr(reply, 'attempts', None)}; the cut said: {retried[0][1] if retried else '(not cut)'}")
    gc, gout = api.json("POST", "/v1/vram", {"reserve_mib": None})
    print(f"  grow back (plain POST /v1/vram, reserve null) -> {gc}, expert cache "
          f"{gout.get('expert_cache_mib') if isinstance(gout, dict) else '?'} MiB")
    return (verdict(code == 200 and isinstance(out, dict) and out.get("preempt", {}).get("outcome") in ("cut", "waited"),
                    "the preempt answered 200 with an outcome")
            + verdict(took < 2.0 or a.grace > 0, "the resize call came back in under 2 s (not applicable with a grace)")
            + verdict(bool(reply) and reply.ok and (retried or out.get("preempt", {}).get("outcome") == "waited"),
                      "the cut request was sent again and got its answer (or ended within the grace)"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("check", choices=["fair", "messages", "loop", "penalty", "preempt"])
    p.add_argument("--base", default="http://127.0.0.1:18731")
    p.add_argument("--key")
    p.add_argument("--tries", type=int, default=6)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=16384, dest="max_tokens")
    p.add_argument("--presence-penalty", type=float, default=None, dest="presence_penalty")
    p.add_argument("--prompt-file")
    p.add_argument("--reserve", type=int, default=5800)
    p.add_argument("--grace", type=float, default=0)
    a = p.parse_args()
    api = Api(a.base, a.key)
    fails = {"fair": check_fair, "messages": check_messages, "loop": check_loop, "penalty": check_penalty,
             "preempt": check_preempt}[a.check](api, a)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
