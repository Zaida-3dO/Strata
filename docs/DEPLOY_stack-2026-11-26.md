# Deploy runbook: fair queue + loop close + VRAM preempt on Calliope (stack of 2026-11-26)

One window, one new folder, one launch. The branch `integrate/strata-stack` (fork `Zaida-3dO/Strata`) holds three
changes that were reviewed one by one and are deployed together:

| Change | What it does | Board item |
|---|---|---|
| Fair queue (+ its review fixes) | The engine's single turn goes to the waiting requests in arrival order; the log says `queue: request ... has waited 5s ...` | c9fdc6ff |
| Loop close | 256 of one token inside the thinking close the thinking with the wrap-up sentence and the model answers; Flash-Next's thinking default gets `presence_penalty 1.5` | e0cdd72a |
| VRAM preempt | `POST /v1/vram` with `"preempt": true` resizes at once and cuts the running request with a retryable error (`docs/VRAM_PREEMPT.md`) | e7c3d1b2 |

Without `preempt` a resize still waits its turn behind the running and the waiting requests; with `preempt` it goes to the
front. Nothing in this runbook was run when it was written: Strata was deliberately unloaded and a GPU benchmark was
running. It is written from `terminal\0.1.51-fork`, `launch\solo.choices.json`, the keepalive script and the memory
`reference-strata-on-calliope` (all read, none touched). Everything is PowerShell unless it says otherwise. Run the
steps in order; **every step has a gate: do not go on if it does not hold.**

```powershell
$S    = "$env:LOCALAPPDATA\Strata"                                   # C:\Users\opsij\AppData\Local\Strata
$REPO = "C:\Users\opsij\Documents\Coding\strata\strata-wt-integrate"   # any checkout whose origin is Zaida-3dO/Strata
$OLD  = "$S\terminal\0.1.51-fork"                                    # what runs today (the rollback target)
$NEW  = "$S\terminal\0.1.51-stack-2026-11-26"
$STR  = "$S\terminal\current\bin\strata.cmd"
```

## 1. Pre-checks (nothing is changed)

1. **The benchmark is finished and nothing talks to Strata.** The GPU test chat has said it is done. Then:
   ```powershell
   Get-NetTCPConnection -LocalPort 18731 -State Established -ErrorAction SilentlyContinue   # no client connected
   nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv                # only Whisper (and the desktop)
   Get-Content "$S\runtime\launch-solo.server.log" -Tail 5                                  # nothing in the last minutes
   ```
   Do not poke Strata with a request to "see": any request loads an unloaded model.
2. **Read-only list of what runs.** (Strata may be up or down; this only tells you what step 3 must have stopped.)
   ```powershell
   Get-Process strata, strata-vision -ErrorAction SilentlyContinue | Select-Object Id, ProcessName, Path
   Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'Strata\\terminal' } |
       Select-Object ProcessId, Name, CommandLine
   ```
3. **Whisper is up, first.** Strata sizes its expert cache to the VRAM that is free when it loads, so Whisper must hold
   its memory before Strata starts (memory `reference-strata-on-calliope`, 8 Nov: started after Strata, Strata fell to
   12-13 tok/s).
   ```powershell
   curl.exe -s -o NUL -w "%{http_code}`n" http://127.0.0.1:18950/health        # 200
   ```
   If it is not 200, start Whisper and wait for 200, then wait 20 s more. Immich ML is the known open case (its models,
   loaded later, page Strata out): do not run an Immich job during the checks.
4. **Pause the keepalive task** so it does not relaunch the old code while you work (it relaunches Solo after 3 failed
   health checks, about 3 minutes, through the `terminal\current` junction):
   ```powershell
   schtasks /End /TN "Strata Server"
   schtasks /Query /TN "Strata Server" /FO LIST | Select-String "Status"       # not "Running"
   ```
   If this is denied, it needs Ope's elevated shell. The trigger is "at startup": it stays stopped until it is started
   again (step 11) or the PC reboots.
5. **Back up the three config files** the launch rewrites (so the rollback is exact):
   ```powershell
   foreach ($f in "launch\solo.choices.json","launch\solo.json","install\strata-iq3_s.json") {
       Copy-Item "$S\$f" "$S\$f.bak-20261126-stack" }
   ```
   (`solo.choices.json` is what `launch --preset Solo` and the keepalive read; the other two are written again by every launch.)

## 2. Build the new folder beside the live one (never over it)

```powershell
git -C $REPO fetch origin integrate/strata-stack
$SHA = (git -C $REPO rev-parse origin/integrate/strata-stack).Trim()         # the 40-character tip you deploy
"$SHA"                                                                        # write it on the board items
robocopy $OLD $NEW /E /XD __pycache__ /NFL /NDL /NJH /NJS                    # a FULL copy; robocopy exit codes 0-7 are fine
# the changed files, byte for byte from the commit (LF endings; Python does not care)
git -C $REPO archive $SHA serve/server.py serve/fairlock.py tools/strata_retry.py tools/stack_live_check.py `
    docs/VRAM_PREEMPT.md docs/VRAM_ELASTIC.md docs/DETAILS.md docs/DEPLOY_stack-2026-11-26.md -o "$env:TEMP\stack.tar"
tar -xf "$env:TEMP\stack.tar" -C $NEW
[IO.File]::WriteAllText("$NEW\COMMIT", $SHA)                                  # it held 060317d1... (the fair queue's parent)
```

Gate (all must hold; nothing is copied from the test files on purpose):

```powershell
Get-ChildItem $NEW -Recurse -Directory -Filter __pycache__                    # prints nothing
(Get-Content "$NEW\COMMIT") -eq $SHA                                          # True
(git hash-object "$NEW\serve\server.py") -eq (git -C $REPO rev-parse "${SHA}:serve/server.py")     # True
(git hash-object "$NEW\serve\fairlock.py") -eq (git -C $REPO rev-parse "${SHA}:serve/fairlock.py") # True
(git hash-object "$NEW\strata_cli\launch\runner.py") -eq (git hash-object "$OLD\strata_cli\launch\runner.py") # True: the idle-unload patch came along
python -c "import ast,sys;[ast.parse(open(f,encoding='utf-8').read()) for f in sys.argv[1:]]" "$NEW\serve\server.py" "$NEW\serve\fairlock.py"   # no output
Test-Path $OLD                                                                # True: the old folder is untouched
```

Recursive compare, so a stale copied file cannot slip in: `$NEW\serve` must match the commit's `serve\` folder file for file
(same names, same bytes; `__pycache__` is left out).

```powershell
git -C $REPO archive $SHA serve -o "$env:TEMP\serve-commit.tar"
$CMP = "$env:TEMP\serve-commit"; Remove-Item $CMP -Recurse -Force -ErrorAction SilentlyContinue; New-Item -ItemType Directory $CMP | Out-Null
tar -xf "$env:TEMP\serve-commit.tar" -C $CMP
function Hashes($root) { Get-ChildItem $root -Recurse -File | Where-Object { $_.FullName -notmatch '__pycache__' } |
    ForEach-Object { "{0}  {1}" -f (Get-FileHash $_.FullName -Algorithm SHA256).Hash, $_.FullName.Substring($root.Length) } | Sort-Object }
Compare-Object (Hashes "$CMP\serve") (Hashes "$NEW\serve")                # prints nothing: same files, same bytes
```
The same in bash (Git Bash or Linux; `$NEW` is a path):
```sh
CMP=$(mktemp -d); git -C "$REPO" archive "$SHA" serve | tar -x -C "$CMP"
diff -r -x __pycache__ "$CMP/serve" "$NEW/serve"                          # prints nothing
```
Any line either prints is a file that is missing, extra or different: re-run the `archive` and `tar` lines above and compare again.

## 3. Stop Strata, then prove nothing is left

`launch --restart` does **nothing** when the same choices are already running, so the old code would keep serving and
the checks would test it. Always stop first. Stop with the **old** junction still in place.

```powershell
& $STR launch --stop
Start-Sleep 10
Get-Process strata, strata-vision -ErrorAction SilentlyContinue | Select-Object Id, ProcessName, Path   # nothing
Get-NetTCPConnection -LocalPort 18731 -ErrorAction SilentlyContinue                                      # nothing
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'Strata\\terminal' } | Select-Object ProcessId, Name   # nothing
```

**Gate: all three print nothing.** WMI-launched Strata runs under an elevated token, so `launch --stop` can say
"Stopped solo" and leave `strata.exe` or `strata-vision.exe` holding RAM, VRAM and folder locks. If anything is left, **stop
here**: clearing it needs Ope's *elevated* `taskkill /PID <pid> /F` for each exact pid (never by image name) or a reboot.
Do not launch on top of leftovers. Re-run the three commands after Ope has done it.

## 4. Repoint the junction (only now: Strata is stopped and idle)

```powershell
cmd /c rmdir "$S\terminal\current"                      # removes the link only; never rd /s, del or Remove-Item -Recurse on it
cmd /c mklink /J "$S\terminal\current" "$NEW"
cmd /c dir /AL "$S\terminal"                            # current [...\0.1.51-stack-2026-11-26]
Test-Path "$S\terminal\current\serve\fairlock.py"       # True
```

## 5. Launch detached (never from a Helm chat's shell) with `vram_elastic` on

A launch from a chat's shell lives in Helm's process tree and dies later without a log line (seen twice on 8 Nov), so it
goes through WMI. **This is where the decision D1 lives (section 9): the default, and what this step does, is `vram_elastic` ON.**
`--set vram-elastic` adds the engine's `--vram-elastic` flag; without it the preempt answers 400 `vram_not_elastic` and
nothing is cut. **`--set` alone lasts for this start only** (`strata_cli/launch/cmd.py`: flags next to `--preset` change this start
only; the preset file is written only when `--save-preset` is given, after the launch has run). So the command carries
`--save-preset Solo`, which writes the choices, `vram-elastic` included, back to `launch\solo.choices.json`; that file is
what the keepalive's `launch --preset Solo` reads. (For a launch without elastic, leave `--set vram-elastic` and
`--save-preset Solo` out.)

```powershell
$cmd = 'cmd.exe /c ""' + $STR + '" launch --preset Solo --set vram-elastic --save-preset Solo --yes --no-tui"'
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmd }
"{0} pid {1}" -f $r.ReturnValue, $r.ProcessId                                # 0 = created
$up = $false
1..72 | ForEach-Object {                                                       # up to 6 minutes (the model loads from disk)
    try { if ((Invoke-WebRequest http://127.0.0.1:18731/health -UseBasicParsing -TimeoutSec 5).StatusCode -eq 200) { $up = $true; break } } catch {}
    Start-Sleep 5 }
$up                                                                            # True
Select-String "$S\runtime\launch-solo.server.log" -Pattern "unloads after 900" | Select-Object -Last 1   # the idle-unload patch is on
```

**Gate: elastic is kept for good, not just for this start.** Read the saved preset and prove a keepalive relaunch would start
elastic (none of these starts anything):

```powershell
Select-String "$S\launch\solo.choices.json" -Pattern "vram-elastic"      # one line: "--vram-elastic" is in "extra"
(Get-Item "$S\launch\solo.choices.json").LastWriteTime                    # just now (this launch wrote it)
& $STR launch --preset Solo --plan                                        # changes nothing; read it for --vram-elastic
```
If the first line prints nothing, the choices were not saved: stop Strata and launch once more with the same command (check that
`--save-preset Solo` is in it). The keepalive (and any reboot) runs `launch --preset Solo` with no `--set`, so that file is the
only thing that keeps elastic on; step 8d.1 checks the running server (`"elastic": true`). Look at 8d.1 again after step 11
(the task restarted) and after any reboot. (I think `--plan` lists the engine flags; the file line is the proof.)

If it is not up after 6 minutes, read the last 40 lines of `runtime\launch-solo.server.log` and `launch\solo.engine.log`
before doing anything else; an engine that does not know a flag fails here, and that is a rollback case (section 10).

## 6. Prove the NEW code is running (not just that something answers)

Do all four; the first two cannot be satisfied by the old code.

1. **Processes.** `Get-CimInstance Win32_Process | ? { $_.CommandLine -match 'Strata\\terminal' } | select ProcessId, Name, CommandLine`
   lists the server and the engine; the junction path (`terminal\current`) hides which folder it points to, so also
   `cmd /c dir /AL "$S\terminal"` (step 4) and the next line:
2. **The new folder has compiled code, the old one has not been touched since:**
   `Get-ChildItem "$NEW\serve\__pycache__\server*.pyc" | select Name, LastWriteTime` (created by this launch) and
   `(Get-Item "$OLD\serve\__pycache__\server*.pyc").LastWriteTime` (older; the old folder is not used).
3. **A line only the new code prints.** After check 8a below the log has `queue: request <id> has waited 5s for the
   engine, K ahead of it`. (Also from the very first request after the launch: `GET /v1/status` answers; its `vram`
   block has `"elastic": true` when step 5's flag took.)
4. **The preempt fields exist** (proves the new `/v1/vram` code without cutting or resizing anything: the old code ignores fields it
   does not know and would answer 200). With the key in `$key` (never print it):
   ```powershell
   $key = (Get-Content "$S\launch\solo.choices.json" | ConvertFrom-Json).key
   curl.exe -s -w "`n%{http_code}`n" -H "Authorization: Bearer $key" -H "Content-Type: application/json" `
       -d '{\"preempt_grace_s\": 5}' http://127.0.0.1:18731/v1/vram
   ```
   Expected: HTTP **400** and `{"error": {"type": "invalid_request_error", "message": "preempt_grace_s needs \"preempt\": true"}}`
   (`serve/server.py`, the `/v1/vram` branch: a grace without `preempt: true` raises a `ValueError`, which becomes a 400 before the
   engine is touched; the log has `400 invalid request: preempt_grace_s needs "preempt": true`). The body has no `reserve_mib` and no
   `preempt`, so nothing can be resized or cut. A **200** means the old code answers. (`presence_penalty 1.5` under
   `default_generation_settings.params` in `GET /props` is a separate sanity check of the loop close; its proof is the wrap-up line in 8b.)

## 7. Smoke test

```powershell
$env:STRATA_KEY = $key
python "$NEW\tools\stack_live_check.py" messages --live   # one thinking request on /v1/messages: end_turn, an answer, no error
```
and one ordinary chat from Helm on the local model. Look in the log for errors (`Select-String "$S\runtime\launch-solo.server.log" -Pattern "Traceback|rror" | select -Last 5`).
**Gate: it answers, the speed is not obviously worse** (baseline from the memory on the 5090, Solo, IQ3_S: about 160-180 tok/s
on short answers, 145-150 in real agent work, prompt reading 4,600-6,400 tok/s; the engine log prints `N generated in X ms (Y tok/s)`
for every request: `Get-Content "$S\launch\solo.engine.log" -Tail 6`).

## 8. Live checks (in this order; each one is PASS/FAIL, record the numbers on the board items)

All use `tools\stack_live_check.py` (from the new folder, no dependencies). **It sends nothing unless you pass `--live` (the Solo
server on 127.0.0.1:18731; only then is its key read from `solo.choices.json`) or `--base URL`; the preempt check also needs
`--yes-cut-requests`. Every command below has `--live` already** (a bare run prints why and exits 2 without sending anything). A request is any
chat from a client; Helm crews must not be running on the local model during checks 8a and 8d.

**8a. Fair order** (item c9fdc6ff): one slow request and three waiters answer in arrival order.
```powershell
python "$NEW\tools\stack_live_check.py" fair --live
Select-String "$S\runtime\launch-solo.server.log" -Pattern "queue: request" | Select-Object -Last 8
```
PASS: the three waiters get their first text after the slow one ended, in the order W1, W2, W3; the log has
`queue: request <id> has waited 5s for the engine, K ahead of it` for them and `admitted after Ns` later (K = 1, 2, 3 ahead).
Known limit, not a failure: the order is per turn, not per request; a request that must also load the model or encode
images takes a new place each time.

**8b. Loop close** (item e0cdd72a). Three parts:

1. *Replay a loop.* The prompt that looped before is the L2 `parse_duration` task in
   `C:\Users\opsij\Tools\gpu-tests\tier-bench-2026-11-24\loopfix\` (`run_arms.py`, `lf.py`; the `control` arm, request fields
   identical but `presence_penalty 0` to match the old behaviour; the zeros loop hit 2 of 8 runs there). Run it until one loops:
   `python run_arms.py control 8 '{"presence_penalty": 0}'` in that folder (its `lf.py` points at Strata; edit the base
   URL/key there if needed), or the quick form from the new folder:
   `python "$NEW\tools\stack_live_check.py" loop --live --tries 8 --presence-penalty 0 [--prompt-file the-task.txt]` (it names the first
   try with a 200-character single-character run). To force one: `temperature 0 / top_k 1` plus `presence_penalty 0` on a
   prompt that makes the model count digits.
2. *What PASS looks like.* The server log has
   `[strata] the thinking repeated one token ('0') 256 times in a row: wrapping up the thinking so the model answers (kept 8 of the 256 copies in its context)`,
   then a non-empty answer and finish `stop` (before: empty content, finish `length`, log `the reply repeated one token`).
   The answer must be sensible. A healthy request must **not** show the line (an ordinary thinking request: run
   `python "$NEW\tools\stack_live_check.py" messages --live` again and grep the log).
3. *Judge the trim (decides, does not gate).* The engine's recurrent state cannot rewind, so the second pass may have to
   re-read all the thinking before the loop, not "a few tokens". Read the two prompt lines the loop request wrote:
   `Select-String "$S\launch\solo.engine.log" -Pattern "strata serve: prompt" | Select-Object -Last 4`
   (`prompt N tokens = R reused + P read in X ms`). First pass: `P` = the whole prompt. Second pass (after the wrap-up):
   - `P` small (a few hundred tokens, under ~1 s): the trim pays; keep `REPEAT_CLOSE_KEEP = 8`.
   - `P` large (more than ~10k tokens, or X above ~5 s at the 4.6-6.4k tok/s reading speed): the trim does not help; note
     it, and decide whether to drop it (keep the whole run) in the next patch. **Not a rollback reason.**
   Also compare the answer's quality with and without: repeat the same looping prompt once more.

   *Spot check of `presence_penalty 1.5` on coding and tool calls* (the reviewer's medium finding: Qwen's own card says 0
   for precise coding; evidence for 1.5 is 8 runs on one task):
   ```powershell
   python "$NEW\tools\stack_live_check.py" penalty --live --runs 3
   ```
   It runs a coding task and a tool call, 3 times each, with the default and with `presence_penalty 0`. PASS: both arms
   answer every time and the tool-call arguments parse. If the default arm is worse, or Ope would rather not: turn the
   penalty off without a code change: put `"sampling": {"presence_penalty": 0}` in the run config (or `POST /settings`, or send
   `presence_penalty: 0` per request); the loop close stays on. This is decision D2.

**8c. The `/v1/messages` route end to end** (Claude Code's route). `python "$NEW\tools\stack_live_check.py" messages --live`
(a thinking request: `stop_reason end_turn`, a text block, no `error` event, `message_stop` last) and, for the loop, the
same loop request as 8b sent to `/v1/messages` with `thinking: {type: enabled}`. Then one real Claude Code turn on the local
model with the harness that Helm uses (`tools-local\cc-env.sh` sets `ANTHROPIC_BASE_URL` and the key):
`claude -p "list the files in this folder and say how many there are"` with a tool call in it. PASS: it ends normally.

**8d. Preempt** (item e7c3d1b2):

1. *Elastic is really on.* `GET /v1/status` has `"vram": {"elastic": true, ...}`; the engine log has no
   `--vram-elastic is off: <reason>` line (`Select-String "$S\launch\solo.engine.log" -Pattern "vram.elastic"`), and its INFO line
   reports `vram_elastic=1`. **This is the part the builder could not check on the deployed 0.1.51 binary.** If the log says
   it is off, read the reason (layer split, helper caches, HIP, segment under 64 MiB), stop 8d and report it.
2. *Cut a long generation.*
   ```powershell
   python "$NEW\tools\stack_live_check.py" preempt --live --yes-cut-requests --reserve 5800
   python "$NEW\tools\stack_live_check.py" preempt --live --yes-cut-requests --reserve 5800 --grace 20     # with preempt_grace_s
   ```
   It starts a long generation (retrying through `tools\strata_retry.py`), sends `POST /v1/vram {"reserve_mib": 5800, "preempt": true}`
   after 20 streamed events, prints the answer (`outcome`, `cut`, `turn_wait_ms`, `resize_ms`, `took_ms`), the retry, and grows the
   cache back with a plain `{"reserve_mib": null}`. **Record: the resize time** (REPORT of 24 Nov: 10-36 ms for ~3.5 GiB on the
   5090; the call as a whole should come back well under a second), `preempt.cut` (1) and `outcome` (`cut`; with the grace and a
   short answer: `waited`).
3. *The cut error, by hand* (what a non-retrying client sees): start `curl.exe -N` on `/v1/messages` with a long request in one window, run
   the `POST /v1/vram` with `preempt: true` from another; the stream ends with `event: error` and `overloaded_error` /
   `vram_preempted` (no `message_stop`); `/v1/chat/completions` ends with an error chunk and `[DONE]`.
4. *Retry.* The same request sent again through `tools\strata_retry.py` succeeds (the second attempt reads its prompt from the
   prefix cache: first token in about 0.2 s). Then the real thing: a Claude Code run on the local model
   (`tools-local\cc-env.sh`, `claude -p` with a long task) while you run the preempt from a second window: PASS when Claude
   Code resends by itself and finishes (measured on a fake engine with version 2.1.289 in print mode; **not** measured on a real
   model, interactive mode or Helm's own worker).
5. `--parallel`: only unit-tested; the Solo preset has 1 slot, so nothing to try here.
6. After every preempt: `GET /v1/status` is idle, the next normal request answers, and the log has
   `VRAM preempt (cut): cut N request(s), the turn came after X ms, the resize took Y ms`.

**8d.2 Priority "now" lane** (item 31eba9c3; only if the server has it: `"priority_lane": true` in `strata-<model>.json`,
off by default - without that key skip this). The start log must carry `priority lane: on (X-Strata-Priority: now ...)`. Then:
```powershell
python "$NEW\tools\stack_live_check.py" now --live --yes-cut-requests
```
It starts a long generation, sends one short request with `X-Strata-Priority: now` mid-stream, and prints the
`GET /v1/priority/events` timeline. **PASS:** the now answered while the long request was still going (cut once,
`now_cuts` up), the long request retried through `tools\strata_retry.py`, took its resume credit (`resume_credit_hits`)
and finished with a full answer; afterwards `/v1/status` `"priority"` shows it and the server is idle. Run this only
after 8d has grown the cache back.

**8e.** Rollback criteria are in section 10.

**Known limits to watch** (from the review; none is a rollback reason):

- An MCP loop cut in round 2 can run tools twice: the client's retry sends the whole turn again, so a tool that already ran in the cut
  round runs again. Watch for a doubled tool call after a cut during an MCP tool loop.
- `--parallel` holders include the waiting requests, not only the running ones: with more than one slot the count can look
  higher than the number of requests being answered.
- A request that finishes in the same step as a preempt ends as a retryable error even though its answer was complete. Harmless: the
  client sends it again and gets the answer from the prefix cache.

## 9. Decisions for Ope

- **D1: `vram_elastic` on for good? Default: ON.** Needed for any preempt (the lease/release controller). The engine allocates the
  expert cache in 512 MiB segments instead of one block; the cost on the 5090 is not measured on this PC (the docs say "without the
  flag nothing changes"; with it: unknown, see check 7's speed). Where it lives: `launch\solo.choices.json` is the preset the
  keepalive's `launch --preset Solo` reads (`"extra"` holds `--vram-reserve-mib`, `--ple-io`, ...). `launch --set vram-elastic`
  (step 5) adds `"--vram-elastic"` to `extra` and rewrites `launch\solo.json` and `install\strata-iq3_s.json` (both derived, never
  edit those two by hand). Step 5 now carries `--save-preset Solo`, so it is saved; the check is in step 5's gate. If the line is
  missing, stop Strata and launch once more with the step 5 command. **Without it in
  `solo.choices.json`, the next keepalive relaunch or reboot starts Strata without elastic and the preempt answers 400.** To turn
  it off for good: `--set vram-elastic=off` on a launch (and the same check), or restore `solo.choices.json.bak-20261126-stack`.
- **D2: keep `presence_penalty 1.5` as the thinking default?** Decide after check 8b (coding + tool-call spot check). Default: keep;
  flip through the run config block, no code change.
- **D3: keep the trim (`REPEAT_CLOSE_KEEP`)?** Decide from the second pass's `prompt_read` in 8b. Default: keep until measured.

## 10. Rollback

**Roll back when any of these happens** (at any step after 4):

- Not healthy 6 minutes after the launch, or the engine exits at start (unknown flag, VRAM failure).
- The new code is not what answers (6.2/6.3 fail) and a relaunch does not fix it.
- 8a: a waiter is answered out of order, a request hangs, `GET /v1/status` stays busy after all requests ended, or the log shows a
  traceback from `fairlock`/`_run`.
- Speed regression of more than about 10% on the same short-answer test against the baseline in step 7, with elastic on (first try
  `--set vram-elastic=off` and the same code; if the speed returns, keep the code and drop elastic: D1 becomes off).
- The loop close fires on a healthy request (the wrap-up line without a loop) or an answer is cut off.
- A preempt leaves the engine unresponsive, a resize hangs, or a cut request is not retryable (Claude Code does not resend).

**Not** a rollback reason: the trim re-reading a lot (D3), `presence_penalty` quality (D2), a cut that Claude Code did not retry in
an untested mode (record it).

Steps (keep the new folder; nothing is deleted):

```powershell
& $STR launch --stop                                           # the junction still points at the new folder: fine
Start-Sleep 10                                                 # then the same three leftover checks as step 3; leftovers need Ope's elevated taskkill /PID
cmd /c rmdir "$S\terminal\current"
cmd /c mklink /J "$S\terminal\current" "$OLD"                  # or "$S\terminal\0.1.51" for the stock install
foreach ($f in "launch\solo.choices.json","launch\solo.json","install\strata-iq3_s.json") {
    Copy-Item "$S\$f.bak-20261126-stack" "$S\$f" -Force }     # the choices as they were before this deploy
$cmd = 'cmd.exe /c ""' + $STR + '" launch --preset Solo --yes --no-tui"'
Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmd }
```
then `/health`, `unloads after 900` in the log, and step 11. Record why on the three board items.

## 11. Restore the keepalive and record the result

```powershell
schtasks /Run /TN "Strata Server"
Get-Content "$S\tools-local\strata-keepalive.log" -Tail 4      # 'keepalive started' ... 'Whisper is up' ... 'Strata already up'
```
Then on the three board items (c9fdc6ff, e0cdd72a, e7c3d1b2): the deployed commit (`$SHA`), what each live check showed, the numbers
(resize ms, `outcome`, the second pass's `prompt_read`), the decisions D1-D3 and whether the benchmark baseline held.
Update the memory `reference-strata-on-calliope` (new folder name, `vram_elastic` in Solo, the `queue:` line, the preempt) and keep
`terminal\0.1.51-fork` as the rollback target.
