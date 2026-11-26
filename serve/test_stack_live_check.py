"""tools/stack_live_check.py sends nothing unless it is told where (and, for preempt, that cutting is fine).

CPU only: a counting HTTP server on a free port stands in for Strata; LOCALAPPDATA points at a temp folder with a
fake solo.choices.json, so the live key is never the real one.  STACK_LIVE_CHECK_SCRIPT swaps in another copy of the
script (the control run against the parent commit; the bare-run test is skipped then, because the parent's bare run
would reach the live server).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

OTHER = os.environ.get("STACK_LIVE_CHECK_SCRIPT")
SCRIPT = Path(OTHER) if OTHER else Path(__file__).resolve().parent.parent / "tools" / "stack_live_check.py"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _reply(self, body):
        self.server.seen.append((self.command, self.path, self.headers.get("Authorization")))
        raw = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/v1/models":
            self._reply({"data": [{"id": "m"}]})
        else:
            self._reply({"vram": {"elastic": False}})        # preempt then stops at "not elastic": nothing is cut

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self._reply({})


class StackLiveCheck(unittest.TestCase):
    def setUp(self):
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.srv.seen = []
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.srv.server_address[1]}"
        self.tmp = tempfile.TemporaryDirectory()
        launch = Path(self.tmp.name, "Strata", "launch")
        launch.mkdir(parents=True)
        (launch / "solo.choices.json").write_text(json.dumps({"key": "LIVE-KEY"}))
        self.env = {k: v for k, v in os.environ.items() if k != "STRATA_KEY"}
        self.env["LOCALAPPDATA"] = self.tmp.name

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()
        self.tmp.cleanup()

    def run_check(self, *args):
        return subprocess.run([sys.executable, str(SCRIPT), *args], env=self.env, capture_output=True, text=True, timeout=60)

    @unittest.skipIf(OTHER, "the parent's bare run would reach the live server")
    def test_a_bare_run_sends_nothing_and_fails(self):
        for check in ("fair", "messages", "loop", "penalty", "preempt"):
            r = self.run_check(check)
            self.assertEqual(r.returncode, 2, (check, r.stderr))
            self.assertIn("Nothing was sent", r.stderr)

    def test_preempt_needs_the_yes_even_with_a_base(self):
        r = self.run_check("preempt", "--base", self.base)
        self.assertEqual(r.returncode, 2)
        self.assertEqual(self.srv.seen, [])

    def test_an_explicit_base_works_and_does_not_read_the_live_key(self):
        r = self.run_check("preempt", "--base", self.base, "--yes-cut-requests")
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)      # FAIL on purpose: the mock is not elastic, nothing is cut
        self.assertTrue(self.srv.seen)
        self.assertTrue(all(a is None for _, _, a in self.srv.seen), self.srv.seen)

    def test_a_key_from_the_environment_is_used_with_a_base(self):
        self.env["STRATA_KEY"] = "ENV-KEY"
        self.run_check("preempt", "--base", self.base, "--yes-cut-requests")
        self.assertTrue(self.srv.seen)
        self.assertTrue(all(a == "Bearer ENV-KEY" for _, _, a in self.srv.seen), self.srv.seen)

    def test_live_reads_the_key_file(self):
        from importlib import util
        spec = util.spec_from_file_location("slc", SCRIPT)
        mod = util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.CHOICES = str(Path(self.tmp.name, "Strata", "launch", "solo.choices.json"))
        saved = os.environ.pop("STRATA_KEY", None)
        try:
            self.assertEqual(mod.Api(self.base, read_live_key=True).headers["Authorization"], "Bearer LIVE-KEY")
            self.assertNotIn("Authorization", mod.Api(self.base).headers)
        finally:
            if saved:
                os.environ["STRATA_KEY"] = saved


if __name__ == "__main__":
    unittest.main()
