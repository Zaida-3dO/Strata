"""A well-formed call naming a tool the request did not declare is still a call - never printed text.

The 2026-11-23 incident: qwen wrote `<tool_call><function=app_stage_settings>...` with the name outside the
declared set; the translator re-emitted the whole call as content, the Approve card never came, and the agent
saw its own call as prose.  A name the harness has but the turn did not advertise must become a real tool_call
with no schema, so the harness answers tool-not-found and the agent recovers.  The JSON form (`recover`) and
a call inside the thinking get the same treatment.

    python -m unittest serve.test_tool_call_undeclared -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serve.frontend import OutputParser, json_tool_call  # noqa: E402

TOOLS = [{"name": "read", "parameters": {"properties": {"path": {"type": "string"}}}}]
CALL = ('<tool_call>\n<function=app_stage_settings>\n<parameter=changes>\n{"a": 1}\n</parameter>\n</function>\n'
        '</tool_call>')
WANT = ("app_stage_settings", {"changes": {"a": 1}})


def run(text, thinking, tools, stream_tools, recover, declared_only, step):
    p = OutputParser(thinking=thinking, tools=tools, stream_tools=stream_tools, recover=recover,
                     declared_only=declared_only)
    evs = []
    for i in range(0, len(text), step):                # fed in small pieces, as a stream arrives
        evs += p.feed(text[i:i + step])
    evs += p.finish("stop")
    calls = [(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"]
    return calls, "".join(e.text for e in evs if e.kind == "content"), p


class UndeclaredName(unittest.TestCase):
    def check_call(self, text, want, thinking=False, tools=TOOLS, declared_only=False):
        for stream_tools in (False, True):
            for recover in (False, True):
                for step in (1, 3, 7, 4096):
                    with self.subTest(stream_tools=stream_tools, recover=recover, step=step):
                        calls, content, _ = run(text, thinking, tools, stream_tools, recover, declared_only, step)
                        self.assertEqual(calls, [want])
                        self.assertEqual(content.strip(), "")

    def test_unknown_name_in_answer_becomes_a_real_call(self):
        self.check_call(CALL, WANT)

    def test_no_tools_declared_at_all_declared_only(self):
        # Anthropic with no tools this turn: the model imitating its trained format still gets a call, not prose.
        self.check_call(CALL, WANT, tools=None, declared_only=True)

    def test_name_counted_for_the_server_log(self):
        p = OutputParser(thinking=False, tools=TOOLS)
        for i in range(0, len(CALL), 7):
            p.feed(CALL[i:i + 7])
        p.finish("stop")
        self.assertEqual(p.unread, 1)
        self.assertEqual(p.unread_names, ["app_stage_settings"])

    def test_call_inside_the_thinking_recovers(self):
        for stream_tools in (False, True):
            for step in (1, 7, 4096):
                with self.subTest(stream_tools=stream_tools, step=step):
                    p = OutputParser(thinking=True, tools=TOOLS, stream_tools=stream_tools)
                    evs = []
                    text = "I'll try that tool.\n" + CALL + "\n"
                    for i in range(0, len(text), step):
                        evs += p.feed(text[i:i + step])
                    evs += p.finish("stop")
                    calls = [(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"]
                    self.assertEqual(calls, [("app_stage_settings", {"changes": {"a": 1}})])

    def test_json_form_of_an_unknown_name_is_a_call(self):
        # the JSON spelling is only read as a call with tool_call_recovery on (unchanged), but the name gate is gone
        text = '<tool_call>\n{"name": "not_declared", "arguments": {"a": 1}}\n</tool_call>'
        for step in (1, 3, 7, 4096):
            with self.subTest(step=step):
                calls, content, _ = run(text, False, TOOLS, False, True, False, step)
                self.assertEqual(calls, [("not_declared", {"a": 1})])
                self.assertEqual(content.strip(), "")

    def test_declared_names_are_unchanged(self):
        text = '<tool_call>\n<function=read>\n<parameter=path>\na.py\n</parameter>\n</function>\n</tool_call>'
        self.check_call(text, ("read", {"path": "a.py"}))


class JsonToolCallGate(unittest.TestCase):
    def test_unknown_names_pass_the_gate(self):
        self.assertEqual(json_tool_call('{"name": "x", "arguments": {"a": 1}}', {}).arguments, {"a": 1})

    def test_prose_still_returns_none(self):
        self.assertIsNone(json_tool_call("not json at all", {}))
        self.assertIsNone(json_tool_call('{"arguments": {"a": 1}}', {}))
        self.assertIsNone(json_tool_call('{"name": "  ", "arguments": {}}', {}))


if __name__ == "__main__":
    unittest.main()
