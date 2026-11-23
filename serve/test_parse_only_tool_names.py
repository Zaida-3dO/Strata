"""parse_only_tool_names: names the harness has but this turn did not advertise are parse knowledge only.

Ope, 2026-11-23: the translator must not decide existence from the declared list.  Helm knows its full tool
surface and sends it with the request; the model's prompt still only ever sees the advertised subset, while
calls for the rest translate as valid calls.  The field is additive: absent or malformed, nothing changes.

    python -m unittest serve.test_parse_only_tool_names -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serve.frontend import OutputParser, parse_only_tools, with_parse_only  # noqa: E402


class RequestField(unittest.TestCase):
    def test_names_and_entries_are_parsed(self):
        got = parse_only_tools({"parse_only_tool_names": [
            "app_help",
            {"name": "app_stage_settings", "input_schema": {"properties": {"x": {"type": "string"}}}},
            {"name": "  "},
            7,
            "",
        ]})
        self.assertEqual(got, [{"name": "app_help"},
                               {"name": "app_stage_settings",
                                "parameters": {"properties": {"x": {"type": "string"}}}}])

    def test_absent_or_malformed_field_adds_nothing(self):
        self.assertEqual(parse_only_tools({}), [])
        self.assertEqual(parse_only_tools({"parse_only_tool_names": "app_help"}), [])
        self.assertEqual(parse_only_tools({"parse_only_tool_names": None}), [])

    def test_merge_keeps_declared_tools_winning_the_name_clash(self):
        declared = [{"name": "read", "parameters": {"properties": {"path": {"type": "string"}}}}]
        merged = with_parse_only(declared, {"parse_only_tool_names": ["read", "app_help"]})
        self.assertEqual(merged, declared + [{"name": "app_help"}])
        self.assertIsNone(with_parse_only(None, {}))


class ParseOnlyInAction(unittest.TestCase):
    def test_parse_only_name_is_accepted_with_its_schema(self):
        # "stage" is not in the advertised tools but arrives with a schema: the "42" stays a STRING, proving
        # the parse-only entry travelled to the translator with its parameter types.
        tools = with_parse_only([{"name": "read", "parameters": {"properties": {"path": {"type": "string"}}}}],
                                {"parse_only_tool_names": [{"name": "stage",
                                                            "parameters": {"properties": {
                                                                "value": {"type": "string"}}}}]})
        text = ('<tool_call>\n<function=stage>\n<parameter=value>\n42\n</parameter>\n</function>\n</tool_call>')
        p = OutputParser(thinking=False, tools=tools, declared_only=True)
        evs = []
        for i in range(0, len(text), 5):
            evs += p.feed(text[i:i + 5])
        evs += p.finish("stop")
        calls = [(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"]
        self.assertEqual(calls, [("stage", {"value": "42"})])   # "42" stayed a string: the schema was seen

    def test_advertised_prompt_surface_is_untouched_by_the_helper(self):
        # the merge only ever feeds the translator; the prompt side keeps calling prepare() with the request's
        # own tools - assert the declared list is not mutated by with_parse_only
        declared = [{"name": "read"}]
        with_parse_only(declared, {"parse_only_tool_names": ["extra"]})
        self.assertEqual(declared, [{"name": "read"}])


if __name__ == "__main__":
    unittest.main()
