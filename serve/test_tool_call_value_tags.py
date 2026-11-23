"""A parameter value may quote the call format itself: the closer is found by the parameter NAME, not the tag.

Seen live 2026-11-23: a Write whose content documents the call format got truncated - `param_end` closed the
value at the first literal `</parameter>` whose follower looked right, and the tool ran with half the file.
The name check (#210 kept and sharpened): the follower must open a parameter this tool has and this call has
not written yet, or close the function in a way that can really end it.  And finish(): a call the cut caught
inside no longer dumps its raw text - it becomes a real call with the parameters that closed whole, so the
harness answers tool-not-found or a schema error and the agent recovers.

    python -m unittest serve.test_tool_call_value_tags -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serve.frontend import OutputParser, param_end  # noqa: E402

TOOLS = [{"name": "write", "parameters": {"properties": {"path": {"type": "string"},
                                                         "content": {"type": "string"}}}}]


def run(text, step):
    p = OutputParser(thinking=False, tools=TOOLS, stream_tools=True)
    evs = []
    for i in range(0, len(text), step):               # fed in small pieces, as a stream arrives
        evs += p.feed(text[i:i + step])
    evs += p.finish("stop")
    return ([(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"],
            "".join(e.text for e in evs if e.kind == "content"))


class ValueTags(unittest.TestCase):
    def test_example_reusing_a_used_parameter_name_does_not_close_the_value(self):
        text = ('<tool_call>\n<function=write>\n<parameter=path>\nb.py\n</parameter>\n<parameter=content>\n'
                'sample:\n</parameter>\n<parameter=path>\nx\n</parameter>\nend\n</parameter>\n</function>\n</tool_call>')
        want = {"path": "b.py", "content": "sample:\n</parameter>\n<parameter=path>\nx\n</parameter>\nend"}
        for step in (1, 3, 7, 4096):
            with self.subTest(step=step):
                calls, content = run(text, step)
                self.assertEqual(calls, [("write", want)])
                self.assertEqual(content.strip(), "")

    def test_literal_function_closer_inside_the_value_with_more_text_does_not_end_the_call(self):
        text = ('<tool_call>\n<function=write>\n<parameter=path>\nb.py\n</parameter>\n<parameter=content>\n'
                'line1\n</parameter>\n</function>\nline2\n</parameter>\n</function>\n</tool_call>')
        want = {"path": "b.py", "content": "line1\n</parameter>\n</function>\nline2"}
        for step in (1, 3, 7, 4096):
            with self.subTest(step=step):
                calls, content = run(text, step)
                self.assertEqual(calls, [("write", want)])
                self.assertEqual(content.strip(), "")

    def test_undeclared_tool_keeps_the_loose_rule_but_still_calls(self):
        text = ('<tool_call>\n<function=mystery>\n<parameter=x>\nplain text\n</parameter>\n</function>\n'
                '</tool_call>')
        for step in (1, 7, 4096):
            with self.subTest(step=step):
                calls, content = run(text, step)
                self.assertEqual(calls, [("mystery", {"x": "plain text"})])
                self.assertEqual(content.strip(), "")

    def test_param_end_name_check_unit(self):
        def conf(props, used):
            from serve.frontend import _param_confirmer
            return _param_confirmer(props, used)
        props, used = {"content": {}, "path": {}}, {"path"}
        # follower names a parameter already used by this call: part of the value
        self.assertEqual(-1, param_end("v\n</parameter>\n<parameter=path>\nrest", confirm=conf(props, used)))
        # follower names a fresh declared parameter: the value ends there
        self.assertEqual(2, param_end("v\n</parameter>\n<parameter=content>\nnext", confirm=conf(props, used)))
        # follower closes the function and a real closer comes later: rejected, and the text ends undecided
        self.assertEqual(-2, param_end("v\n</parameter>\n</function>\nmore\n</parameter>", confirm=conf(props, used)))
        # prose after the function closer with no closer ever coming: undecided until final
        self.assertEqual(-2, param_end("v\n</parameter>\n</function>\nand prose", confirm=conf(props, used)))
        self.assertEqual(2, param_end("v\n</parameter>\n</function>\nand prose", final=True,
                                      confirm=conf(props, used)))


class CutCalls(unittest.TestCase):
    def test_cut_call_becomes_a_call_with_the_closed_parameters_only(self):
        text = ('<tool_call>\n<function=write>\n<parameter=path>\nb.py\n</parameter>\n<parameter=content>\n'
                'half a file')
        for step in (1, 7, 4096):
            for reason in ("length", "cancel", "disconnect"):
                with self.subTest(step=step, reason=reason):
                    p = OutputParser(thinking=False, tools=TOOLS, stream_tools=False)
                    evs = []
                    for i in range(0, len(text), step):
                        evs += p.feed(text[i:i + step])
                    evs += p.finish(reason)
                    self.assertEqual([(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"],
                                     [("write", {"path": "b.py"})])
                    self.assertEqual("".join(e.text for e in evs if e.kind == "content").strip(), "")

    def test_natural_stop_quoting_the_format_stays_text(self):
        text = "Format: <tool_call>\n<function=read>\n<parameter=path>\na.py\n</parameter>\n</function>\nexplained."
        p = OutputParser(thinking=False, tools=[{"name": "read",
                                                 "parameters": {"properties": {"path": {"type": "string"}}}}],
                         stream_tools=True)
        evs = []
        for i in range(0, len(text), 7):
            evs += p.feed(text[i:i + 7])
        evs += p.finish("stop")
        self.assertEqual([e for e in evs if e.kind == "tool_call"], [])
        self.assertEqual("".join(e.text for e in evs if e.kind == "content"), text)

    def test_cut_invoke_block_becomes_a_call(self):
        text = ('<function_calls><invoke name="search"><parameter name="query">\nsolar flux</parameter>'
                '<parameter name="deep">\ntr')
        p = OutputParser(thinking=False, tools=None, declared_only=True)
        evs = []
        for i in range(0, len(text), 5):
            evs += p.feed(text[i:i + 5])
        evs += p.finish("length")
        self.assertEqual([(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"],
                         [("search", {"query": "solar flux"})])
        self.assertEqual("".join(e.text for e in evs if e.kind == "content").strip(), "")

    def test_cut_before_the_name_arrives_stays_text(self):
        p = OutputParser(thinking=False, tools=None, declared_only=True)
        p.feed("<function_calls><invoke name=\"sea")
        evs = p.finish("length")
        self.assertEqual([e for e in evs if e.kind == "tool_call"], [])
        self.assertIn("function_calls", "".join(e.text for e in evs if e.kind == "content"))


if __name__ == "__main__":
    unittest.main()
