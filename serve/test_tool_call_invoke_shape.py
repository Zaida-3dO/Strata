"""The Claude Code wire shape - function_calls / invoke / parameter name="..." - translates, never prints.

Seen live 2026-11-23: a local model imitating Claude wrote the whole call in Claude's markup (with the trainer's
`antml:` namespace on the tags); Strata had no grammar for it and it landed in the transcript as text.  The
translator now reads both plain and antml spellings, with and without declared tools, types values against the
tool's schema when it has one, and keeps a quoted example inside a code fence a quote.

    python -m unittest serve.test_tool_call_invoke_shape -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serve.frontend import OutputParser  # noqa: E402

PLAIN = ('<function_calls><invoke name="search"><parameter name="query">\nsolar flux\n</parameter>'
         '</invoke></function_calls>')
ANTML = ('<antml:function_calls>\n<antml:invoke name="get_weather">\n<antml:parameter name="city">\n'
         'Oslo\n</antml:parameter>\n</antml:invoke>\n</antml:function_calls>')
TOOLS = [{"name": "run", "parameters": {"properties": {"count": {"type": "integer"},
                                                       "code": {"type": "string"}}}}]


def run(text, tools, step):
    p = OutputParser(thinking=False, tools=tools, stream_tools=False, declared_only=True)
    evs = []
    for i in range(0, len(text), step):                # fed in small pieces, as a stream arrives
        evs += p.feed(text[i:i + step])
    evs += p.finish("stop")
    return ([(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"],
            "".join(e.text for e in evs if e.kind == "content"))


class InvokeShape(unittest.TestCase):
    def check(self, text, want, tools=None):
        for step in (1, 3, 7, 4096):
            with self.subTest(step=step):
                calls, content = run(text, tools, step)
                self.assertEqual(calls, [want])
                self.assertEqual(content.strip(), "")

    def test_plain_shape_is_a_call(self):
        self.check(PLAIN, ("search", {"query": "solar flux"}))

    def test_antml_namespace_is_a_call(self):
        self.check(ANTML, ("get_weather", {"city": "Oslo"}))

    def test_unknown_name_still_becomes_a_call(self):
        self.check(PLAIN, ("search", {"query": "solar flux"}), tools=TOOLS)   # not declared: harness answers

    def test_schema_types_the_values(self):
        text = ('<function_calls><invoke name="run"><parameter name="count">\n5\n</parameter>\n'
                '<parameter name="code">\nx = 1\n</parameter></invoke></function_calls>')
        self.check(text, ("run", {"count": 5, "code": "x = 1"}), tools=TOOLS)

    def test_quoted_in_a_code_fence_stays_text(self):
        text = "The format looks like this:\n```\n" + PLAIN + "\n```\nand that runs it."
        for step in (1, 7, 4096):
            with self.subTest(step=step):
                calls, content = run(text, None, step)
                self.assertEqual(calls, [])
                self.assertEqual(content, text)

    def test_prose_around_the_block_stays_text(self):
        text = "Sending it now.\n" + PLAIN + "\nWatch this space."
        for step in (1, 7, 4096):
            with self.subTest(step=step):
                calls, content = run(text, None, step)
                self.assertEqual(calls, [("search", {"query": "solar flux"})])
                self.assertEqual(content.strip(), "Sending it now.Watch this space.")   # newlines around a call are eaten, as everywhere else

    def test_a_value_holding_the_format_still_ends_at_the_real_closer(self):
        text = ('<function_calls><invoke name="run"><parameter name="code">\n'
                'example:\n</parameter>\n</invoke>\nstill the value\n</parameter>'
                '</invoke></function_calls>')
        for step in (1, 3, 7, 4096):
            with self.subTest(step=step):
                calls, content = run(text, TOOLS, step)
                self.assertEqual(calls, [("run", {"code": "example:\n</parameter>\n</invoke>\nstill the value"})])
                self.assertEqual(content.strip(), "")

    def test_two_invokes_in_one_answer(self):
        text = PLAIN + "\n" + ANTML
        calls, content = run(text, None, 4096)
        self.assertEqual(calls, [("search", {"query": "solar flux"}), ("get_weather", {"city": "Oslo"})])


if __name__ == "__main__":
    unittest.main()
