import unittest

from trace_compression import (
    collapse_aliases,
    compress_trace,
    expand_aliases,
)


class TraceCompressionTests(unittest.TestCase):
    def test_short_input_passes_through(self):
        text = "Compiling alpha v1.0\nCompiling beta v2.0\n"
        self.assertEqual(compress_trace(text).compressed, text)

    def test_template_run_reduces_tokens(self):
        lines = [f"Compiling crate_{i} v1.0.{i}" for i in range(20)]
        text = "\n".join(lines)
        result = compress_trace(text)
        self.assertLess(result.compressed_tokens, result.original_tokens)
        self.assertIn(lines[0], result.compressed)
        self.assertIn(lines[-1], result.compressed)
        for i in range(1, 19):
            self.assertIn(f"crate_{i}", result.compressed)
            self.assertIn(f"v1.0.{i}", result.compressed)

    def test_progress_collapse_is_visible(self):
        text = "\n".join(
            f"download package {pct}% complete" for pct in range(0, 101, 10)
        )
        result = compress_trace(text)
        self.assertIn("intermediate progress steps", result.compressed)
        self.assertIn("0%->100%", result.compressed)

    def test_mismatched_shapes_are_not_forced(self):
        text = (
            "Compiling alpha v1.0\n"
            "different line with extra fields here\n"
            "Compiling beta v2.0"
        )
        self.assertEqual(compress_trace(text).compressed, text)

    def test_alias_round_trip(self):
        digest = "sha256:a1b2c3d4e5f67890abcdef1234567890abcdef1234567890"
        text = "\n".join(
            [
                f"built image {digest}",
                f"pushed image {digest}",
                f"tagged image {digest}",
                f"verified image {digest}",
            ]
        )
        collapsed, markers = collapse_aliases(text)
        self.assertTrue(markers)
        self.assertEqual(expand_aliases(collapsed), text)

    def test_aliasing_never_increases_size(self):
        text = "short short short"
        result = compress_trace(text)
        self.assertLessEqual(result.compressed_tokens, result.original_tokens)


if __name__ == "__main__":
    unittest.main()
