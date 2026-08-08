"""Deterministic compression for repetitive execution traces.

Public portfolio snapshot extracted from a larger private workflow. The goal is
simple: reduce repetitive build/test/tool output without silently rewriting the
information an engineer may need later.

No model inference is used. Transformations are deterministic, visible, and
accepted only when they reduce the estimated token cost.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass

_PROGRESS_RE = re.compile(r"^\d{1,3}%$")
_LONG_TOKEN_RE = re.compile(r"\S{16,}")


@dataclass(frozen=True)
class CompressionResult:
    compressed: str
    original_tokens: int
    compressed_tokens: int
    markers: tuple[str, ...]

    @property
    def reduction(self) -> float:
        if not self.original_tokens:
            return 0.0
        return 1 - (self.compressed_tokens / self.original_tokens)

    def as_dict(self) -> dict[str, object]:
        return {
            "compressed": self.compressed,
            "original_tokens": self.original_tokens,
            "compressed_tokens": self.compressed_tokens,
            "compression_ratio": round(self.reduction, 6),
            "markers": list(self.markers),
        }


def estimate_tokens(text: str) -> int:
    """Cheap tokenizer-independent estimate used only for the benefit gate."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def _varying_positions(rows: list[list[str]]) -> list[int]:
    width = len(rows[0])
    return [i for i in range(width) if len({row[i] for row in rows}) > 1]


def _progress_position(rows: list[list[str]], varying: list[int]) -> int | None:
    for pos in varying:
        values = [row[pos] for row in rows]
        if not all(_PROGRESS_RE.fullmatch(value) for value in values):
            continue
        numbers = [int(value[:-1]) for value in values]
        if numbers == sorted(numbers):
            return pos
    return None


def _candidate_run(lines: list[str], start: int) -> tuple[int, list[list[str]]] | None:
    """Find a contiguous same-shape run starting at *start*.

    The matcher deliberately refuses variable-width lines. Missing a possible
    compression opportunity is safer than guessing that two unlike structures
    represent the same template.
    """
    first = lines[start].split()
    if len(first) < 2:
        return None

    rows = [first]
    end = start + 1
    while end < len(lines):
        row = lines[end].split()
        if len(row) != len(first):
            break
        rows.append(row)
        end += 1

    if len(rows) < 3:
        return None
    return end, rows


def collapse_templates(text: str) -> tuple[str, list[str]]:
    if not text:
        return text, []

    lines = text.splitlines()
    output: list[str] = []
    markers: list[str] = []
    cursor = 0

    while cursor < len(lines):
        candidate = _candidate_run(lines, cursor)
        if candidate is None:
            output.append(lines[cursor])
            cursor += 1
            continue

        end, rows = candidate
        varying = _varying_positions(rows)
        constant_count = len(rows[0]) - len(varying)

        # Keep the rule intentionally conservative: at most two fields may
        # vary, and there must be at least two stable fields anchoring the run.
        if not varying or len(varying) > 2 or constant_count < 2:
            output.append(lines[cursor])
            cursor += 1
            continue

        progress_pos = _progress_position(rows, varying)
        original = "\n".join(lines[cursor:end])

        if progress_pos is not None:
            percentages = [row[progress_pos] for row in rows]
            middle_count = max(0, len(rows) - 2)
            rendered_lines = [
                lines[cursor],
                f"[... {middle_count} intermediate progress steps "
                f"({percentages[0]}->{percentages[-1]}) collapsed ...]",
                lines[end - 1],
            ]
            marker = f"template:progress:{len(rows)}"
        else:
            values = [" ".join(row[pos] for pos in varying) for row in rows[1:-1]]
            rendered_lines = [
                lines[cursor],
                f"[... {len(rows) - 2} more like this, varying: {', '.join(values)} ...]",
                lines[end - 1],
            ]
            marker = f"template:lossless:{len(rows)}"

        rendered = "\n".join(rendered_lines)
        if estimate_tokens(rendered) >= estimate_tokens(original):
            output.append(lines[cursor])
            cursor += 1
            continue

        output.extend(rendered_lines)
        markers.append(marker)
        cursor = end

    # splitlines() intentionally removes the final newline; restore it when
    # present so non-transformed inputs remain byte-for-byte stable.
    result = "\n".join(output)
    if text.endswith("\n"):
        result += "\n"
    return result, markers


def _long_token_counts(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for match in _LONG_TOKEN_RE.finditer(text):
        token = match.group(0)
        counts[token] = counts.get(token, 0) + 1
    return counts


def collapse_aliases(text: str) -> tuple[str, list[str]]:
    """Replace profitable repeated long tokens with reversible aliases."""
    counts = _long_token_counts(text)
    candidates = [token for token, count in counts.items() if count >= 3]
    if not candidates:
        return text, []

    body = text
    legend: list[str] = []

    for token in sorted(candidates, key=len, reverse=True):
        alias = f"<<A{len(legend) + 1}>>"
        trial_body = body.replace(token, alias)
        trial_entry = f"{alias} = {token}"

        before = estimate_tokens(body)
        after = estimate_tokens(trial_body) + estimate_tokens(trial_entry)
        if after >= before:
            continue

        body = trial_body
        legend.append(trial_entry)

    if not legend:
        return text, []

    header = "[alias legend — exact substitutions]"
    return header + "\n" + "\n".join(legend) + "\n\n" + body, [f"alias:{len(legend)}"]


def expand_aliases(text: str) -> str:
    header = "[alias legend — exact substitutions]"
    if not text.startswith(header):
        return text

    legend_block, body = text.split("\n\n", 1)
    for line in legend_block.splitlines()[1:]:
        alias, separator, value = line.partition(" = ")
        if separator:
            body = body.replace(alias, value)
    return body


def compress_trace(text: str) -> CompressionResult:
    original_tokens = estimate_tokens(text)
    template_text, template_markers = collapse_templates(text)
    final_text, alias_markers = collapse_aliases(template_text)

    return CompressionResult(
        compressed=final_text,
        original_tokens=original_tokens,
        compressed_tokens=estimate_tokens(final_text),
        markers=tuple(template_markers + alias_markers),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic execution-trace compression")
    parser.add_argument("--file", help="Read input from a file instead of stdin")
    args = parser.parse_args()

    if args.file:
        with open(args.file, "r", encoding="utf-8", errors="replace") as handle:
            text = handle.read()
    else:
        text = sys.stdin.read()

    json.dump(compress_trace(text).as_dict(), sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
