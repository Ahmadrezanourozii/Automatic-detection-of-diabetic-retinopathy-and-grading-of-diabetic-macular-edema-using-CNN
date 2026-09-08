r"""
wrap_numbers.py — wrap every Persian numeral in \num{} so it RENDERS in the right order.

WHY THIS EXISTS. XeTeX's bidi implementation is not a full Unicode Bidirectional Algorithm.
A Persian decimal written in logical order -- ۰٫۵ -- comes out of xepersian as ۵٫۰, because the
decimal separator does not bind the two digit runs into one number and the runs are then laid
out right-to-left like any other RTL content. The same happens with "/" and with a Latin full
stop. Verified by rendering, not assumed.

Two ways to live with that. The first is to type the groups backwards so the renderer's quirk
undoes them; that is what the earliest chapters did, and it works. It is also unreadable,
ungreppable, uncheckable against a ledger, and silently wrong the moment the engine changes.

The second is this: keep the source in logical order, and wrap each numeral in \num{}, which is
\LR{} -- an explicit left-to-right run. Verified to render correctly for decimals, intervals,
percentages and plain integers.

\lr{} does NOT work: it switches to the Latin font, which has no Persian digits, and the number
disappears from the page entirely. That failure is silent too, which is why it is written down
here rather than left to be rediscovered.

Usage:  python3 tools/wrap_numbers.py thesis/*.tex
r"""
from __future__ import annotations
import re
import sys

NUMERAL = re.compile(r"[۰-۹]+(?:[٫/][۰-۹]+)?")
# spans whose contents must not be touched: already wrapped, LaTeX control arguments,
# and the deliberately-wrong numerals of the provenance section
PROTECTED = re.compile(
    r"\\(?:num|wrongnum|lr|LR|label|ref|cite|bibliography|input|includegraphics|"
    r"usepackage|documentclass|settextfont|setlatintextfont|setdigitfont)\s*(?:\[[^\]]*\])?\{[^{}]*\}"
)


def wrap_line(line: str) -> tuple[str, int]:
    if line.lstrip().startswith("%"):
        return line, 0
    body, comment = (line.split("%", 1) + [None])[:2] if "%" in line else (line, None)

    holes, kept = [], []

    def stash(m):
        holes.append(m.group(0))
        return f"\x00{len(holes)-1}\x00"

    body = PROTECTED.sub(stash, body)
    n = 0

    def wrap(m):
        nonlocal n
        n += 1
        return "\\num{" + m.group(0) + "}"

    body = NUMERAL.sub(wrap, body)
    body = re.sub(r"\x00(\d+)\x00", lambda m: holes[int(m.group(1))], body)
    return (body + ("%" + comment if comment is not None else "")), n


def main(paths):
    total = 0
    for p in paths:
        src = open(p, encoding="utf-8").read()
        out, n = [], 0
        for line in src.splitlines(keepends=True):
            w, k = wrap_line(line)
            out.append(w)
            n += k
        open(p, "w", encoding="utf-8").write("".join(out))
        print(f"  {p}: wrapped {n}")
        total += n
    print(f"wrapped {total} numerals in \\num{{}}")


if __name__ == "__main__":
    main(sys.argv[1:])
