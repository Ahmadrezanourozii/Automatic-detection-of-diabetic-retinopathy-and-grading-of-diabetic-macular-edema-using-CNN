"""
check_thesis_numbers.py — refuse to let a hand-typed number back into the thesis.

The original failure of this project was not a modelling mistake. It was that chapter 4
reported 91.6 % and 87.6 % with no run behind either figure, and nothing in the workflow
could tell. The generated tables in docs/generated/ fix the *supply* of numbers; this fixes
the *demand*, by failing when the chapter states a result that no archived run produced.

What it does
  1. Collects every result-shaped number from docs/generated/summary.json — accuracies,
     QWKs, per-class recalls, floors, sensitivities — as the set of legitimate values.
  2. Scans the thesis chapter for numeric literals that look like reported results
     (percentages, and decimals in [0,1] with three places), skipping anything inside a
     \\input-ed generated file, a comment, or a citation/label.
  3. Reports every literal that does not match a legitimate value within tolerance.

It also handles Persian-Indic digits (۰۱۲۳۴۵۶۷۸۹), because the thesis is typeset in Persian
and the original 91.6 % appears there as ۹۱٫۶.

Usage:
    python src/check_thesis_numbers.py --chapter ../thesis-chegeni/tex/chapter4.tex
Exit code is non-zero if any unexplained number is found.
"""
from __future__ import annotations
import argparse, json, os, re, sys

FA_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
FA_MAP = {ord(d): str(i) for i, d in enumerate(FA_DIGITS)}
FA_MAP[ord("٫")] = "."          # Arabic decimal separator
FA_MAP[ord("٪")] = "%"


def normalise(text):
    return text.translate(FA_MAP)


def legit_values(summary_path):
    """Every number an archived run actually produced, as percentages and as raw values."""
    if not os.path.exists(summary_path):
        return set(), None
    s = json.load(open(summary_path))
    vals = set()

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif isinstance(o, (int, float)) and o is not None:
            f = float(o)
            vals.add(round(f, 4))
            if 0.0 <= f <= 1.0:
                vals.add(round(f * 100, 1))      # the same quantity as a percentage
    walk(s.get("metrics", s))
    return vals, s.get("best_run")


# A percentage in this thesis is written three ways: 91.6\%, 91.6%, and -- the one that
# matters, because it is how the original unverifiable claims appear -- "۹۱٫۶ درصد", with
# the unit as a Persian word. Missing that form made the first version of this checker pass
# on exactly the numbers it was written to catch (ISSUES.md §14).
NUM_RE = re.compile(
    r"(?<![\w.])(\d{1,3}(?:\.\d{1,3})?)\s*(\\%|%|\u062f\u0631\u0635\u062f)?")
SKIP_LINE = re.compile(r"^\s*%|\\label|\\ref|\\cite|\\includegraphics|\\input|"
                       r"\\section|\\subsection|\\chapter|\\begin|\\end|\\usepackage")


def scan(chapter_path, vals, tol=0.15):
    raw = open(chapter_path, encoding="utf-8", errors="replace").read()
    text = normalise(raw)
    problems = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if SKIP_LINE.search(line):
            continue
        stripped = re.sub(r"%.*$", "", line)          # trailing LaTeX comment
        for m in NUM_RE.finditer(stripped):
            token, pct = m.group(1), m.group(2)
            try:
                v = float(token)
            except ValueError:
                continue
            # a result claim is: any number carrying a percent unit, or a bare decimal in
            # [0,1] with 2+ places (a QWK, an F1, an AUC)
            looks_like_result = bool(pct) or (
                "." in token and 0.0 <= v <= 1.0 and len(token.split(".")[1]) >= 2)
            if not looks_like_result:
                continue
            if v in (0.0, 100.0, 1.0):
                continue                              # trivially not a claim
            if any(abs(v - g) <= tol for g in vals):
                continue
            problems.append((lineno, m.group(0).strip(), line.strip()[:90]))
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chapter", required=True)
    ap.add_argument("--summary", default="docs/generated/summary.json")
    ap.add_argument("--tol", type=float, default=0.15)
    a = ap.parse_args()

    vals, best = legit_values(a.summary)
    if not vals:
        print(f"no generated numbers in {a.summary} — run src/report.py first")
        sys.exit(1)
    print(f"{len(vals)} legitimate values from run {best} ({a.summary})")

    if not os.path.exists(a.chapter):
        print(f"chapter not found: {a.chapter}")
        sys.exit(1)

    problems = scan(a.chapter, vals, a.tol)
    for lineno, tok, ctx in problems:
        print(f"  UNEXPLAINED  {os.path.basename(a.chapter)}:{lineno}  {tok}\n"
              f"               {ctx}")
    print(f"\n{len(problems)} numeric literal(s) in the chapter match no archived result.")
    if problems:
        print("Each one must either come from a \\input-ed generated table, or be deleted. "
              "A number in the thesis that no run produced is the failure this project "
              "started from (ISSUES.md §1).")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
