"""
farsi_lint.py — شمارش کلمات و ساختارهای ممنوع در متن فارسی پایان‌نامه.

قاعده‌ای که در پرامپت نهایی توافق شد: تا زمانی که شمارش صفر نشده، فصل تمام‌شده اعلام نمی‌شود.
این اسکریپت آن قاعده را از «نیت» به «آزمون» تبدیل می‌کند.

Usage:  python3 tools/farsi_lint.py thesis/chapter3.tex thesis/chapter4.tex
"""
from __future__ import annotations
import re
import sys

FORBIDDEN = {
    "می‌باشد": "«است»",
    "میباشد": "«است»",
    "می‌باشند": "«هستند»",
    "می‌گردد": "«می‌شود»",
    "میگردد": "«می‌شود»",
    "می‌گردند": "«می‌شوند»",
    "لازم به ذکر است": "حذف شود",
    "شایان ذکر است": "حذف شود",
    "در دنیای امروز": "حذف شود",
    "با توجه به اهمیت روزافزون": "حذف شود",
    "نقش بسزایی": "حذف شود",
    "به عنوان یک ابزار قدرتمند": "حذف شود",
    "انجام پذیرفت": "«انجام شد»",
    "صورت پذیرفت": "«انجام شد»",
    "قابل توجه": "عدد بگذار",
    "چشمگیر": "عدد بگذار",
    "گاهاً": "غلط — «گاهی»",
    "تلفناً": "غلط",
}

PATTERNS = {
    r"مورد\s+\S+\s+قرار\s+(گرفت|می‌گیرد|گرفته|گرفتند)": "مجهول‌سازی بی‌دلیل — فعل ساده",
    r"[كي]": "«ک» یا «ی» عربی — باید فارسی باشد",
}

EZAFE = {"هٔ": r"هٔ", "ه‌ی": r"ه‌ی"}


def strip_noise(text: str) -> str:
    """کد، دستورهای لاتک و متن لاتین شمرده نمی‌شوند."""
    body = re.sub(r"\\begin\{lstlisting\}.*?\\end\{lstlisting\}", "", text, flags=re.S)
    body = re.sub(r"\\begin\{verbatim\}.*?\\end\{verbatim\}", "", body, flags=re.S)
    body = re.sub(r"\\(lr|url|texttt|cite|ref|label|includegraphics)\{[^}]*\}", "", body)
    body = re.sub(r"%.*", "", body)
    # محیط‌های جدول و شکل نثر نیستند؛ شمردنشان به‌عنوان «جملهٔ بلند» مثبت کاذب می‌سازد
    body = re.sub(r"\\begin\{tabular\}.*?\\end\{tabular\}", "", body, flags=re.S)
    body = re.sub(r"\\begin\{(table|figure)\}.*?\\end\{(table|figure)\}", "", body, flags=re.S)
    body = re.sub(r"\\(section|subsection|subsubsection|caption)\{[^}]*\}", "", body)
    return body


def scan(path):
    body = strip_noise(open(path, encoding="utf-8").read())
    hits = {}
    for w, fix in FORBIDDEN.items():
        n = body.count(w)
        if n:
            hits[w] = (n, fix)
    for pat, fix in PATTERNS.items():
        n = len(re.findall(pat, body))
        if n:
            hits[pat] = (n, fix)
    ez = {k: len(re.findall(v, body)) for k, v in EZAFE.items()}
    sentences = [s.strip() for s in re.split(r"[.؟!]\s", body) if s.strip()]
    longs = [s for s in sentences if len(s) > 260]
    return hits, ez, longs


def main(paths):
    total = 0
    for p in paths:
        try:
            hits, ez, longs = scan(p)
        except FileNotFoundError:
            print(f"\n=== {p} ===\n  (فایل هنوز نوشته نشده)")
            continue
        n = sum(v[0] for v in hits.values())
        mixed = sum(1 for v in ez.values() if v) > 1
        total += n + (1 if mixed else 0)
        print(f"\n=== {p} ===")
        if hits:
            for w, (c, fix) in sorted(hits.items(), key=lambda kv: -kv[1][0]):
                print(f"  {c:4d}  {w[:44]:44s} -> {fix}")
        else:
            print("  کلمهٔ ممنوع: ۰")
        state = "❌ مخلوط شده" if mixed else "✅ یکدست"
        print(f"  کسرهٔ اضافه: {ez}  {state}")
        print(f"  جمله‌های بلندتر از ۲۶۰ نویسه: {len(longs)}")
        for s in longs[:3]:
            print(f"     … {s[:90]}")
    print(f"\nمجموع تخلف‌ها: {total}")
    return 0 if total == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["thesis/chapter3.tex", "thesis/chapter4.tex"]))
