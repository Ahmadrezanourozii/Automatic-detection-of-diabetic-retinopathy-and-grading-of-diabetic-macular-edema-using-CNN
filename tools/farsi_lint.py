"""
farsi_lint.py — شمارش کلمات و ساختارهای ممنوع در متن فارسی پایان‌نامه.

قاعده‌ای که در پرامپت نهایی توافق شد: تا زمانی که شمارش صفر نشده، فصل تمام‌شده اعلام نمی‌شود.
این اسکریپت آن قاعده را از «نیت» به «آزمون» تبدیل می‌کند.

Usage:  python3 tools/farsi_lint.py thesis/chapter3.tex thesis/chapter4.tex
"""
from __future__ import annotations
import json
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

LEDGER = "docs/generated/thesis_numbers.json"

# جداکنندهٔ اعشار در متن فارسی این پایان‌نامه U+066B است، همان چیزی که src/report.py
# در جدول‌های تولیدشده می‌نویسد. «/» جداکنندهٔ کسر و تاریخ است و در متن پذیرفته نیست.
FA_DEC = "\u066b"
NUMERAL = re.compile(r"[۰-۹]+(?:[" + FA_DEC + r"/][۰-۹]+)?")


def numbers_in(body):
    """هر عدد فارسیِ نثر، به همراه جای آن. اعداد لاتین بررسی نمی‌شوند."""
    return [(m.group(), m.start()) for m in NUMERAL.finditer(body)]


def check_numbers(path, ledger):
    """هر عدد فارسی باید در دفتر اعداد تولیدشده باشد و با جداکنندهٔ درست نوشته شود.

    این آزمون سه شکست را با هم می‌گیرد: عددی که با دست تایپ شده، عددی که کهنه شده، و
    عددی که ارقامش وارونه نوشته شده — یعنی همان اشتباهی که «۵/۰» را به جای «۰٫۵»
    نوشت و آستانهٔ سیگموئید را ۵ کرد. آنچه این آزمون ثابت نمی‌کند این است که عدد
    برای *این جمله* درست باشد؛ هیچ آزمون خودکاری آن را ثابت نمی‌کند.
    """
    body = strip_noise(open(path, encoding="utf-8").read())
    allowed = {row["fa"] for row in ledger.values()}
    bad_sep, unknown = [], []
    for tok, _ in numbers_in(body):
        if "/" in tok:
            bad_sep.append(tok)
        elif FA_DEC in tok or int(tok.translate(TO_LATIN)) >= 100:
            if tok not in allowed:
                unknown.append(tok)
    return bad_sep, unknown


TO_LATIN = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")


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
    ledger = None
    if "--numbers" in paths:
        paths = [p for p in paths if p != "--numbers"]
        try:
            ledger = json.load(open(LEDGER, encoding="utf-8"))
        except FileNotFoundError:
            print(f"دفتر اعداد پیدا نشد: {LEDGER}\n"
                  f"نخست اجرا کنید: python3 src/thesis_numbers.py --datasets <root>")
            return 1
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
        if ledger is not None:
            bad_sep, unknown = check_numbers(p, ledger)
            total += len(bad_sep) + len(unknown)
            if bad_sep:
                print(f"  جداکنندهٔ اعشار نادرست («/» به جای «٫»): {len(bad_sep)}")
                print(f"     {sorted(set(bad_sep))[:8]}")
            if unknown:
                print(f"  عددی که در دفتر اعداد نیست: {len(unknown)}")
                print(f"     {sorted(set(unknown))[:8]}")
            if not bad_sep and not unknown:
                print("  اعداد: همه در دفتر اعداد تولیدشده هستند ✅")
    print(f"\nمجموع تخلف‌ها: {total}")
    return 0 if total == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["thesis/chapter3.tex", "thesis/chapter4.tex"]))
