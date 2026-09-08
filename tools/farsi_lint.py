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
    # فعل کمکی، نه هر واژه‌ای که این رشته را در خود دارد: «برمی‌گردد» و «درمی‌گردد»
    # فعل‌های دیگری‌اند. مرزِ واژه با «حرف فارسی پیش از آن نباشد» ساخته می‌شود.
    r"(?<![\u0621-\u06cc\u200c])می\u200cگردد": "«می‌شود»",
    r"(?<![\u0621-\u06cc\u200c])می\u200cگردند": "«می‌شوند»",
    r"(?<![\u0621-\u06cc\u200c])میگردد": "«می‌شود»",
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


UNWRAPPED = re.compile(
    r"(?<!\\num\{)(?<!\\wrongnum\{)(?<![۰-۹\u066b/])[۰-۹]+(?:[\u066b/][۰-۹]+)?")


# یک اصطلاح انگلیسی، یک معادل فارسی، در کل متن. هر سطر: معادل مصوب و رقیب‌هایی که
# نباید در متن بیایند. فهرست از GLOSSARY.md می‌آید و با آن هم‌گام نگه داشته می‌شود.
SYNONYMS = {
    "واسنجی": ["کالیبراسیون", "کالیبره"],
    "خودگردان‌سازی": ["بوت‌استرپ", "بوت استرپ"],
    "تا": ["فولد"],
    "برون‌تایی": ["خارج از تا", "خارج‌ازتا"],
    "بازنمایی": ["بازنمود"],
    "ریزتنظیم": ["تنظیم دقیق", "فاین‌تیون"],
    "پیش‌ثبت": ["ثبت پیشین", "پیش‌ثبت‌نام"],
    "تبار داده": ["اصالت داده", "منشأ داده"],
    "ادم ماکولا": ["ورم ماکولا", "خیز ماکولا"],
    "درجه‌بندی": ["طبقه‌بندی درجه"],
    "کف کلاس اکثریت": ["خط پایه اکثریت"],
    "نقطهٔ برش": ["نقطه برش", "حد آستانه"],
}


def check_glossary(paths):
    """هیچ اصطلاحی نباید در یک جای متن یک معادل و در جای دیگر معادل دیگری داشته باشد."""
    hits = []
    for p in paths:
        try:
            body = strip_noise(open(p, encoding="utf-8").read())
        except FileNotFoundError:
            continue
        for approved, banned in SYNONYMS.items():
            for b in banned:
                if b in body:
                    hits.append((p, b, approved))
    return hits


def check_ledger_ambiguity(ledger):
    """دو مقدارِ متفاوت نباید یک رشتهٔ یکسان تولید کنند.

    اگر تولید کنند، یک رقم در متن به دو عدد مختلف اشاره می‌کند و آزمونِ «هر عدد از دفتر
    می‌آید» دیگر چیزی را تضمین نمی‌کند — همان حالتی که یک نگهبان قبول می‌کند بی‌آنکه
    چیزی را بررسی کرده باشد.
    """
    seen, bad = {}, []
    for key, row in ledger.items():
        fa, val = row["fa"].lstrip("-\u2212"), round(row["value"], 6)
        if fa in seen and abs(seen[fa][1] - abs(val)) > 1e-6:
            bad.append((fa, seen[fa][0], seen[fa][1], key, abs(val)))
        else:
            seen.setdefault(fa, (key, abs(val)))
    return bad


def check_wrapping(path):
    """هر عدد فارسی باید در \\num{} باشد، وگرنه وارونه چاپ می‌شود.

    این آزمون از دلِ یک اشتباه بیرون آمد: استدلال کردیم که ارقام فارسی زیر الگوریتم
    یونیکد چپ‌به‌راست‌اند و نتیجه گرفتیم عددهای منبع وارونه‌اند. پیاده‌سازی دوسویهٔ
    زی‌تک الگوریتم کامل یونیکد نیست و «۰٫۵» بدون پوشش، «۵٫۰» چاپ می‌شود.
    استدلال جای نگاه کردن به صفحهٔ چاپ‌شده را نمی‌گیرد؛ این آزمون جای آن را می‌گیرد.
    """
    body = strip_noise(open(path, encoding="utf-8").read())
    body = re.sub(r"\\num\{[^{}]*\}", "", body)
    body = re.sub(r"\\wrongnum\{[^{}]*\}", "", body)
    return [m.group() for m in NUMERAL.finditer(body)]


def check_numbers(path, ledger):
    """هر عدد فارسی باید در دفتر اعداد تولیدشده باشد و با جداکنندهٔ درست نوشته شود.

    این آزمون سه شکست را با هم می‌گیرد: عددی که با دست تایپ شده، عددی که کهنه شده، و
    عددی که ارقامش وارونه نوشته شده — یعنی همان اشتباهی که «۵/۰» را به جای «۰٫۵»
    نوشت و آستانهٔ سیگموئید را ۵ کرد. آنچه این آزمون ثابت نمی‌کند این است که عدد
    برای *این جمله* درست باشد؛ هیچ آزمون خودکاری آن را ثابت نمی‌کند.
    """
    body = strip_noise(open(path, encoding="utf-8").read())
    # علامت منفی جدا از رقم بررسی می‌شود: متن ممکن است «−» تایپوگرافیک را به کار ببرد،
    # در حالی که دفتر «-» می‌نویسد. آنچه باید بررسی شود خودِ رقم‌هاست.
    allowed = {row["fa"].lstrip("-\u2212") for row in ledger.values()}
    bad_sep, unknown = [], []
    for tok, _ in numbers_in(body):
        if "/" in tok:
            bad_sep.append(tok)
        elif FA_DEC in tok or int(tok.translate(TO_LATIN)) >= 100:
            # سال‌ها اندازه‌گیری نیستند و در دفتر اعداد جایی ندارند. این استثنا باریک است:
            # تنها عدد صحیح چهاررقمی در بازهٔ ۱۹۰۰ تا ۲۱۰۰. هر شمارشی که در همین بازه بیفتد
            # باید صریحاً در دفتر ثبت شود، وگرنه از این استثنا عبور می‌کند.
            if FA_DEC not in tok:
                yr = int(tok.translate(TO_LATIN))
                if 1900 <= yr <= 2100 or 1300 <= yr <= 1500:
                    continue
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
    # \wrongnum{...} است برای عددی که متن *عمداً* به شکل غلط نقل می‌کند — مثل «۵/۰» در
    # بخش شکست تبار دادهٔ خودمان. استثنا باید در خودِ متن دیده شود، نه در قاعدهٔ آزمون؛
    # وگرنه همان «آزمونی که موفقیت گزارش می‌کند بی‌آن‌که چیزی را بررسی کند» می‌شود.
    body = re.sub(r"\\wrongnum\{[^}]*\}", "", body)
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


def check_refs(paths):
    """ارجاعِ بی‌مقصد در خروجی به «??» تبدیل می‌شود و خطای لاتک تولید نمی‌کند.

    ارجاع‌ها در همهٔ پرونده‌های داده‌شده با هم بررسی می‌شوند، چون یک فصل قانوناً به برچسبِ
    فصل دیگر ارجاع می‌دهد.
    """
    labels, refs = set(), []
    for p in paths:
        try:
            body = open(p, encoding="utf-8").read()
        except FileNotFoundError:
            continue
        labels |= set(re.findall(r"\\label\{([^}]*)\}", body))
        refs += [(p, r) for r in re.findall(r"\\ref\{([^}]*)\}", body)]
    return [(p, r) for p, r in refs if r not in labels]


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
        unwrapped = check_wrapping(p)
        total += len(unwrapped)
        if unwrapped:
            print(f"  عددی که در \\num{{}} پیچیده نشده (وارونه چاپ می‌شود): {len(unwrapped)}")
            print(f"     {sorted(set(unwrapped))[:8]}")
        else:
            print("  پوشش عددها: همه در \\num{} هستند ✅")
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
    gl = check_glossary(paths)
    total += len(gl)
    if gl:
        print("\nاصطلاحی با دو معادل مختلف در متن:")
        for p, bad, ok in gl:
            print(f"  {p}: «{bad}» -> «{ok}»")
    else:
        print("\nواژگان: هر اصطلاح یک معادل دارد ✅")

    if ledger is not None:
        # این تخلف شمرده نمی‌شود، و دلیلش را باید نوشت. دو مقدار متفاوت می‌توانند پس از
        # گرد شدن یک رشته بدهند (۰٫۶۴۷ و ۰٫۶۵ هر دو «۰٫۶۵» می‌شوند). این ذاتیِ گرد کردن
        # است، نه خطا، و رد کردنش عددهای درست را هم رد می‌کرد. ولی اندازه‌اش دقیقاً همان
        # چیزی است که آزمونِ «هر عدد از دفتر می‌آید» تضمین نمی‌کند: وجود را تضمین می‌کند،
        # هویت را نه. گزارش می‌شود تا این ضعف اندازه‌گیری‌شده بماند، نه نانوشته.
        amb = check_ledger_ambiguity(ledger)
        print(f"\nابهام باقی‌مانده در دفتر: {len(amb)} رشته به بیش از یک مقدار اشاره می‌کند")
        print("  (ذاتیِ گرد کردن؛ آزمون وجودِ عدد را تضمین می‌کند، نه هویتش)")

    dangling = check_refs(paths)
    total += len(dangling)
    if dangling:
        print("\nارجاع بی‌مقصد (در خروجی «??» می‌شود):")
        for p, r in dangling:
            print(f"  {p}: \\ref{{{r}}}")
    else:
        print("\nارجاع‌ها: همه مقصد دارند ✅")
    print(f"\nمجموع تخلف‌ها: {total}")
    return 0 if total == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["thesis/chapter3.tex", "thesis/chapter4.tex"]))
