# ساخت پایان‌نامه. سه بار xelatex لازم است تا ارجاع‌ها و فهرست‌ها بنشینند.
SHELL := /bin/bash
export PATH := /opt/homebrew/bin:$(PATH)

main.pdf: main.tex frontmatter.tex abstract_en.tex chapter1.tex chapter2.tex chapter3.tex \
          chapter4.tex chapter5.tex parta_report.tex refs.bib
	xelatex -interaction=nonstopmode -halt-on-error main.tex
	-bibtex main
	xelatex -interaction=nonstopmode -halt-on-error main.tex
	xelatex -interaction=nonstopmode -halt-on-error main.tex

# دروازه: هیچ فصلی تا وقتی شمارش تخلف صفر نشده تمام‌شده نیست.
lint:
	python3 ../tools/farsi_lint.py --numbers *.tex

clean:
	rm -f main.aux main.log main.out main.toc main.lot main.bbl main.blg

.PHONY: lint clean
