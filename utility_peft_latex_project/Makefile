.PHONY: all clean

all: main.pdf

main.pdf: main.tex references.bib iclr2026_conference.sty
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	bibtex8 main
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	pdflatex -interaction=nonstopmode -halt-on-error main.tex

clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.toc main.fls main.fdb_latexmk
