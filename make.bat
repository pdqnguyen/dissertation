set main=main
pdflatex %main%.tex
biblatex %main%
pdflatex %main%.tex
pdflatex %main%.tex
pdflatex %main%.tex
DEL *.aux *.log *.out *.lof *.lot *.toc %main%.bbl %main%.blg %main%.syntex.gz %main%.tdo