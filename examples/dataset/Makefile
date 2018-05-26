.PHONY: html
html:
	pweave -f pandoc2html dataset.pmd

.PHONY: rst
rst:
	pweave -f markdown dataset.pmd
	pandoc -s -t rst dataset.md -o dataset.rst
