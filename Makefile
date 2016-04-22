.PHONY: quickstart
quickstart:
	ipython nbconvert --to rst examples/quickstart/quickstart.ipynb
	mv quickstart.rst doc/
