.PHONY: examples
examples:
	jupyter nbconvert --to rst examples/quickstart/quickstart.ipynb
	mv examples/quickstart/quickstart.rst doc/
	jupyter nbconvert --to rst examples/movielens/example.ipynb
	mv examples/movielens/example.rst doc/examples/movielens_implicit.rst
