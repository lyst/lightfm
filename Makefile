.PHONY: examples
examples:
	jupyter nbconvert --to rst examples/quickstart/quickstart.ipynb
	mv examples/quickstart/quickstart.rst doc/
	jupyter nbconvert --to rst examples/movielens/example.ipynb
	mv examples/movielens/example.rst doc/examples/movielens_implicit.rst
	jupyter nbconvert --to rst examples/movielens/learning_schedules.ipynb
	mv examples/movielens/learning_schedules.rst doc/examples/
	mv examples/movielens/learning_schedules_files doc/examples/
.PHONY: update-docs
update-docs:
	pip install -e .
	cd doc && make html
	git fetch origin gh-pages && git checkout gh-pages && \
	rm -rf ./docs/ \
	mkdir ./docs/ \
	cp -r ./doc/_build/html/* ./docs/ \
	&& git add -A ./docs/* \
	&& git commit -m 'Update docs.' && git push origin gh-pages
