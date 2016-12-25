.PHONY: examples
examples:
	jupyter nbconvert --to rst examples/quickstart/quickstart.ipynb
	mv examples/quickstart/quickstart.rst doc/
	jupyter nbconvert --to rst examples/movielens/example.ipynb
	mv examples/movielens/example.rst doc/examples/movielens_implicit.rst
	jupyter nbconvert --to rst examples/movielens/learning_schedules.ipynb
	mv examples/movielens/learning_schedules.rst doc/examples/
	cp -r examples/movielens/learning_schedules_files doc/examples/
	rm -rf examples/movielens/learning_schedules_files
	jupyter nbconvert --to rst examples/stackexchange/hybrid_crossvalidated.ipynb
	mv examples/stackexchange/hybrid_crossvalidated.rst doc/examples/
	jupyter nbconvert --to rst examples/movielens/warp_loss.ipynb
	mv examples/movielens/warp_loss.rst doc/examples/
	cp -r examples/movielens/warp_loss_files doc/examples/
	rm -rf examples/movielens/warp_loss_files
.PHONY: update-docs
update-docs:
	pip install -e . \
	&& cd doc && make html && cd .. \
	&& git fetch origin gh-pages && git checkout gh-pages \
	&& rm -rf ./docs/ \
	&& mkdir ./docs/ \
	&& cp -r ./doc/_build/html/* ./docs/ \
	&& git add -A ./docs/* \
	&& git commit -m 'Update docs.' && git push origin gh-pages
