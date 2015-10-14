# LightFM

![LightFM logo](lightfm.png)

A Python implementation of LightFM, a hybrid recommendation algorithm.

The LightFM model incorporates both item and user metadata into the traditional matrix factorization algorithm. It represents each user and item as the sum of the latent representations of their features, thus allowing recommendations to generalise to new items (via item features) and to new users (via user features).

The details of the approach are described in the LightFM paper, available on [arXiv](http://arxiv.org/abs/1507.08439).

The model can be trained using four methods:

- logistic loss: useful when both positive (1) and negative (-1) interactions
                 are present.
- BPR: Bayesian Personalised Ranking [1] pairwise loss. Maximises the
       prediction difference between a positive example and a randomly
       chosen negative example. Useful when only positive interactions
       are present and optimising ROC AUC is desired.
- WARP: Weighted Approximate-Rank Pairwise [2] loss. Maximises
        the rank of positive examples by repeatedly sampling negative
        examples until a rank violating one is found. Useful when only
        positive interactions are present and optimising the top of
        the recommendation list (precision@k) is desired.
- k-OS WARP: k-th order statistic loss [3]. A modification of WARP that uses the k-th
             positive example for any given user as a basis for pairwise updates.

Two learning rate schedules are implemented:
- adagrad: [4]
- adadelta: [5]

## Installation
Install from pypi using pip: `pip install lightfm`.

Note for OSX users: due to its use of OpenMP, `lightfm` does not compile under Clang. To install it, you will need a reasonably recent version of `gcc` (from Homebrew for instance). This should be picked up by `setup.py`; if it is not, please open an issue.

Building with the default Python distribution included in OSX is also not supported; please try the version from Homebrew or Anaconda.

## Usage
Model fitting is very straightforward.

Create a model instance with the desired latent dimensionality
```python
from lightfm import LightFM

model = LightFM(no_components=30)
```

Assuming `train` is a (no_users, no_items) sparse matrix (with 1s denoting positive, and -1s negative interactions), you can fit a traditional matrix factorization model by calling
```python
model.fit(train, epochs=20)
```
This will train a traditional MF model, as no user or item features have been supplied.

To get predictions, call `model.predict`:
```python
predictions = model.predict(test_user_ids, test_item_ids)
```

User and item features can be incorporated into training by passing them into the `fit` method. Assuming `user_features` is a (no_users, no_user_features) sparse matrix (and similarly for `item_features`), you can call
```python
model.fit(train,
          user_features=user_features,
          item_features=item_features,
          epochs=20)
predictions = model.predict(test_user_ids,
                            test_item_ids,
                            user_features=user_features,
                            item_features=item_features)
```
to train the model and obtain predictions.

Both training and prediction can employ multiple cores for speed:
```python
model.fit(train, epochs=20, num_threads=4)
predictions = model.predict(test_user_ids, test_item_ids, num_threads=4)
```

This implementation uses asynchronous stochastic gradient descent [6] for training. This can lead to lower accuracy when the interaction matrix (or the feature matrices) are very dense and a large number of threads is used. In practice, however, training on a sparse dataset with 20 threads does not lead to a measurable loss of accuracy.

In an implicit feedback setting, the BPR, WARP, or k-OS WARP loss functions can be used. If `train` is a sparse matrix with positive entries representing positive interactions, the model can be trained as follows:
```python
model = LightFM(no_components=30, loss='warp')
model.fit(train, epochs=20)
```

## Examples

Check the `examples` directory for more examples.

The [Movielens example](/examples/movielens/example.ipynb) shows how to use `lightfm` on the Movielens dataset, both with and without using movie metadata. [Another example](/examples/movielens/learning_schedules.ipynb) compares the performance of the adagrad and adadelta learning schedules.

The [Cross Validated example](/examples/crossvalidated/example.ipynb) shows how to use `lightfm` on a dataset from [stats.stackexchange.com](http://stats.stackexchange.com) with both item and user features.

The [Kaggle coupon purchase prediction](https://github.com/tdeboissiere/Kaggle/blob/master/Ponpare/ponpare_lightfm.ipynb) example applies LightFM to predicting coupon purchases.

## Development
Pull requests are welcome. To install for development:

1. Clone the repository: `git clone git@github.com:lyst/lightfm.git`
2. Install it for development using pip: `cd lightfm && pip install -e .`
3. You can run tests by running `python setupy.py test`.

When making changes to the `.pyx` extension files, you'll need to run `python setup.py cythonize` in order to produce the extension `.c` files before running `pip install -e .`.

## References
[1] Rendle, Steffen, et al. "BPR: Bayesian personalized ranking from implicit feedback."
Proceedings of the Twenty-Fifth Conference on Uncertainty in Artificial
Intelligence. AUAI Press, 2009.

[2] Weston, Jason, Samy Bengio, and Nicolas Usunier. "Wsabie: Scaling up to large
vocabulary image annotation." IJCAI. Vol. 11. 2011.

[3] Weston, Jason, Hector Yee, and Ron J. Weiss. "Learning to rank recommendations with
the k-order statistic loss." Proceedings of the 7th ACM conference on Recommender systems. ACM, 2013.

[4] Duchi, John, Elad Hazan, and Yoram Singer. "Adaptive subgradient methods
for online learning and stochastic optimization." The Journal of Machine Learning Research 12 (2011): 2121-2159.

[5] Zeiler, Matthew D. "ADADELTA: An adaptive learning rate method."
arXiv preprint arXiv:1212.5701 (2012).

[6] Recht, Benjamin, et al. "Hogwild: A lock-free approach to parallelizing stochastic gradient descent." Advances in Neural Information Processing Systems. 2011.