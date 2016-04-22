
Quickstart
==========

In this example, we'll build an implicit feedback recommender using the
Movielens 100k dataset (http://grouplens.org/datasets/movielens/100k/).

The code behind this example is available as a `Jupyter
notebook <https://github.com/lyst/lightfm/tree/master/examples/quickstart/quickstart.ipynb>`__

LightFM includes functions for getting and processing this dataset, so
obtaining it is quite easy.

.. code:: python

    import numpy as np
    
    from lightfm.datasets import fetch_movielens
    
    data = fetch_movielens(min_rating=5.0)

This downloads the dataset and automatically pre-processes it into
sparse matrices suitable for further calculation. In particular, it
prepares the sparse user-item matrices, containing positive entries
where a user interacted with a product, and zeros otherwise.

We have two such matrices, a training and a testing set. Both have
around 1000 users and 1700 items. We'll train the model on the train
matrix but test it on the test matrix.

.. code:: python

    print(repr(data['train']))
    print(repr(data['test']))


.. parsed-literal::

    <943x1682 sparse matrix of type '<type 'numpy.int32'>'
    	with 19048 stored elements in COOrdinate format>
    <943x1682 sparse matrix of type '<type 'numpy.int32'>'
    	with 2153 stored elements in COOrdinate format>


We need to import the model class to fit the model:

.. code:: python

    from lightfm import LightFM

We're going to use the WARP (Weighted Approximate-Rank Pairwise) model.
WARP is an implicit feedback model: all interactions in the training
matrix are treated as positive signals, and products that users did not
interact with they implicitly do not like. The goal of the model is to
score these implicit positives highly while assigining low scores to
implicit negatives.

Model training is accomplished via SGD (stochastic gradient descent).
This means that for every pass through the data --- an epoch --- the
model learns to fit the data more and more closely. We'll run it for 10
epochs in this example. We can also run it on multiple cores, so we'll
set that to 2. (The dataset in this example is too small for that to
make a difference, but it will matter on bigger datasets.)

.. code:: python

    model = LightFM(loss='warp')
    %time model.fit(data['train'], epochs=30, num_threads=2)


.. parsed-literal::

    CPU times: user 1.55 s, sys: 4 ms, total: 1.56 s
    Wall time: 838 ms




.. parsed-literal::

    <lightfm.lightfm.LightFM at 0x7f978c58ea50>



Done! We should now evaluate the model to see how well it's doing. We're
most interested in how good the ranking produced by the model is.
Precision@k is one suitable metric, expressing the percentage of top k
items in the ranking the user has actually interacted with. ``lightfm``
implements a number of metrics in the ``evaluation`` module.

.. code:: python

    from lightfm.evaluation import precision_at_k

We'll measure precision in both the train and the test set.

.. code:: python

    print("Train precision: %.2f" % precision_at_k(model, data['train'], k=5).mean())
    print("Test precision: %.2f" % precision_at_k(model, data['test'], k=5).mean())


.. parsed-literal::

    Train precision: 0.43
    Test precision: 0.04


Unsurprisingly, the model fits the train set better than the test set.

For an alternative way of judging the model, we can sample a couple of
users and get their recommendations. To make predictions for given user,
we pass the id of that user and the ids of all products we want
predictions for into the ``predict`` method.

.. code:: python

    def sample_recommendation(model, data, user_ids):
        
    
        n_users, n_items = data['train'].shape
    
        for user_id in user_ids:
            known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]
            
            scores = model.predict(user_id, np.arange(n_items))
            top_items = data['item_labels'][np.argsort(-scores)]
            
            print("User %s" % user_id)
            print("     Known positives:")
            
            for x in known_positives[:3]:
                print("        %s" % x)
    
            print("     Recommended:")
            
            for x in top_items[:3]:
                print("        %s" % x)
            
    sample_recommendation(model, data, [3, 25, 450]) 


.. parsed-literal::

    User 3
         Known positives:
            Contact (1997)
            Air Force One (1997)
            In & Out (1997)
         Recommended:
            Air Force One (1997)
            Assignment, The (1997)
            Kiss the Girls (1997)
    User 25
         Known positives:
            Fargo (1996)
            Godfather, The (1972)
            L.A. Confidential (1997)
         Recommended:
            L.A. Confidential (1997)
            Titanic (1997)
            Fargo (1996)
    User 450
         Known positives:
            Event Horizon (1997)
            Scream (1996)
            Conspiracy Theory (1997)
         Recommended:
            Independence Day (ID4) (1996)
            Scream (1996)
            Ransom (1996)


