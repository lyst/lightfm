
#Using lightfm on the Movielens dataset

##Getting the data
The first step is to get the movielens data.

Let's import the utility functions from `data.py`:


    import numpy as np
    import data

The following functions get the dataset, and save it to a local file, and parse it into sparse matrices we can pass into `LightFM`.

In particular, `_build_interaction_matrix` constructs the interaction matrix: a (no_users, no_items) matrix with 1 in place of positive interactions, and -1 in place of negative interactions. For this experiment, any rating lower than 4 is a negative rating.


    import inspect


    print(inspect.getsource(data._build_interaction_matrix))

    def _build_interaction_matrix(rows, cols, data):
        """
        Build the training matrix (no_users, no_items),
        with ratings >= 4.0 being marked as positive and
        the rest as negative.
        """
    
        mat = sp.lil_matrix((rows, cols), dtype=np.int32)
    
        for uid, iid, rating, timestamp in data:
            if rating >= 4.0:
                mat[uid, iid] = 1.0
            else:
                mat[uid, iid] = -1.0
    
        return mat.tocoo()
    


Let's run it! The dataset will be automatically downloaded and processed.


    train, test = data.get_movielens_data()

Let's check the matrices.


    train




    <944x1683 sparse matrix of type '<type 'numpy.int32'>'
    	with 90570 stored elements in COOrdinate format>




    test




    <944x1683 sparse matrix of type '<type 'numpy.int32'>'
    	with 9430 stored elements in COOrdinate format>



Looks good and ready to go.

## Fitting the model
Let's import the lightfm model.


    from lightfm import LightFM


    model = LightFM(no_components=30)

In this case, we set the latent dimensionality of the model to 30. Fitting is straightforward.


    model.fit(train, epochs=50)




    <lightfm.lightfm.LightFM at 0x7f035fac4a10>



Let's try to get a handle on the model accuracy using the ROC AUC score.


    from sklearn.metrics import roc_auc_score
    
    train_predictions = model.predict(train.row,
                                      train.col)


    train_predictions




    array([ 1.16739023, -1.81594849,  0.33082035, ..., -0.84471774,
           -3.59646535, -2.3761344 ])




    roc_auc_score(train.data, train_predictions)




    0.98794140543734321



We've got very high accuracy on the train dataset; let's check the test set.


    test_predictions = model.predict(test.row, test.col)


    roc_auc_score(test.data, test_predictions)




    0.72314175941707015



The accuracy is much lower on the test data, suggesting a high degree of overfitting. We can combat this by regularizing the model.


    model = LightFM(no_components=30, user_alpha=0.0001, item_alpha=0.0001)
    model.fit(train, epochs=50)
    roc_auc_score(test.data, model.predict(test.row, test.col))




    0.76055575505353468



A modicum of regularization gives much better results.

## Using metadata
The promise of `lightfm` is the possibility of using metadata in cold-start scenarios. The Movielens dataset has genre data for the movies it contains. Let's use that to train the `LightFM` model.

The `get_movielens_item_metadata` function constructs a (no_items, no_features) matrix containing features for the movies; if we use genres this will be a (no_items, no_genres) feature matrix.


    item_features = data.get_movielens_item_metadata(use_item_ids=False)
    item_features




    <1683x19 sparse matrix of type '<type 'numpy.int32'>'
    	with 2893 stored elements in LInked List format>



We need to pass these to the `fit` method in order to use them.


    model = LightFM(no_components=30, user_alpha=0.0001, item_alpha=0.0001)
    model.fit(train, item_features=item_features, epochs=50)
    roc_auc_score(test.data, model.predict(test.row, test.col, item_features=item_features))




    0.67300181616251231



This is not as accurate as a pure collaborative filtering solution, but should enable us to make recommendations new movies.

If we add item-specific features back, we should get the original accuracy back.


    item_features = data.get_movielens_item_metadata(use_item_ids=True)
    item_features
    model = LightFM(no_components=30, user_alpha=0.0001, item_alpha=0.0001)
    model.fit(train, item_features=item_features, epochs=50)
    roc_auc_score(test.data, model.predict(test.row, test.col, item_features=item_features))




    0.75583737010915852



## Implicit feedback
So far, we have been treating the signals from the data as binary explicit feedback: either a user likes a movie (score >= 4) or does not. However, in many applications feedback is purely implicit: the items a user interacted with are positive signals, but we have no negative signals.

`lightfm` implements two models suitable for dealing with this sort of data:

- BPR: Bayesian Personalised Ranking [1] pairwise loss. Maximises the prediction difference between a positive example and a randomly chosen negative example. Useful when only positive interactions are present and optimising ROC AUC is desired.
- WARP: Weighted Approximate-Rank Pairwise [2] loss. Maximises the rank of positive examples by repeatedly sampling negative examples until rank violating one is found. Useful when only positive interactions are present and optimising the top of the recommendation list (precision@k) is desired.

[1] Rendle, Steffen, et al. "BPR: Bayesian personalized ranking from implicit feedback." Proceedings of the Twenty-Fifth Conference on Uncertainty in Artificial Intelligence. AUAI Press, 2009.

[2] Weston, Jason, Samy Bengio, and Nicolas Usunier. "Wsabie: Scaling up to large vocabulary image annotation." IJCAI. Vol. 11. 2011.

Before using them, let's first load the data and define some evaluation functions.


    train, test = data.get_movielens_data()
    train.data = np.ones_like(train.data)
    test.data = np.ones_like(test.data)


    from sklearn.metrics import roc_auc_score
    
    
    def precision_at_k(model, ground_truth, k):
        """
        Measure precision at k for model and ground truth.
    
        Arguments:
        - lightFM instance model
        - sparse matrix ground_truth (no_users, no_items)
        - int k
    
        Returns:
        - float precision@k
        """
    
        ground_truth = ground_truth.tocsr()
    
        no_users, no_items = ground_truth.shape
    
        pid_array = np.arange(no_items, dtype=np.int32)
    
        precisions = []
    
        for user_id, row in enumerate(ground_truth):
            uid_array = np.empty(no_items, dtype=np.int32)
            uid_array.fill(user_id)
            predictions = model.predict(uid_array, pid_array, num_threads=4)
    
            top_k = set(np.argsort(-predictions)[:k])
            true_pids = set(row.indices[row.data == 1])
    
            if true_pids:
                precisions.append(len(top_k & true_pids) / float(k))
    
        return sum(precisions) / len(precisions)
    
    
    def full_auc(model, ground_truth):
        """
        Measure AUC for model and ground truth on all items.
    
        Arguments:
        - lightFM instance model
        - sparse matrix ground_truth (no_users, no_items)
    
        Returns:
        - float AUC
        """
    
        ground_truth = ground_truth.tocsr()
    
        no_users, no_items = ground_truth.shape
    
        pid_array = np.arange(no_items, dtype=np.int32)
    
        scores = []
    
        for user_id, row in enumerate(ground_truth):
            uid_array = np.empty(no_items, dtype=np.int32)
            uid_array.fill(user_id)
            predictions = model.predict(uid_array, pid_array, num_threads=4)
    
            true_pids = row.indices[row.data == 1]
    
            grnd = np.zeros(no_items, dtype=np.int32)
            grnd[true_pids] = 1
    
            if len(true_pids):
                scores.append(roc_auc_score(grnd, predictions))
    
        return sum(scores) / len(scores)


Now let's train a BPR model and look at its accuracy.


        model = LightFM(learning_rate=0.05, loss='bpr')
    
        model.fit_partial(train,
                          epochs=10)
    
        train_precision = precision_at_k(model,
                                         train,
                                         10)
        test_precision = precision_at_k(model,
                                        test,
                                        10)
    
        train_auc = full_auc(model, train)
        test_auc = full_auc(model, test)
        
        print('Precision: %s, %s' % (train_precision, test_precision))
        print('AUC: %s, %s' % (train_auc, test_auc))

    Precision: 0.421208907741, 0.0622481442206
    AUC: 0.838544064431, 0.819069444911


The WARP model, on the other hand, optimises for precision@k---we should expect its performance to be better on precision.


        model = LightFM(learning_rate=0.05, loss='warp')
    
        model.fit_partial(train,
                          epochs=10)
    
        train_precision = precision_at_k(model,
                                         train,
                                         10)
        test_precision = precision_at_k(model,
                                        test,
                                        10)
    
        train_auc = full_auc(model, train)
        test_auc = full_auc(model, test)
        
        print('Precision: %s, %s' % (train_precision, test_precision))
        print('AUC: %s, %s' % (train_auc, test_auc))

    Precision: 0.624708377519, 0.110816542948
    AUC: 0.941236748837, 0.904416726513


And that is exactly what we see: we get much higher precision@10 (but the AUC metric is also improved).


    


    
