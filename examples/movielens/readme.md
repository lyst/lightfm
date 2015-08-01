
# Getting the data
The first step is to get the movielens data.

Let's import the utility functions from `data.py`:


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

# Fitting the model
Let's import the lightfm model.


    from lightfm import LightFM


    model = LightFM(no_components=30)

In this case, we set the latent dimensionality of the model to 30. Fitting is straightforward.


    model.fit(train, epochs=50)




    <lightfm.lightfm.LightFM at 0x7feac361c190>



Let's try to get a handle on the model accuracy using the ROC AUC score.


    from sklearn.metrics import roc_auc_score
    
    train_predictions = model.predict(train.row,
                                      train.col)


    train_predictions




    array([ 0.57735386,  0.12810806,  0.70434413, ...,  0.37278502,
            0.1001321 ,  0.07673392])




    roc_auc_score(train.data, train_predictions)




    0.98793016085665009



We've got very high accuracy on the train dataset; let's check the test set.


    test_predictions = model.predict(test.row, test.col)


    roc_auc_score(test.data, test_predictions)




    0.72499325915332191



The accuracy is much lower on the test data, suggesting a high degree of overfitting. We can combat this by regularizing the model.


    model = LightFM(no_components=30, user_alpha=0.0001, item_alpha=0.0001)
    model.fit(train, epochs=50)
    roc_auc_score(test.data, model.predict(test.row, test.col))




    0.76052953487950203



A modicum of regularization gives much better results.

# Using metadata
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




    0.67178594791630175



This is not as accurate as a pure collaborative filtering solution, but should enable us to make recommendations new movies.

If we add item-specific features back, we should get the original accuracy back.


    item_features = data.get_movielens_item_metadata(use_item_ids=True)
    item_features
    model = LightFM(no_components=30, user_alpha=0.0001, item_alpha=0.0001)
    model.fit(train, item_features=item_features, epochs=50)
    roc_auc_score(test.data, model.predict(test.row, test.col, item_features=item_features))




    0.75693132377857264




    
