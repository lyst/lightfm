#!python
#cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

import numpy as np
from cython.parallel import parallel, prange
from libc.stdlib cimport free, malloc
cimport openmp


ctypedef float flt


cdef extern from "math.h" nogil:
    double sqrt(double)
    double exp(double)
    double log(double)
    double floor(double)


cdef extern from "stdlib.h" nogil:
    int rand_r(unsigned int*)


cdef class CSRMatrix:
    """
    Utility class for accessing elements
    of a CSR matrix.
    """

    cdef int[::1] indices
    cdef int[::1] indptr
    cdef int[::1] data

    cdef int rows
    cdef int cols
    cdef int nnz

    def __init__(self, csr_matrix):

        self.indices = csr_matrix.indices
        self.indptr = csr_matrix.indptr
        self.data = csr_matrix.data

        self.rows, self.cols = csr_matrix.shape
        self.nnz = len(self.data)

    cdef int get_row_start(self, int row) nogil:
        """
        Return the pointer to the start of the
        data for row.
        """

        return self.indptr[row]

    cdef int get_row_end(self, int row) nogil:
        """
        Return the pointer to the end of the
        data for row.
        """

        return self.indptr[row + 1]


cdef class FastLightFM:
    """
    Class holding all the model state.
    """

    cdef flt[:, ::1] item_features
    cdef flt[:, ::1] item_feature_gradients

    cdef flt[::1] item_biases
    cdef flt[::1] item_bias_gradients

    cdef flt[:, ::1] user_features
    cdef flt[:, ::1] user_feature_gradients

    cdef flt[::1] user_biases
    cdef flt[::1] user_bias_gradients

    cdef int no_components

    cdef double item_scale
    cdef double user_scale

    def __init__(self,
                 flt[:, ::1] item_features,
                 flt[:, ::1] item_feature_gradients,
                 flt[::1] item_biases,
                 flt[::1] item_bias_gradients,
                 flt[:, ::1] user_features,
                 flt[:, ::1] user_feature_gradients,
                 flt[::1] user_biases,
                 flt[::1] user_bias_gradients,
                 int no_components):

        self.item_features = item_features
        self.item_feature_gradients = item_feature_gradients
        self.item_biases = item_biases
        self.item_bias_gradients = item_bias_gradients
        self.user_features = user_features
        self.user_feature_gradients = user_feature_gradients
        self.user_biases = user_biases
        self.user_bias_gradients = user_bias_gradients

        self.no_components = no_components
        self.item_scale = 1.0
        self.user_scale = 1.0


cdef inline flt sigmoid(flt v) nogil:
    """
    Compute the sigmoid of v.
    """

    return 1.0 / (1.0 + exp(-v))


cdef inline void compute_representation(CSRMatrix features,
                                        flt[:, ::1] feature_embeddings,
                                        flt[::1] feature_biases,
                                        FastLightFM lightfm,
                                        int row_id,
                                        double scale,
                                        float *representation) nogil:
    """
    Compute latent representation for row_id.
    The last element of the representation is the bias.
    """

    cdef int i, j, start_index, stop_index, feature
    cdef flt feature_weight

    start_index = features.get_row_start(row_id)
    stop_index = features.get_row_end(row_id)

    for i in range(lightfm.no_components + 1):
        representation[i] = 0.0

    for i in range(start_index, stop_index):

        feature = features.indices[i]
        feature_weight = features.data[i] * scale

        for j in range(lightfm.no_components):

            representation[j] += feature_weight * feature_embeddings[feature, j]

        representation[lightfm.no_components] += feature_weight * feature_biases[feature]


cdef inline flt compute_prediction_from_repr(flt *user_repr,
                                             flt *item_repr,
                                             int no_components) nogil:

    cdef int i
    cdef flt result

    # Biases
    result = user_repr[no_components] + item_repr[no_components]

    # Latent factor dot product
    for i in range(no_components):
        result += user_repr[i] * item_repr[i]

    return result


cdef inline double update_biases(CSRMatrix feature_indices,
                                 int start,
                                 int stop,
                                 flt[::1] biases,
                                 flt[::1] gradients,
                                 double gradient,
                                 double learning_rate,
                                 double alpha) nogil:
    """
    Perform a SGD update of the bias terms.
    """

    cdef int i, feature
    cdef double feature_weight, local_learning_rate, sum_learning_rate

    sum_learning_rate = 0.0

    for i in range(start, stop):

        feature = feature_indices.indices[i]
        feature_weight = feature_indices.data[i]

        local_learning_rate = learning_rate / sqrt(gradients[feature])
        biases[feature] -= local_learning_rate * feature_weight * gradient
        gradients[feature] += gradient ** 2

        # Lazy regularization: scale up by the regularization
        # parameter.
        biases[feature] *= (1.0 + alpha * local_learning_rate)

        sum_learning_rate += local_learning_rate

    return sum_learning_rate


cdef inline double update_features(CSRMatrix feature_indices,
                                   flt[:, ::1] features,
                                   flt[:, ::1] gradients,
                                   int component,
                                   int start,
                                   int stop,
                                   double gradient,
                                   double learning_rate,
                                   double alpha) nogil:
    """
    Update feature vectors.
    """

    cdef int i, feature,
    cdef double feature_weight, local_learning_rate, sum_learning_rate

    sum_learning_rate = 0.0

    for i in range(start, stop):

        feature = feature_indices.indices[i]
        feature_weight = feature_indices.data[i]

        local_learning_rate = learning_rate / sqrt(gradients[feature, component])
        features[feature, component] -= local_learning_rate * feature_weight * gradient
        gradients[feature, component] += gradient ** 2

        # Lazy regularization: scale up by the regularization
        # parameter.
        features[feature, component] *= (1.0 + alpha * local_learning_rate)

        sum_learning_rate += local_learning_rate

    return sum_learning_rate


cdef inline void update(double loss,
                        CSRMatrix item_features,
                        CSRMatrix user_features,
                        int user_id,
                        int item_id,
                        flt *user_repr,
                        flt *it_repr,
                        FastLightFM lightfm,
                        double learning_rate,
                        double item_alpha,
                        double user_alpha) nogil:
    """
    Apply the gradient step.
    """

    cdef int i, j, item_start_index, item_stop_index, user_start_index, user_stop_index
    cdef double avg_learning_rate
    cdef flt item_component, user_component

    avg_learning_rate = 0.0

    # Get the iteration ranges for features
    # for this training example.
    item_start_index = item_features.get_row_start(item_id)
    item_stop_index = item_features.get_row_end(item_id)

    user_start_index = user_features.get_row_start(user_id)
    user_stop_index = user_features.get_row_end(user_id)

    avg_learning_rate += update_biases(item_features, item_start_index, item_stop_index,
                                       lightfm.item_biases, lightfm.item_bias_gradients,
                                       loss, learning_rate, item_alpha)
    avg_learning_rate += update_biases(user_features, user_start_index, user_stop_index,
                                       lightfm.user_biases, lightfm.user_bias_gradients,
                                       loss, learning_rate, user_alpha)

    # Update latent representations.
    for i in range(lightfm.no_components):

        user_component = user_repr[i]
        item_component = it_repr[i]

        avg_learning_rate += update_features(item_features, lightfm.item_features,
                                             lightfm.item_feature_gradients,
                                             i, item_start_index, item_stop_index,
                                             loss * user_component, learning_rate, item_alpha)
        avg_learning_rate += update_features(user_features, lightfm.user_features,
                                             lightfm.user_feature_gradients,
                                             i, user_start_index, user_stop_index,
                                             loss * item_component, learning_rate, user_alpha)

    avg_learning_rate /= ((lightfm.no_components + 1) * (user_stop_index - user_start_index)
                          + (lightfm.no_components + 1) * (item_stop_index - item_start_index))

    # Update the scaling factors for lazy regularization, using the average learning rate
    # of features updated for this example.
    lightfm.item_scale *= (1 - item_alpha * avg_learning_rate)
    lightfm.user_scale *= (1 - user_alpha * avg_learning_rate)


cdef inline void warp_update(double loss,
                             CSRMatrix item_features,
                             CSRMatrix user_features,
                             int user_id,
                             int positive_item_id,
                             int negative_item_id,
                             flt *user_repr,
                             flt *pos_it_repr,
                             flt *neg_it_repr,
                             FastLightFM lightfm,
                             double learning_rate,
                             double item_alpha,
                             double user_alpha) nogil:
    """
    Apply the gradient step.
    """

    cdef int i, j, positive_item_start_index, positive_item_stop_index
    cdef int  user_start_index, user_stop_index, negative_item_start_index, negative_item_stop_index
    cdef double avg_learning_rate
    cdef flt positive_item_component, negative_item_component, user_component

    avg_learning_rate = 0.0

    # Get the iteration ranges for features
    # for this training example.
    positive_item_start_index = item_features.get_row_start(positive_item_id)
    positive_item_stop_index = item_features.get_row_end(positive_item_id)

    negative_item_start_index = item_features.get_row_start(negative_item_id)
    negative_item_stop_index = item_features.get_row_end(negative_item_id)

    user_start_index = user_features.get_row_start(user_id)
    user_stop_index = user_features.get_row_end(user_id)

    avg_learning_rate += update_biases(item_features, positive_item_start_index,
                                       positive_item_stop_index,
                                       lightfm.item_biases, lightfm.item_bias_gradients,
                                       -loss, learning_rate, item_alpha)
    avg_learning_rate += update_biases(item_features, negative_item_start_index,
                                       negative_item_stop_index,
                                       lightfm.item_biases, lightfm.item_bias_gradients,
                                       loss, learning_rate, item_alpha)
    avg_learning_rate += update_biases(user_features, user_start_index, user_stop_index,
                                       lightfm.user_biases, lightfm.user_bias_gradients,
                                       loss, learning_rate, user_alpha)

    # Update latent representations.
    for i in range(lightfm.no_components):

        user_component = user_repr[i]
        positive_item_component = pos_it_repr[i]
        negative_item_component = neg_it_repr[i]

        avg_learning_rate += update_features(item_features, lightfm.item_features,
                                             lightfm.item_feature_gradients,
                                             i, positive_item_start_index, positive_item_stop_index,
                                             -loss * user_component, learning_rate, item_alpha)
        avg_learning_rate += update_features(item_features, lightfm.item_features,
                                             lightfm.item_feature_gradients,
                                             i, negative_item_start_index, negative_item_stop_index,
                                             loss * user_component, learning_rate, item_alpha)
        avg_learning_rate += update_features(user_features, lightfm.user_features,
                                             lightfm.user_feature_gradients,
                                             i, user_start_index, user_stop_index,
                                             loss * (negative_item_component -
                                                     positive_item_component),
                                             learning_rate, user_alpha)

    avg_learning_rate /= ((lightfm.no_components + 1) * (user_stop_index - user_start_index)
                          + (lightfm.no_components + 1) *
                          (positive_item_stop_index - positive_item_start_index)
                          + (lightfm.no_components + 1)
                          * (negative_item_stop_index - negative_item_start_index))

    # Update the scaling factors for lazy regularization, using the average learning rate
    # of features updated for this example.
    lightfm.item_scale *= (1 - item_alpha * avg_learning_rate)
    lightfm.user_scale *= (1 - user_alpha * avg_learning_rate)


cdef inline void regularize(FastLightFM lightfm,
                            double item_alpha,
                            double user_alpha) nogil:
    """
    Apply accumulated L2 regularization to all features.
    """

    cdef int i, j
    cdef int no_features = lightfm.item_features.shape[0]
    cdef int no_users = lightfm.user_features.shape[0]

    for i in range(no_features):
        for j in range(lightfm.no_components):
            lightfm.item_features[i, j] *= lightfm.item_scale

        lightfm.item_biases[i] *= lightfm.item_scale

    for i in range(no_users):
        for j in range(lightfm.no_components):
            lightfm.user_features[i, j] *= lightfm.user_scale
        lightfm.user_biases[i] *= lightfm.user_scale

    lightfm.item_scale = 1.0
    lightfm.user_scale = 1.0


def fit_logistic(CSRMatrix item_features,
                 CSRMatrix user_features,
                 int[::1] user_ids,
                 int[::1] item_ids,
                 int[::1] Y,
                 int[::1] shuffle_indices,
                 FastLightFM lightfm,
                 double learning_rate,
                 double item_alpha,
                 double user_alpha,
                 int num_threads):
    """
    Fit the LightFM model.
    """

    cdef int i, no_examples, user_id, item_id, row
    cdef double prediction, loss
    cdef int y, y_row
    cdef flt *user_repr
    cdef flt *it_repr

    no_examples = Y.shape[0]

    with nogil, parallel(num_threads=num_threads):

        user_repr = <flt *>malloc(sizeof(flt) * (lightfm.no_components + 1))
        it_repr = <flt *>malloc(sizeof(flt) * (lightfm.no_components + 1))

        for i in prange(no_examples):

            row = shuffle_indices[i]

            user_id = user_ids[row]
            item_id = item_ids[row]

            compute_representation(user_features,
                                   lightfm.user_features,
                                   lightfm.user_biases,
                                   lightfm,
                                   user_id,
                                   lightfm.user_scale,
                                   user_repr)
            compute_representation(item_features,
                                   lightfm.item_features,
                                   lightfm.item_biases,
                                   lightfm,
                                   item_id,
                                   lightfm.item_scale,
                                   it_repr)

            prediction = sigmoid(compute_prediction_from_repr(user_repr,
                                                              it_repr,
                                                              lightfm.no_components))

            # Any value less or equal to zero
            # is a negative interaction.
            y_row = Y[row]
            if y_row <= 0:
                y = 0
            else:
                y = 1

            loss = (prediction - y)
            update(loss,
                   item_features,
                   user_features,
                   user_id,
                   item_id,
                   user_repr,
                   it_repr,
                   lightfm,
                   learning_rate,
                   item_alpha,
                   user_alpha)

        free(user_repr)
        free(it_repr)

    regularize(lightfm,
               item_alpha,
               user_alpha)


def fit_warp(CSRMatrix item_features,
             CSRMatrix user_features,
             int[::1] user_ids,
             int[::1] item_ids,
             int[::1] Y,
             int[::1] shuffle_indices,
             FastLightFM lightfm,
             double learning_rate,
             double item_alpha,
             double user_alpha,
             int num_threads):
    """
    Fit the model using the WARP loss.
    """

    cdef int i, no_examples, user_id, positive_item_id, gamma, max_sampled
    cdef int negative_item_id, sampled, row
    cdef double positive_prediction, negative_prediction, violation, weight
    cdef double loss, MAX_LOSS
    cdef flt *representation
    cdef unsigned int[::1] random_states

    random_states = np.random.randint(0,
                                      np.iinfo(np.int32).max,
                                      size=num_threads).astype(np.uint32)

    no_examples = Y.shape[0]
    gamma = 10
    MAX_LOSS = 10.0

    max_sampled = item_features.rows / gamma

    with nogil, parallel(num_threads=num_threads):

        user_repr = <flt *>malloc(sizeof(flt) * (lightfm.no_components + 1))
        pos_it_repr = <flt *>malloc(sizeof(flt) * (lightfm.no_components + 1))
        neg_it_repr = <flt *>malloc(sizeof(flt) * (lightfm.no_components + 1))

        for i in prange(no_examples):
            row = shuffle_indices[i]

            user_id = user_ids[row]
            positive_item_id = item_ids[row]

            if not Y[row] == 1:
                continue

            compute_representation(user_features,
                                   lightfm.user_features,
                                   lightfm.user_biases,
                                   lightfm,
                                   user_id,
                                   lightfm.user_scale,
                                   user_repr)
            compute_representation(item_features,
                                   lightfm.item_features,
                                   lightfm.item_biases,
                                   lightfm,
                                   positive_item_id,
                                   lightfm.item_scale,
                                   pos_it_repr)

            positive_prediction = compute_prediction_from_repr(user_repr,
                                                               pos_it_repr,
                                                               lightfm.no_components)

            violation = 0
            sampled = 0

            while sampled < max_sampled:

                sampled = sampled + 1
                negative_item_id = (rand_r(&random_states[openmp.omp_get_thread_num()])
                                    % item_features.rows)

                if positive_item_id == negative_item_id:
                    break

                compute_representation(item_features,
                                       lightfm.item_features,
                                       lightfm.item_biases,
                                       lightfm,
                                       negative_item_id,
                                       lightfm.item_scale,
                                       neg_it_repr)

                negative_prediction = compute_prediction_from_repr(user_repr,
                                                                   neg_it_repr,
                                                                   lightfm.no_components)

                if negative_prediction > positive_prediction - 1:
                    weight = log(floor((item_features.rows - 1) / sampled))
                    violation = 1 - positive_prediction + negative_prediction
                    loss = weight * violation

                    # Clip gradients for numerical stability.
                    if loss > MAX_LOSS:
                        loss = MAX_LOSS

                    warp_update(loss,
                                item_features,
                                user_features,
                                user_id,
                                positive_item_id,
                                negative_item_id,
                                user_repr,
                                pos_it_repr,
                                neg_it_repr,
                                lightfm,
                                learning_rate,
                                item_alpha,
                                user_alpha)
                    break

        free(user_repr)
        free(pos_it_repr)
        free(neg_it_repr)

    regularize(lightfm,
               item_alpha,
               user_alpha)


def fit_bpr(CSRMatrix item_features,
            CSRMatrix user_features,
            int[::1] user_ids,
            int[::1] item_ids,
            int[::1] Y,
            int[::1] shuffle_indices,
            FastLightFM lightfm,
            double learning_rate,
            double item_alpha,
            double user_alpha,
            int num_threads):
    """
    Fit the model using the BPR loss.
    """

    cdef int i, no_examples, user_id, positive_item_id
    cdef int negative_item_id, sampled, row
    cdef double positive_prediction, negative_prediction
    cdef unsigned int[::1] random_states

    random_states = np.random.randint(0,
                                      np.iinfo(np.int32).max,
                                      size=num_threads).astype(np.uint32)

    no_examples = Y.shape[0]

    with nogil, parallel(num_threads=num_threads):

        user_repr = <flt *>malloc(sizeof(flt) * (lightfm.no_components + 1))
        pos_it_repr = <flt *>malloc(sizeof(flt) * (lightfm.no_components + 1))
        neg_it_repr = <flt *>malloc(sizeof(flt) * (lightfm.no_components + 1))

        for i in prange(no_examples):
            row = shuffle_indices[i]

            if not Y[row] == 1:
                continue

            user_id = user_ids[row]
            positive_item_id = item_ids[row]
            negative_item_id = (rand_r(&random_states[openmp.omp_get_thread_num()])
                                % item_features.rows)

            compute_representation(user_features,
                                   lightfm.user_features,
                                   lightfm.user_biases,
                                   lightfm,
                                   user_id,
                                   lightfm.user_scale,
                                   user_repr)
            compute_representation(item_features,
                                   lightfm.item_features,
                                   lightfm.item_biases,
                                   lightfm,
                                   positive_item_id,
                                   lightfm.item_scale,
                                   pos_it_repr)
            compute_representation(item_features,
                                   lightfm.item_features,
                                   lightfm.item_biases,
                                   lightfm,
                                   negative_item_id,
                                   lightfm.item_scale,
                                   neg_it_repr)

            positive_prediction = compute_prediction_from_repr(user_repr,
                                                               pos_it_repr,
                                                               lightfm.no_components)
            negative_prediction = compute_prediction_from_repr(user_repr,
                                                               neg_it_repr,
                                                               lightfm.no_components)

            warp_update(sigmoid(positive_prediction - negative_prediction),
                        item_features,
                        user_features,
                        user_id,
                        positive_item_id,
                        negative_item_id,
                        user_repr,
                        pos_it_repr,
                        neg_it_repr,
                        lightfm,
                        learning_rate,
                        item_alpha,
                        user_alpha)

        free(user_repr)
        free(pos_it_repr)
        free(neg_it_repr)

    regularize(lightfm,
               item_alpha,
               user_alpha)


def predict_lightfm(CSRMatrix item_features,
                    CSRMatrix user_features,
                    int[::1] user_ids,
                    int[::1] item_ids,
                    double[::1] predictions,
                    FastLightFM lightfm,
                    int num_threads):
    """
    Generate predictions.
    """

    cdef int i, no_examples
    cdef flt *user_repr
    cdef flt *it_repr

    no_examples = predictions.shape[0]

    with nogil, parallel(num_threads=num_threads):

        user_repr = <flt *>malloc(sizeof(flt) * (lightfm.no_components + 1))
        it_repr = <flt *>malloc(sizeof(flt) * (lightfm.no_components + 1))

        for i in prange(no_examples):

            compute_representation(user_features,
                                   lightfm.user_features,
                                   lightfm.user_biases,
                                   lightfm,
                                   user_ids[i],
                                   lightfm.user_scale,
                                   user_repr)
            compute_representation(item_features,
                                   lightfm.item_features,
                                   lightfm.item_biases,
                                   lightfm,
                                   item_ids[i],
                                   lightfm.item_scale,
                                   it_repr)

            predictions[i] = compute_prediction_from_repr(user_repr,
                                                          it_repr,
                                                          lightfm.no_components)
