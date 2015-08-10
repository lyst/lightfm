#!python
#cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

import numpy as np
from cython.parallel import parallel, prange
from libc.stdlib cimport rand
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


cdef inline double sigmoid(double v) nogil:
    """
    Compute the sigmoid of v.
    """

    return 1.0 / (1.0 + exp(-v))


cdef inline double compute_component_sum(CSRMatrix feature_indices,
                                         flt[:, ::1] features,
                                         int component,
                                         int start,
                                         int stop) nogil:
    """
    Compute the sum of given features along a given component.
    """

    cdef int i, feature
    cdef double component_sum, feature_weight

    component_sum = 0.0

    for i in range(start, stop):

        feature = feature_indices.indices[i]
        feature_weight = feature_indices.data[i]

        component_sum += feature_weight * features[feature, component]

    return component_sum


cdef inline double compute_bias_sum(CSRMatrix feature_indices,
                                    flt[::1] biases,
                                    int start,
                                    int stop) nogil:
    """
    Compute the sum of bias terms for given features.
    """

    cdef int i, feature
    cdef double bias_sum, feature_weight

    bias_sum = 0.0

    for i in range(start, stop):

        feature = feature_indices.indices[i]
        feature_weight = feature_indices.data[i]

        bias_sum += feature_weight * biases[feature]

    return bias_sum


cdef inline double compute_prediction(CSRMatrix item_features,
                                      CSRMatrix user_features,
                                      int user_id,
                                      int item_id,
                                      FastLightFM lightfm) nogil:
    """
    Compute prediction.
    """

    cdef int i, j, item_start_index, item_stop_index
    cdef int user_start_index, user_stop_index
    cdef int feature

    cdef double item_component, user_component
    cdef double prediction
    cdef double feature_weight

    # Get the iteration ranges for features
    # for this training example.
    item_start_index = item_features.get_row_start(item_id)
    item_stop_index = item_features.get_row_end(item_id)

    user_start_index = user_features.get_row_start(user_id)
    user_stop_index = user_features.get_row_end(user_id)

    # Initialize prediction.
    prediction = 0.0

    # Add the inner product of feature blocks. Results are
    # scaled down by accumulated lazy regularization.
    for i in range(lightfm.no_components):

        item_component = (compute_component_sum(item_features, lightfm.item_features, i,
                                                item_start_index, item_stop_index)
                          * lightfm.item_scale)
        user_component = (compute_component_sum(user_features, lightfm.user_features, i,
                                                user_start_index, user_stop_index)
                          * lightfm.user_scale)

        prediction += item_component * user_component

    # Add biases. Scaled down by lazy regularization.
    prediction += compute_bias_sum(item_features, lightfm.item_biases,
                                   item_start_index, item_stop_index) * lightfm.item_scale
    prediction += compute_bias_sum(user_features, lightfm.user_biases,
                                   user_start_index, user_stop_index) * lightfm.user_scale

    return prediction


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
                        FastLightFM lightfm,
                        double learning_rate,
                        double item_alpha,
                        double user_alpha) nogil:
    """
    Apply the gradient step.
    """

    cdef int i, j, item_start_index, item_stop_index, user_start_index, user_stop_index
    cdef double avg_learning_rate, item_component, user_component

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

        item_component = (compute_component_sum(item_features, lightfm.item_features, i,
                                                item_start_index, item_stop_index)
                          * lightfm.item_scale)
        user_component = (compute_component_sum(user_features, lightfm.user_features, i,
                                                user_start_index, user_stop_index)
                          * lightfm.user_scale)

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
                             FastLightFM lightfm,
                             double learning_rate,
                             double item_alpha,
                             double user_alpha) nogil:
    """
    Apply the gradient step.
    """

    cdef int i, j, positive_item_start_index, positive_item_stop_index
    cdef int  user_start_index, user_stop_index, negative_item_start_index, negative_item_stop_index
    cdef double avg_learning_rate, positive_item_component, negative_item_component, user_component

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

        positive_item_component = (compute_component_sum(item_features, lightfm.item_features, i,
                                                         positive_item_start_index,
                                                         positive_item_stop_index)
                                   * lightfm.item_scale)
        negative_item_component = (compute_component_sum(item_features, lightfm.item_features, i,
                                                         negative_item_start_index,
                                                         negative_item_stop_index)
                                   * lightfm.item_scale)
        user_component = (compute_component_sum(user_features, lightfm.user_features, i,
                                                user_start_index, user_stop_index)
                          * lightfm.user_scale)

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

    no_examples = Y.shape[0]

    with nogil:
        for i in prange(no_examples, num_threads=num_threads):
            row = shuffle_indices[i]

            user_id = user_ids[row]
            item_id = item_ids[row]

            prediction = sigmoid(compute_prediction(item_features,
                                                    user_features,
                                                    user_id,
                                                    item_id,
                                                    lightfm))

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
                   lightfm,
                   learning_rate,
                   item_alpha,
                   user_alpha)

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
    cdef unsigned int[::1] random_states

    random_states = np.random.randint(0,
                                      np.iinfo(np.int32).max,
                                      size=num_threads).astype(np.uint32)

    no_examples = Y.shape[0]
    gamma = 10
    MAX_LOSS = 10.0

    max_sampled = item_features.rows / gamma

    with nogil:
        for i in prange(no_examples, num_threads=num_threads):
            row = shuffle_indices[i]

            user_id = user_ids[row]
            positive_item_id = item_ids[row]

            if not Y[row] == 1:
                continue

            positive_prediction = compute_prediction(item_features,
                                                     user_features,
                                                     user_id,
                                                     positive_item_id,
                                                     lightfm)

            violation = 0
            sampled = 0

            while sampled < max_sampled:

                sampled = sampled + 1
                negative_item_id = (rand_r(&random_states[openmp.omp_get_thread_num()])
                                    % item_features.rows)

                if positive_item_id == negative_item_id:
                    break

                negative_prediction = compute_prediction(item_features,
                                                         user_features,
                                                         user_id,
                                                         negative_item_id,
                                                         lightfm)

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
                                lightfm,
                                learning_rate,
                                item_alpha,
                                user_alpha)
                    break

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

    with nogil:
        for i in prange(no_examples, num_threads=num_threads):
            row = shuffle_indices[i]

            if not Y[row] == 1:
                continue

            user_id = user_ids[row]
            positive_item_id = item_ids[row]
            negative_item_id = (rand_r(&random_states[openmp.omp_get_thread_num()])
                                % item_features.rows)

            positive_prediction = compute_prediction(item_features,
                                                     user_features,
                                                     user_id,
                                                     positive_item_id,
                                                     lightfm)
            negative_prediction = compute_prediction(item_features,
                                                     user_features,
                                                     user_id,
                                                     negative_item_id,
                                                     lightfm)

            warp_update(sigmoid(positive_prediction - negative_prediction),
                        item_features,
                        user_features,
                        user_id,
                        positive_item_id,
                        negative_item_id,
                        lightfm,
                        learning_rate,
                        item_alpha,
                        user_alpha)

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

    no_examples = predictions.shape[0]

    with nogil:
        for i in prange(no_examples, num_threads=num_threads):
            predictions[i] = compute_prediction(item_features,
                                                user_features,
                                                user_ids[i],
                                                item_ids[i],
                                                lightfm)
