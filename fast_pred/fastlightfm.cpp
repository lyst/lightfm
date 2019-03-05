#include "math.h"
#include <memory.h>
#include <stdlib.h>
#include <string.h>

#include "fastlightfm.h"

#ifdef __APPLE__
typedef int (*compar_t)(void *, const void *, const void *);
#else
typedef int (*compar_t)(const void *, const void *, void *);
#endif // __APPLE__

static void qsort_with_context(void *base, int nmemb, int size, compar_t compar,
                               void *arg) {
#ifdef __APPLE__
  qsort_r(base, nmemb, size, arg, static_cast<compar_t>(compar));
#else
  qsort_r(base, nmemb, size, static_cast<compar_t>(compar), arg);
#endif // __APPLE__
}

/*
 * Utility class for accessing elements
 * of a CSR matrix.
 */
struct CSRMatrix {
  cnpy::NpyArray indices;
  cnpy::NpyArray indptr;
  cnpy::NpyArray data;

  int rows, cols, nnz;

  CSRMatrix(cnpy::NpyArray &indices, cnpy::NpyArray &indptr,
            cnpy::NpyArray &data, cnpy::NpyArray &shape) {
    this->indices = indices;
    this->indptr = indptr;
    this->data = data;
    this->rows = shape.data<int>()[0];
    this->cols = shape.data<int>()[1];
    this->nnz = data.num_vals;
  }

  /*
   * Return the pointer to the start of the
   * data for row.
   */
  int get_row_start(int row) { return indptr.data<int>()[row]; }

  /*
   *  Return the pointer to the end of the
   *  data for row.
   */
  int get_row_end(int row) { return indptr.data<int>()[row + 1]; }
};

struct FastLightFMCache {
  float *user_repr;
  int *user_repr_cached;
  float *item_repr;
  int *item_repr_cached;
  int stride;

  FastLightFMCache(int num_of_items, int num_of_users, int stride) {
    user_repr = (float *)malloc(sizeof(float) * num_of_users * stride);
    user_repr_cached = (int *)malloc(sizeof(int) * num_of_users);
    memset(user_repr_cached, 0, sizeof(int) * num_of_users);
    item_repr = (float *)malloc(sizeof(float) * num_of_items * stride);
    item_repr_cached = (int *)malloc(sizeof(int) * num_of_items);
    memset(item_repr_cached, 0, sizeof(int) * num_of_items);
    this->stride = stride;
  }

  virtual ~FastLightFMCache() {
    free(user_repr);
    free(user_repr_cached);
    free(item_repr);
    free(item_repr_cached);
  }
};

/*
 * Compute latent representation for row_id.
 * The last element of the representation is the bias.
 */
static void compute_representation(CSRMatrix *features,
                                   float * /*feature_embeddings*/,
                                   float *feature_biases, int no_components,
                                   int row_id, double scale,
                                   float *representation) {
  for (int i = 0; i < no_components + 1; i++) {
    representation[i] = 0.0;
  }

  int start_index = features->get_row_start(row_id);
  int stop_index = features->get_row_end(row_id);

  for (int i = start_index; i < stop_index; i++) {
    int feature = features->indices.data<int>()[i];
    float feature_weight = features->data.data<float>()[i] * scale;

    for (int j = 0; j < no_components; j++) {
      // TODO: the index needs to be fixed
      // representation[j] += feature_weight * feature_embeddings[feature, j];
    }

    representation[no_components] += feature_weight * feature_biases[feature];
  }
}

static float compute_prediction_from_repr(float *user_repr, float *item_repr,
                                          int no_components) {
  // Biases
  float result = user_repr[no_components] + item_repr[no_components];

  // Latent factor dot product
  for (int i = 0; i < no_components; i++) {
    result += user_repr[i] * item_repr[i];
  }

  return result;
}

#ifdef __APPLE__
static int pred_compar(void *arg, const void *a, const void *b) {
#else
static int pred_compar(const void *a, const void *b, void *arg) {
#endif // __APPLE__
  float x = ((float *)arg)[((int *)a)[0]];
  float y = ((float *)arg)[((int *)b)[0]];

  if (x > y) {
    return -1;
  } else if (x < y) {
    return 1;
  }

  return 0;
}

FastLightFM::~FastLightFM() {
  delete item_features;
  delete user_features;
}

void FastLightFM::predict(CSRMatrix *item_features, CSRMatrix *user_features,
                          int *user_ids, int *item_ids, double *predictions,
                          int no_examples, long *top_k_indice, long top_k) {
  /*
     Generate predictions.
     */
  int start_idx = 0;
  float min_value = 0;
  float pred = 0;
  int *pred_table = (int *)malloc(sizeof(int) * (top_k + 1));
  float *pred_value = (float *)malloc(sizeof(float) * no_examples);

  for (int i = 0; i < no_examples; i++) {
    if (lightfm_cache->user_repr_cached[user_ids[i]] == 0) {
      compute_representation(
          user_features, this->user_embeddings.data<float>(),
          this->user_biases.data<float>(), this->no_components, user_ids[i],
          this->user_scale,
          &lightfm_cache->user_repr[lightfm_cache->stride * user_ids[i]]);
      lightfm_cache->user_repr_cached[user_ids[i]] = 1;
    }

    if (lightfm_cache->item_repr_cached[item_ids[i]] == 0) {
      compute_representation(
          item_features, this->item_embeddings.data<float>(),
          this->item_biases.data<float>(), this->no_components, item_ids[i],
          this->item_scale,
          &lightfm_cache->item_repr[lightfm_cache->stride * item_ids[i]]);
      lightfm_cache->item_repr_cached[item_ids[i]] = 1;
    }

    pred = compute_prediction_from_repr(
        &lightfm_cache->user_repr[lightfm_cache->stride * user_ids[i]],
        &lightfm_cache->item_repr[lightfm_cache->stride * item_ids[i]],
        this->no_components);
    predictions[i] = pred_value[i] = pred;

    if (start_idx < top_k) {
      if (start_idx == 0) {
        min_value = pred;
      }

      pred_table[start_idx] = i;
      start_idx++;
      if (pred < min_value) {
        min_value = pred;
      }
    } else {
      if (pred <= min_value) {
        continue;
      }

      pred_table[top_k] = i;
      qsort_with_context(pred_table, top_k + 1, sizeof(int), pred_compar,
                         pred_value);
      min_value = pred_value[pred_table[top_k - 1]];
    }
  }

  for (int t = 0; t < top_k; top_k++) {
    top_k_indice[t] = pred_table[t];
  }

  free(pred_value);
  free(pred_table);
}

static CSRMatrix *loadCSRMatrix(cnpy::npz_t csrmatrix) {
  cnpy::NpyArray indices = csrmatrix["indices"];
  cnpy::NpyArray indptr = csrmatrix["indptr"];
  cnpy::NpyArray data = csrmatrix["data"];
  cnpy::NpyArray shape = csrmatrix["shape"];
  return new CSRMatrix(indices, indptr, data, shape);
}

// Load the model
bool FastLightFM::load(std::string dir) {
  cnpy::npz_t data = cnpy::npz_load(dir + "/model.npz");
  // TODO: we need to implement this.
  // model = LightFM(**hyper_params["model_params"])
  item_embeddings = data["item_embeddings"];
  item_biases = data["item_biases"];
  user_embeddings = data["user_embeddings"];
  user_biases = data["user_biases"];
  user_features = loadCSRMatrix(cnpy::npz_load(dir + "/user-features.npz"));
  item_features = loadCSRMatrix(cnpy::npz_load(dir + "/item-features.npz"));
  return true;
}
