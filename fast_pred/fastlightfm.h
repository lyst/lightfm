#ifndef FASTLIGHTFM_H
#define FASTLIGHTFM_H

#include "cnpy.h"
#include <string>

struct CSRMatrix;

/**
 * Class holds all the model state.
 */
class FastLightFM {
  CSRMatrix *item_features;
  CSRMatrix *user_features;

  cnpy::NpyArray item_embeddings;
  cnpy::NpyArray item_biases;

  cnpy::NpyArray user_embeddings;
  cnpy::NpyArray user_biases;

  int no_components;

  double item_scale;
  double user_scale;

  struct FastLightFMCache *lightfm_cache;

public:
  FastLightFM()
      : item_features(nullptr), user_features(nullptr), no_components(0),
        item_scale(0.0), user_scale(0.0), lightfm_cache(nullptr) {}

  virtual ~FastLightFM();

  bool load(std::string dir);

  void predict(CSRMatrix *item_features, CSRMatrix *user_features,
               int *user_ids, int *item_ids, double *predictions,
               int no_examples, long *top_k_indice, long top_k);
};

#endif // FASTLIGHTFM_H
