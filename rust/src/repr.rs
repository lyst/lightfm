//! Representation computation helpers

use crate::csr_matrix::CSRMatrixView;
use crate::model::FastLightFMView;

/// Representation vector: [latent_factors..., bias]
/// Size is no_components + 1
pub type Repr = Vec<f32>;

/// Allocate a new representation vector
#[inline]
pub fn new_repr(no_components: i32) -> Repr {
    vec![0.0f32; (no_components + 1) as usize]
}

/// Compute latent representation for a row.
/// The representation is a weighted sum of feature embeddings.
/// The last element is the bias.
#[inline]
pub fn compute_representation(
    features: &CSRMatrixView,
    lightfm: &FastLightFMView,
    row_id: i32,
    scale: f32,
    is_user: bool,
    repr: &mut Repr,
) {
    let start_index = features.get_row_start(row_id);
    let stop_index = features.get_row_end(row_id);

    // Zero out the representation
    for i in 0..=lightfm.no_components {
        repr[i as usize] = 0.0;
    }

    for i in start_index..stop_index {
        let feature = features.index(i);
        let feature_weight = features.value(i) * scale;

        // Add weighted feature embeddings
        for j in 0..lightfm.no_components {
            let embedding = if is_user {
                lightfm.get_user_feature(feature, j)
            } else {
                lightfm.get_item_feature(feature, j)
            };
            repr[j as usize] += feature_weight * embedding;
        }

        // Add weighted bias
        let bias = if is_user {
            lightfm.get_user_bias(feature)
        } else {
            lightfm.get_item_bias(feature)
        };
        repr[lightfm.no_components as usize] += feature_weight * bias;
    }
}

/// Compute user representation
#[inline]
pub fn compute_user_representation(
    user_features: &CSRMatrixView,
    lightfm: &FastLightFMView,
    user_id: i32,
    repr: &mut Repr,
) {
    compute_representation(
        user_features,
        lightfm,
        user_id,
        lightfm.get_user_scale() as f32,
        true,
        repr,
    );
}

/// Compute item representation
#[inline]
pub fn compute_item_representation(
    item_features: &CSRMatrixView,
    lightfm: &FastLightFMView,
    item_id: i32,
    repr: &mut Repr,
) {
    compute_representation(
        item_features,
        lightfm,
        item_id,
        lightfm.get_item_scale() as f32,
        false,
        repr,
    );
}

/// Compute prediction from user and item representations.
/// Result is sum of biases + dot product of latent factors.
#[inline]
pub fn compute_prediction_from_repr(
    user_repr: &Repr,
    item_repr: &Repr,
    no_components: i32,
) -> f32 {
    let n = no_components as usize;
    // Add biases
    let mut result = user_repr[n] + item_repr[n];

    // Dot product of latent factors
    // Using iterators allows the compiler to optimize out bounds checks and auto-vectorize
    result += user_repr[0..n]
        .iter()
        .zip(item_repr[0..n].iter())
        .map(|(u, i)| u * i)
        .sum::<f32>();

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_prediction_from_repr() {
        let user_repr = vec![1.0, 2.0, 0.5]; // [latent1, latent2, bias]
        let item_repr = vec![2.0, 1.0, 0.3];

        let prediction = compute_prediction_from_repr(&user_repr, &item_repr, 2);

        // Expected: biases (0.5 + 0.3) + dot product (1*2 + 2*1) = 0.8 + 4.0 = 4.8
        assert!((prediction - 4.8).abs() < 1e-5);
    }
}
