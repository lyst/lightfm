//! Logistic (sigmoid cross-entropy) loss function

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::csr_matrix::CSRMatrix;
use crate::model::FastLightFM;
use crate::optim::{check_and_regularize, regularize, update};
use crate::repr::{compute_item_representation, compute_prediction_from_repr, compute_user_representation, new_repr};
use crate::utils::sigmoid;

/// Fit the LightFM model using logistic loss.
#[pyfunction]
#[pyo3(signature = (item_features, user_features, user_ids, item_ids, y, sample_weight, shuffle_indices, lightfm, _learning_rate, item_alpha, user_alpha, num_threads))]
pub fn fit_logistic(
    py: Python<'_>,
    item_features: &CSRMatrix,
    user_features: &CSRMatrix,
    user_ids: PyReadonlyArray1<i32>,
    item_ids: PyReadonlyArray1<i32>,
    y: PyReadonlyArray1<f32>,
    sample_weight: PyReadonlyArray1<f32>,
    shuffle_indices: PyReadonlyArray1<i32>,
    lightfm: &FastLightFM,
    _learning_rate: f64,
    item_alpha: f64,
    user_alpha: f64,
    num_threads: i32,
) -> PyResult<()> {
    let user_ids = user_ids.as_slice()?;
    let item_ids = item_ids.as_slice()?;
    let y = y.as_slice()?;
    let sample_weight = sample_weight.as_slice()?;
    let shuffle_indices = shuffle_indices.as_slice()?;

    let no_examples = y.len();

    // Create views for thread-safe access
    let item_features_view = item_features.view();
    let user_features_view = user_features.view();
    let lightfm_view = lightfm.view();

    // Release GIL for parallel computation
    py.allow_threads(|| {
        // Set up thread pool
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads as usize)
            .build()
            .unwrap();

        pool.install(|| {
            // Process examples in parallel
            (0..no_examples).into_par_iter().for_each(|i| {
                let row = shuffle_indices[i] as usize;
                let user_id = user_ids[row];
                let item_id = item_ids[row];
                let weight = sample_weight[row] as f32;

                // Allocate thread-local representations
                let mut user_repr = new_repr(lightfm_view.no_components);
                let mut item_repr = new_repr(lightfm_view.no_components);

                {
                    // Compute representations
                    compute_user_representation(
                        &user_features_view,
                        &lightfm_view,
                        user_id,
                        &mut user_repr,
                    );
                    compute_item_representation(
                        &item_features_view,
                        &lightfm_view,
                        item_id,
                        &mut item_repr,
                    );

                    // Compute prediction
                    let prediction = sigmoid(compute_prediction_from_repr(
                        &user_repr,
                        &item_repr,
                        lightfm_view.no_components,
                    ));

                    // Compute target (any value <= 0 is negative)
                    let target = if y[row] <= 0.0 { 0.0 } else { 1.0 };

                    // Compute loss
                    let loss = weight * (prediction - target);

                    // Apply gradient update
                    let (item_scale_update, user_scale_update) = update(
                        loss,
                        &item_features_view,
                        &user_features_view,
                        user_id,
                        item_id,
                        &user_repr,
                        &item_repr,
                        &lightfm_view,
                        item_alpha as f32,
                        user_alpha as f32,
                    );
                    
                    lightfm_view.multiply_item_scale(item_scale_update);
                    lightfm_view.multiply_user_scale(user_scale_update);
                } // Read lock dropped here

                // Check for regularization
                check_and_regularize(&lightfm_view);
            });
        });

        // Final regularization
        regularize(&lightfm_view);

        // Sync scales back
        lightfm.sync_scales(&lightfm_view);
    });

    Ok(())
}
