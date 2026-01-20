//! BPR (Bayesian Personalized Ranking) loss function

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::csr_matrix::CSRMatrix;
use crate::model::FastLightFM;
use crate::optim::{check_and_regularize, regularize, warp_update};
use crate::repr::{
    compute_item_representation, compute_prediction_from_repr, compute_user_representation,
    new_repr,
};
use crate::utils::{sigmoid, ThreadRng};

/// Fit the LightFM model using BPR loss.
#[pyfunction]
#[pyo3(signature = (item_features, user_features, interactions, user_ids, item_ids, y, sample_weight, shuffle_indices, lightfm, _learning_rate, item_alpha, user_alpha, num_threads, random_state))]
pub fn fit_bpr(
    py: Python<'_>,
    item_features: &CSRMatrix,
    user_features: &CSRMatrix,
    interactions: &CSRMatrix,
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
    random_state: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let user_ids = user_ids.as_slice()?;
    let item_ids = item_ids.as_slice()?;
    let y = y.as_slice()?;
    let sample_weight = sample_weight.as_slice()?;
    let shuffle_indices = shuffle_indices.as_slice()?;

    let no_examples = y.len();

    // Generate random seeds for each thread
    let random_seeds: Vec<u32> = random_state
        .call_method1(
            "randint",
            (0i64, i32::MAX as i64, (num_threads as usize,)),
        )?
        .call_method1("astype", ("uint32",))?
        .extract::<Vec<u32>>()?;

    // Create views for thread-safe access
    let item_features_view = item_features.view();
    let user_features_view = user_features.view();
    let interactions_view = interactions.view();
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

                // Skip negative interactions
                if y[row] <= 0.0 {
                    return;
                }

                let weight = sample_weight[row] as f64;
                let user_id = user_ids[row];
                let positive_item_id = item_ids[row];

                // Get thread-local RNG (use thread index mod num_threads)
                let thread_id = rayon::current_thread_index().unwrap_or(0) % num_threads as usize;
                let mut rng = ThreadRng::new(random_seeds[thread_id] as u64 + i as u64);

                // Sample a negative item
                let mut negative_item_id = item_ids[(rng.randint(no_examples as i32)) as usize];

                // Try to find a negative item that's not in positives
                for _ in 0..no_examples {
                    negative_item_id = item_ids[(rng.randint(no_examples as i32)) as usize];
                    if !interactions_view.in_positives(negative_item_id, user_id) {
                        break;
                    }
                }

                // Allocate thread-local representations
                let mut user_repr = new_repr(lightfm_view.no_components);
                let mut pos_item_repr = new_repr(lightfm_view.no_components);
                let mut neg_item_repr = new_repr(lightfm_view.no_components);

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
                        positive_item_id,
                        &mut pos_item_repr,
                    );
                    compute_item_representation(
                        &item_features_view,
                        &lightfm_view,
                        negative_item_id,
                        &mut neg_item_repr,
                    );

                    // Compute predictions
                    let positive_prediction = compute_prediction_from_repr(
                        &user_repr,
                        &pos_item_repr,
                        lightfm_view.no_components,
                    );
                    let negative_prediction = compute_prediction_from_repr(
                        &user_repr,
                        &neg_item_repr,
                        lightfm_view.no_components,
                    );

                    // BPR loss: weight * (1 - sigmoid(pos - neg))
                    let loss = weight * (1.0 - sigmoid(positive_prediction - negative_prediction));

                    // Apply gradient update
                    let (item_scale_update, user_scale_update) = warp_update(
                        loss,
                        &item_features_view,
                        &user_features_view,
                        user_id,
                        positive_item_id,
                        negative_item_id,
                        &user_repr,
                        &pos_item_repr,
                        &neg_item_repr,
                        &lightfm_view,
                        item_alpha,
                        user_alpha,
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
