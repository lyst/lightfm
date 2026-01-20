//! Prediction functions for LightFM

use numpy::{PyReadonlyArray1, PyReadwriteArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::csr_matrix::CSRMatrix;
use crate::model::FastLightFM;
use crate::repr::{
    compute_item_representation, compute_prediction_from_repr, compute_user_representation,
    new_repr,
};

/// Generate predictions for user-item pairs.
#[pyfunction]
#[pyo3(signature = (item_features, user_features, user_ids, item_ids, predictions, lightfm, num_threads))]
pub fn predict_lightfm(
    py: Python<'_>,
    item_features: &CSRMatrix,
    user_features: &CSRMatrix,
    user_ids: PyReadonlyArray1<i32>,
    item_ids: PyReadonlyArray1<i32>,
    mut predictions: PyReadwriteArray1<f32>,
    lightfm: &FastLightFM,
    num_threads: i32,
) -> PyResult<()> {
    let user_ids = user_ids.as_slice()?;
    let item_ids = item_ids.as_slice()?;
    let predictions_slice = predictions.as_slice_mut()?;

    let no_examples = predictions_slice.len();

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
            // We need to collect results because we can't safely write to predictions in parallel
            let results: Vec<(usize, f32)> = (0..no_examples)
                .into_par_iter()
                .map(|i| {
                    let mut user_repr = new_repr(lightfm_view.no_components);
                    let mut item_repr = new_repr(lightfm_view.no_components);

                    compute_user_representation(
                        &user_features_view,
                        &lightfm_view,
                        user_ids[i],
                        &mut user_repr,
                    );
                    compute_item_representation(
                        &item_features_view,
                        &lightfm_view,
                        item_ids[i],
                        &mut item_repr,
                    );

                    let prediction = compute_prediction_from_repr(
                        &user_repr,
                        &item_repr,
                        lightfm_view.no_components,
                    );

                    (i, prediction as f32)
                })
                .collect();

            // Write results back
            for (i, pred) in results {
                predictions_slice[i] = pred;
            }
        });
    });

    Ok(())
}

/// Compute ranking positions for test interactions.
#[pyfunction]
#[pyo3(signature = (item_features, user_features, test_interactions, train_interactions, ranks, lightfm, num_threads))]
pub fn predict_ranks(
    py: Python<'_>,
    item_features: &CSRMatrix,
    user_features: &CSRMatrix,
    test_interactions: &CSRMatrix,
    train_interactions: &CSRMatrix,
    mut ranks: PyReadwriteArray1<f32>,
    lightfm: &FastLightFM,
    num_threads: i32,
) -> PyResult<()> {
    let ranks_slice = ranks.as_slice_mut()?;

    // Create views for thread-safe access
    let item_features_view = item_features.view();
    let user_features_view = user_features.view();
    let test_interactions_view = test_interactions.view();
    let train_interactions_view = train_interactions.view();
    let lightfm_view = lightfm.view();

    let num_users = test_interactions_view.rows;
    let num_items = test_interactions_view.cols;

    // Figure out max predictions buffer size
    let mut max_predictions_size = 0;
    for user_id in 0..num_users {
        let row_size = test_interactions_view.get_row_end(user_id)
            - test_interactions_view.get_row_start(user_id);
        if row_size > max_predictions_size {
            max_predictions_size = row_size;
        }
    }

    // Release GIL for parallel computation
    py.allow_threads(|| {
        // Set up thread pool
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads as usize)
            .build()
            .unwrap();

        pool.install(|| {
            // Collect rank updates from all users
            let rank_updates: Vec<(i32, f32)> = (0..num_users)
                .into_par_iter()
                .flat_map(|user_id| {
                    let row_start = test_interactions_view.get_row_start(user_id);
                    let row_stop = test_interactions_view.get_row_end(user_id);

                    if row_stop == row_start {
                        // No test interactions for this user
                        return Vec::new();
                    }

                    let mut user_repr = new_repr(lightfm_view.no_components);
                    let mut item_repr = new_repr(lightfm_view.no_components);

                    // Compute user representation
                    compute_user_representation(
                        &user_features_view,
                        &lightfm_view,
                        user_id,
                        &mut user_repr,
                    );

                    // Compute predictions for test items
                    let mut item_ids = Vec::with_capacity((row_stop - row_start) as usize);
                    let mut predictions = Vec::with_capacity((row_stop - row_start) as usize);

                    for i in row_start..row_stop {
                        let item_id = test_interactions_view.index(i);
                        compute_item_representation(
                            &item_features_view,
                            &lightfm_view,
                            item_id,
                            &mut item_repr,
                        );
                        let prediction = compute_prediction_from_repr(
                            &user_repr,
                            &item_repr,
                            lightfm_view.no_components,
                        );
                        item_ids.push(item_id);
                        predictions.push(prediction as f32);
                    }

                    // Initialize rank increments
                    let mut rank_increments = vec![0.0f32; (row_stop - row_start) as usize];

                    // Compare against all items
                    for item_id in 0..num_items {
                        // Skip items in training set
                        if train_interactions_view.in_positives(item_id, user_id) {
                            continue;
                        }

                        compute_item_representation(
                            &item_features_view,
                            &lightfm_view,
                            item_id,
                            &mut item_repr,
                        );
                        let prediction = compute_prediction_from_repr(
                            &user_repr,
                            &item_repr,
                            lightfm_view.no_components,
                        ) as f32;

                        // Count how many test items this item ranks above
                        for j in 0..(row_stop - row_start) as usize {
                            if item_id != item_ids[j] && prediction >= predictions[j] {
                                rank_increments[j] += 1.0;
                            }
                        }
                    }

                    // Return (index, rank_increment) pairs
                    (0..(row_stop - row_start))
                        .map(|j| (row_start + j, rank_increments[j as usize]))
                        .collect::<Vec<_>>()
                })
                .collect();

            // Apply rank updates
            for (idx, increment) in rank_updates {
                ranks_slice[idx as usize] += increment;
            }
        });
    });

    Ok(())
}

/// Calculate AUC from rank data.
#[pyfunction]
#[pyo3(signature = (ranks, num_train_positives, rank_data, auc, num_threads))]
pub fn calculate_auc_from_rank(
    py: Python<'_>,
    ranks: &CSRMatrix,
    num_train_positives: PyReadonlyArray1<i32>,
    mut rank_data: PyReadwriteArray1<f32>,
    mut auc: PyReadwriteArray1<f32>,
    num_threads: i32,
) -> PyResult<()> {
    let num_train_positives = num_train_positives.as_slice()?;
    let rank_data_slice = rank_data.as_slice_mut()?;
    let auc_slice = auc.as_slice_mut()?;

    let ranks_view = ranks.view();
    let num_users = ranks_view.rows;
    let num_items = ranks_view.cols;

    // Release GIL for parallel computation
    py.allow_threads(|| {
        // Set up thread pool
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads as usize)
            .build()
            .unwrap();

        pool.install(|| {
            // Calculate AUC for each user
            let auc_results: Vec<(i32, f32)> = (0..num_users)
                .into_par_iter()
                .map(|user_id| {
                    let row_start = ranks_view.get_row_start(user_id);
                    let row_stop = ranks_view.get_row_end(user_id);

                    let num_positives = row_stop - row_start;
                    let num_negatives =
                        num_items - (num_positives + num_train_positives[user_id as usize]);

                    // If there's only one class, return 0.5
                    if num_positives == 0 || num_negatives == num_items {
                        return (user_id, 0.5);
                    }

                    // Get ranks for this user and sort them
                    let mut user_ranks: Vec<f32> = (row_start..row_stop)
                        .map(|i| rank_data_slice[i as usize])
                        .collect();
                    user_ranks.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                    // Calculate AUC
                    let mut user_auc = 0.0f32;
                    for (i, &rank) in user_ranks.iter().enumerate() {
                        // Adjust rank by subtracting number of higher-ranked positives
                        let mut adjusted_rank = rank - i as f32;
                        if adjusted_rank < 0.0 {
                            adjusted_rank = 0.0;
                        }

                        // Probability of rank inversion
                        user_auc += 1.0 - adjusted_rank / num_negatives as f32;
                    }

                    if num_positives > 0 {
                        user_auc /= num_positives as f32;
                    }

                    (user_id, user_auc)
                })
                .collect();

            // Write results back
            for (user_id, user_auc) in auc_results {
                auc_slice[user_id as usize] = user_auc;
            }
        });
    });

    Ok(())
}
