//! WARP k-OS (k-th Order Statistic) loss function

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
use crate::utils::{ThreadRng, MAX_LOSS};

/// A pair of (index, value) for sorting positive items by prediction
#[derive(Clone, Copy)]
struct Pair {
    idx: i32,
    val: f64,
}

/// Fit the LightFM model using WARP k-OS loss.
#[pyfunction]
#[pyo3(signature = (item_features, user_features, data, user_ids, shuffle_indices, lightfm, _learning_rate, item_alpha, user_alpha, k, n, num_threads, random_state))]
pub fn fit_warp_kos(
    py: Python<'_>,
    item_features: &CSRMatrix,
    user_features: &CSRMatrix,
    data: &CSRMatrix,
    user_ids: PyReadonlyArray1<i32>,
    shuffle_indices: PyReadonlyArray1<i32>,
    lightfm: &FastLightFM,
    _learning_rate: f64,
    item_alpha: f64,
    user_alpha: f64,
    k: i32,
    n: i32,
    num_threads: i32,
    random_state: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let user_ids = user_ids.as_slice()?;
    let shuffle_indices = shuffle_indices.as_slice()?;

    let no_examples = user_ids.len();

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
    let data_view = data.view();
    let lightfm_view = lightfm.view();
    let max_sampled = lightfm_view.max_sampled;
    let num_items = item_features_view.rows;

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

                // Get the user's positive items
                let user_pids_start = data_view.get_row_start(user_id);
                let user_pids_stop = data_view.get_row_end(user_id);

                if user_pids_stop == user_pids_start {
                    return;
                }

                // Get thread-local RNG
                let thread_id = rayon::current_thread_index().unwrap_or(0) % num_threads as usize;
                let mut rng = ThreadRng::new(random_seeds[thread_id] as u64 + i as u64);

                // Allocate thread-local representations
                let mut user_repr = new_repr(lightfm_view.no_components);
                let mut pos_item_repr = new_repr(lightfm_view.no_components);
                let mut neg_item_repr = new_repr(lightfm_view.no_components);

                // Compute user representation
                compute_user_representation(
                    &user_features_view,
                    &lightfm_view,
                    user_id,
                    &mut user_repr,
                );

                // Sample n positive items and compute their predictions
                let no_positives = n.min(user_pids_stop - user_pids_start);
                let mut pos_pairs: Vec<Pair> = Vec::with_capacity(no_positives as usize);

                for _ in 0..no_positives {
                    let sampled_positive_item_id = data_view.index(
                        rng.sample_range(user_pids_start, user_pids_stop),
                    );

                    compute_item_representation(
                        &item_features_view,
                        &lightfm_view,
                        sampled_positive_item_id,
                        &mut pos_item_repr,
                    );

                    let sampled_positive_prediction = compute_prediction_from_repr(
                        &user_repr,
                        &pos_item_repr,
                        lightfm_view.no_components,
                    );

                    pos_pairs.push(Pair {
                        idx: sampled_positive_item_id,
                        val: sampled_positive_prediction,
                    });
                }

                // Sort by prediction (descending - highest first)
                pos_pairs.sort_by(|a, b| {
                    b.val.partial_cmp(&a.val).unwrap_or(std::cmp::Ordering::Equal)
                });

                // Get the k-th order statistic (1-indexed, so k-1 for 0-indexed)
                let k_idx = (k.min(no_positives) - 1) as usize;
                let positive_item_id = pos_pairs[k_idx].idx;
                let positive_prediction = pos_pairs[k_idx].val;

                // Compute the positive item representation again (since we may have overwritten it)
                compute_item_representation(
                    &item_features_view,
                    &lightfm_view,
                    positive_item_id,
                    &mut pos_item_repr,
                );

                // Sample negatives until we find a violating one (WARP step)
                let mut sampled = 0;

                while sampled < max_sampled {
                    sampled += 1;
                    let negative_item_id = rng.randint(num_items);

                    // Compute negative item representation
                    compute_item_representation(
                        &item_features_view,
                        &lightfm_view,
                        negative_item_id,
                        &mut neg_item_repr,
                    );

                    let negative_prediction = compute_prediction_from_repr(
                        &user_repr,
                        &neg_item_repr,
                        lightfm_view.no_components,
                    );

                    // Check for margin violation
                    if negative_prediction > positive_prediction - 1.0 {
                        // Skip if the negative is actually a positive
                        if data_view.in_positives(negative_item_id, user_id) {
                            continue;
                        }

                        // Compute WARP loss: log(floor((num_items - 1) / sampled))
                        let mut loss = ((num_items - 1) as f64 / sampled as f64).floor().ln();

                        // Clip gradients for numerical stability
                        if loss > MAX_LOSS {
                            loss = MAX_LOSS;
                        }

                        // Apply gradient update
                        warp_update(
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

                        break;
                    }
                }

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
