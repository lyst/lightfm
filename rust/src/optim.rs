//! Gradient update functions (Adagrad/Adadelta)

use crate::csr_matrix::CSRMatrixView;
use crate::model::FastLightFMView;
use crate::repr::Repr;
use crate::utils::MAX_REG_SCALE;

/// Update biases with Adagrad or Adadelta
#[inline]
pub fn update_biases(
    feature_indices: &CSRMatrixView,
    start: i32,
    stop: i32,
    lightfm: &FastLightFMView,
    gradient: f32,
    alpha: f32,
    is_user: bool,
) -> f32 {
    let mut sum_learning_rate = 0.0;

    if lightfm.adadelta {
        for i in start..stop {
            let feature = feature_indices.index(i);
            let feature_weight = feature_indices.value(i);

            let (grad, mom, bias) = if is_user {
                (
                    lightfm.get_user_bias_gradient(feature),
                    lightfm.get_user_bias_momentum(feature),
                    lightfm.get_user_bias(feature),
                )
            } else {
                (
                    lightfm.get_item_bias_gradient(feature),
                    lightfm.get_item_bias_momentum(feature),
                    lightfm.get_item_bias(feature),
                )
            };

            let new_grad = lightfm.rho * grad
                + (1.0 - lightfm.rho) * (feature_weight * gradient).powi(2);
            let local_learning_rate =
                ((mom + lightfm.eps) / (new_grad + lightfm.eps)).sqrt();
            let update = local_learning_rate * gradient * feature_weight;
            let new_mom =
                lightfm.rho * mom + (1.0 - lightfm.rho) * update.powi(2);
            let new_bias = (bias - update) * (1.0 + alpha * local_learning_rate);

            if is_user {
                lightfm.set_user_bias_gradient(feature, new_grad);
                lightfm.set_user_bias_momentum(feature, new_mom);
                lightfm.set_user_bias(feature, new_bias);
            } else {
                lightfm.set_item_bias_gradient(feature, new_grad);
                lightfm.set_item_bias_momentum(feature, new_mom);
                lightfm.set_item_bias(feature, new_bias);
            }

            sum_learning_rate += local_learning_rate;
        }
    } else {
        // Adagrad
        for i in start..stop {
            let feature = feature_indices.index(i);
            let feature_weight = feature_indices.value(i);

            let (grad, bias) = if is_user {
                (
                    lightfm.get_user_bias_gradient(feature),
                    lightfm.get_user_bias(feature),
                )
            } else {
                (
                    lightfm.get_item_bias_gradient(feature),
                    lightfm.get_item_bias(feature),
                )
            };

            let local_learning_rate = lightfm.learning_rate as f32 / grad.sqrt();
            let new_bias = (bias - local_learning_rate * feature_weight * gradient)
                * (1.0 + alpha * local_learning_rate);
            let new_grad = grad + (gradient * feature_weight).powi(2);

            if is_user {
                lightfm.set_user_bias_gradient(feature, new_grad);
                lightfm.set_user_bias(feature, new_bias);
            } else {
                lightfm.set_item_bias_gradient(feature, new_grad);
                lightfm.set_item_bias(feature, new_bias);
            }

            sum_learning_rate += local_learning_rate;
        }
    }

    sum_learning_rate
}

/// Update features with Adagrad or Adadelta
#[inline]
pub fn update_features_all(
    feature_indices: &CSRMatrixView,
    start: i32,
    stop: i32,
    lightfm: &FastLightFMView,
    loss: f32,
    multipliers: &[f32],
    alpha: f32,
    is_user: bool,
) -> f32 {
    let mut sum_learning_rate = 0.0;
    let no_components = lightfm.no_components;

    if lightfm.adadelta {
        for i in start..stop {
            let feature = feature_indices.index(i);
            let feature_weight = feature_indices.value(i);

            for c in 0..no_components {
                let gradient = loss * multipliers[c as usize];
                
                let (grad, mom, feat) = if is_user {
                    (
                        lightfm.get_user_feature_gradient(feature, c),
                        lightfm.get_user_feature_momentum(feature, c),
                        lightfm.get_user_feature(feature, c),
                    )
                } else {
                    (
                        lightfm.get_item_feature_gradient(feature, c),
                        lightfm.get_item_feature_momentum(feature, c),
                        lightfm.get_item_feature(feature, c),
                    )
                };

                let new_grad = lightfm.rho * grad
                    + (1.0 - lightfm.rho) * (feature_weight * gradient).powi(2);
                let local_learning_rate =
                    ((mom + lightfm.eps) / (new_grad + lightfm.eps)).sqrt();
                let update = local_learning_rate * gradient * feature_weight;
                let new_mom =
                    lightfm.rho * mom + (1.0 - lightfm.rho) * update.powi(2);
                let new_feat = (feat - update) * (1.0 + alpha * local_learning_rate);

                if is_user {
                    lightfm.set_user_feature_gradient(feature, c, new_grad);
                    lightfm.set_user_feature_momentum(feature, c, new_mom);
                    lightfm.set_user_feature(feature, c, new_feat);
                } else {
                    lightfm.set_item_feature_gradient(feature, c, new_grad);
                    lightfm.set_item_feature_momentum(feature, c, new_mom);
                    lightfm.set_item_feature(feature, c, new_feat);
                }

                sum_learning_rate += local_learning_rate;
            }
        }
    } else {
        // Adagrad
        for i in start..stop {
            let feature = feature_indices.index(i);
            let feature_weight = feature_indices.value(i);

            for c in 0..no_components {
                let gradient = loss * multipliers[c as usize];

                let (grad, feat) = if is_user {
                    (
                        lightfm.get_user_feature_gradient(feature, c),
                        lightfm.get_user_feature(feature, c),
                    )
                } else {
                    (
                        lightfm.get_item_feature_gradient(feature, c),
                        lightfm.get_item_feature(feature, c),
                    )
                };

                let local_learning_rate = lightfm.learning_rate as f32 / grad.sqrt();
                let new_feat = (feat - local_learning_rate * feature_weight * gradient)
                    * (1.0 + alpha * local_learning_rate);
                let new_grad = grad + (gradient * feature_weight).powi(2);

                if is_user {
                    lightfm.set_user_feature_gradient(feature, c, new_grad);
                    lightfm.set_user_feature(feature, c, new_feat);
                } else {
                    lightfm.set_item_feature_gradient(feature, c, new_grad);
                    lightfm.set_item_feature(feature, c, new_feat);
                }

                sum_learning_rate += local_learning_rate;
            }
        }
    }

    sum_learning_rate
}

/// Apply gradient update for a single positive interaction (logistic loss)
pub fn update(
    loss: f32,
    item_features: &CSRMatrixView,
    user_features: &CSRMatrixView,
    user_id: i32,
    item_id: i32,
    user_repr: &Repr,
    item_repr: &Repr,
    lightfm: &FastLightFMView,
    item_alpha: f32,
    user_alpha: f32,
) -> (f64, f64) {
    let item_start = item_features.get_row_start(item_id);
    let item_stop = item_features.get_row_end(item_id);
    let user_start = user_features.get_row_start(user_id);
    let user_stop = user_features.get_row_end(user_id);

    let mut avg_learning_rate = 0.0;

    // Update biases
    avg_learning_rate += update_biases(
        item_features,
        item_start,
        item_stop,
        lightfm,
        loss,
        item_alpha,
        false,
    );
    avg_learning_rate += update_biases(
        user_features,
        user_start,
        user_stop,
        lightfm,
        loss,
        user_alpha,
        true,
    );

    // Update latent factors
    avg_learning_rate += update_features_all(
        item_features,
        item_start,
        item_stop,
        lightfm,
        loss,
        &user_repr[0..lightfm.no_components as usize],
        item_alpha,
        false,
    );
    avg_learning_rate += update_features_all(
        user_features,
        user_start,
        user_stop,
        lightfm,
        loss,
        &item_repr[0..lightfm.no_components as usize],
        user_alpha,
        true,
    );

    // Compute average learning rate
    let num_updates = (lightfm.no_components + 1) * (user_stop - user_start)
        + (lightfm.no_components + 1) * (item_stop - item_start);
    if num_updates > 0 {
        avg_learning_rate /= num_updates as f32;
    }

    // Return regularization scale updates
    (
        1.0 + (item_alpha * avg_learning_rate) as f64,
        1.0 + (user_alpha * avg_learning_rate) as f64,
    )
}

/// Apply gradient update for WARP/BPR (pairwise loss)
pub fn warp_update(
    loss: f32,
    item_features: &CSRMatrixView,
    user_features: &CSRMatrixView,
    user_id: i32,
    positive_item_id: i32,
    negative_item_id: i32,
    user_repr: &Repr,
    pos_item_repr: &Repr,
    neg_item_repr: &Repr,
    lightfm: &FastLightFMView,
    item_alpha: f32,
    user_alpha: f32,
) -> (f64, f64) {
    let pos_item_start = item_features.get_row_start(positive_item_id);
    let pos_item_stop = item_features.get_row_end(positive_item_id);
    let neg_item_start = item_features.get_row_start(negative_item_id);
    let neg_item_stop = item_features.get_row_end(negative_item_id);
    let user_start = user_features.get_row_start(user_id);
    let user_stop = user_features.get_row_end(user_id);

    let mut avg_learning_rate = 0.0;

    // Update biases: positive gets -loss, negative gets +loss
    avg_learning_rate += update_biases(
        item_features,
        pos_item_start,
        pos_item_stop,
        lightfm,
        -loss,
        item_alpha,
        false,
    );
    avg_learning_rate += update_biases(
        item_features,
        neg_item_start,
        neg_item_stop,
        lightfm,
        loss,
        item_alpha,
        false,
    );
    avg_learning_rate += update_biases(
        user_features,
        user_start,
        user_stop,
        lightfm,
        loss,
        user_alpha,
        true,
    );

    // Update latent factors
    
    // Positive item: multiplier is user_repr, loss is -loss
    avg_learning_rate += update_features_all(
        item_features,
        pos_item_start,
        pos_item_stop,
        lightfm,
        -loss,
        &user_repr[0..lightfm.no_components as usize],
        item_alpha,
        false,
    );

    // Negative item: multiplier is user_repr, loss is loss
    avg_learning_rate += update_features_all(
        item_features,
        neg_item_start,
        neg_item_stop,
        lightfm,
        loss,
        &user_repr[0..lightfm.no_components as usize],
        item_alpha,
        false,
    );

    // User: multiplier is (neg_item - pos_item), loss is loss
    // We need to compute the difference vector
    let no_components = lightfm.no_components as usize;
    let mut diff_repr = vec![0.0f32; no_components];
    for c in 0..no_components {
        diff_repr[c] = neg_item_repr[c] - pos_item_repr[c];
    }

    avg_learning_rate += update_features_all(
        user_features,
        user_start,
        user_stop,
        lightfm,
        loss,
        &diff_repr,
        user_alpha,
        true,
    );

    // Compute average learning rate
    let num_updates = (lightfm.no_components + 1) * (user_stop - user_start)
        + (lightfm.no_components + 1) * (pos_item_stop - pos_item_start)
        + (lightfm.no_components + 1) * (neg_item_stop - neg_item_start);
    if num_updates > 0 {
        avg_learning_rate /= num_updates as f32;
    }

    // Return regularization scale updates
    (
        1.0 + (item_alpha * avg_learning_rate) as f64,
        1.0 + (user_alpha * avg_learning_rate) as f64,
    )
}

/// Apply accumulated L2 regularization to item features
pub fn regularize_items(lightfm: &FastLightFMView, item_scale: f64) {
    for i in 0..lightfm.num_item_features {
        for j in 0..lightfm.no_components {
            let val = lightfm.get_item_feature(i, j);
            lightfm.set_item_feature(i, j, (val as f64 / item_scale) as f32);
        }
        let bias = lightfm.get_item_bias(i);
        lightfm.set_item_bias(i, (bias as f64 / item_scale) as f32);
    }
}

/// Apply accumulated L2 regularization to user features
pub fn regularize_users(lightfm: &FastLightFMView, user_scale: f64) {
    for i in 0..lightfm.num_user_features {
        for j in 0..lightfm.no_components {
            let val = lightfm.get_user_feature(i, j);
            lightfm.set_user_feature(i, j, (val as f64 / user_scale) as f32);
        }
        let bias = lightfm.get_user_bias(i);
        lightfm.set_user_bias(i, (bias as f64 / user_scale) as f32);
    }
}

/// Apply accumulated L2 regularization to all features (called at end of epoch)
pub fn regularize(lightfm: &FastLightFMView) {
    // Regularize item features
    let item_scale = lightfm.get_item_scale();
    if item_scale > 1.0 {
        regularize_items(lightfm, item_scale);
        lightfm.reset_item_scale();
    }

    // Regularize user features
    let user_scale = lightfm.get_user_scale();
    if user_scale > 1.0 {
        regularize_users(lightfm, user_scale);
        lightfm.reset_user_scale();
    }
}

/// Check if regularization is needed and apply it (thread-safe, lock-free)
pub fn check_and_regularize(lightfm: &FastLightFMView) {
    use std::sync::atomic::Ordering;

    // Check item scale
    let item_scale_bits = lightfm.item_scale.load(Ordering::Relaxed);
    let item_scale = f64::from_bits(item_scale_bits);

    if item_scale > MAX_REG_SCALE {
        // Try to seize the regularization right by resetting scale to 1.0
        // If we succeed, we regularize using the OLD scale we just read.
        let one_bits = 1.0f64.to_bits();
        if lightfm.item_scale.compare_exchange(
            item_scale_bits,
            one_bits,
            Ordering::Relaxed,
            Ordering::Relaxed
        ).is_ok() {
             regularize_items(lightfm, item_scale);
        }
    }

    // Check user scale
    let user_scale_bits = lightfm.user_scale.load(Ordering::Relaxed);
    let user_scale = f64::from_bits(user_scale_bits);

    if user_scale > MAX_REG_SCALE {
        let one_bits = 1.0f64.to_bits();
        if lightfm.user_scale.compare_exchange(
            user_scale_bits,
            one_bits,
            Ordering::Relaxed,
            Ordering::Relaxed
        ).is_ok() {
             regularize_users(lightfm, user_scale);
        }
    }
}
