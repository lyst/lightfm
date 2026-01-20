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
    gradient: f64,
    alpha: f64,
    is_user: bool,
) -> f64 {
    let mut sum_learning_rate = 0.0;

    if lightfm.adadelta {
        for i in start..stop {
            let feature = feature_indices.index(i);
            let feature_weight = feature_indices.value(i) as f64;

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

            let new_grad = lightfm.rho as f64 * grad as f64
                + (1.0 - lightfm.rho as f64) * (feature_weight * gradient).powi(2);
            let local_learning_rate =
                ((mom as f64 + lightfm.eps as f64) / (new_grad + lightfm.eps as f64)).sqrt();
            let update = local_learning_rate * gradient * feature_weight;
            let new_mom =
                lightfm.rho as f64 * mom as f64 + (1.0 - lightfm.rho as f64) * update.powi(2);
            let new_bias = (bias as f64 - update) * (1.0 + alpha * local_learning_rate);

            if is_user {
                lightfm.set_user_bias_gradient(feature, new_grad as f32);
                lightfm.set_user_bias_momentum(feature, new_mom as f32);
                lightfm.set_user_bias(feature, new_bias as f32);
            } else {
                lightfm.set_item_bias_gradient(feature, new_grad as f32);
                lightfm.set_item_bias_momentum(feature, new_mom as f32);
                lightfm.set_item_bias(feature, new_bias as f32);
            }

            sum_learning_rate += local_learning_rate;
        }
    } else {
        // Adagrad
        for i in start..stop {
            let feature = feature_indices.index(i);
            let feature_weight = feature_indices.value(i) as f64;

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

            let local_learning_rate = lightfm.learning_rate / (grad as f64).sqrt();
            let new_bias = (bias as f64 - local_learning_rate * feature_weight * gradient)
                * (1.0 + alpha * local_learning_rate);
            let new_grad = grad as f64 + (gradient * feature_weight).powi(2);

            if is_user {
                lightfm.set_user_bias_gradient(feature, new_grad as f32);
                lightfm.set_user_bias(feature, new_bias as f32);
            } else {
                lightfm.set_item_bias_gradient(feature, new_grad as f32);
                lightfm.set_item_bias(feature, new_bias as f32);
            }

            sum_learning_rate += local_learning_rate;
        }
    }

    sum_learning_rate
}

/// Update features with Adagrad or Adadelta
#[inline]
pub fn update_features(
    feature_indices: &CSRMatrixView,
    component: i32,
    start: i32,
    stop: i32,
    lightfm: &FastLightFMView,
    gradient: f64,
    alpha: f64,
    is_user: bool,
) -> f64 {
    let mut sum_learning_rate = 0.0;

    if lightfm.adadelta {
        for i in start..stop {
            let feature = feature_indices.index(i);
            let feature_weight = feature_indices.value(i) as f64;

            let (grad, mom, feat) = if is_user {
                (
                    lightfm.get_user_feature_gradient(feature, component),
                    lightfm.get_user_feature_momentum(feature, component),
                    lightfm.get_user_feature(feature, component),
                )
            } else {
                (
                    lightfm.get_item_feature_gradient(feature, component),
                    lightfm.get_item_feature_momentum(feature, component),
                    lightfm.get_item_feature(feature, component),
                )
            };

            let new_grad = lightfm.rho as f64 * grad as f64
                + (1.0 - lightfm.rho as f64) * (feature_weight * gradient).powi(2);
            let local_learning_rate =
                ((mom as f64 + lightfm.eps as f64) / (new_grad + lightfm.eps as f64)).sqrt();
            let update = local_learning_rate * gradient * feature_weight;
            let new_mom =
                lightfm.rho as f64 * mom as f64 + (1.0 - lightfm.rho as f64) * update.powi(2);
            let new_feat = (feat as f64 - update) * (1.0 + alpha * local_learning_rate);

            if is_user {
                lightfm.set_user_feature_gradient(feature, component, new_grad as f32);
                lightfm.set_user_feature_momentum(feature, component, new_mom as f32);
                lightfm.set_user_feature(feature, component, new_feat as f32);
            } else {
                lightfm.set_item_feature_gradient(feature, component, new_grad as f32);
                lightfm.set_item_feature_momentum(feature, component, new_mom as f32);
                lightfm.set_item_feature(feature, component, new_feat as f32);
            }

            sum_learning_rate += local_learning_rate;
        }
    } else {
        // Adagrad
        for i in start..stop {
            let feature = feature_indices.index(i);
            let feature_weight = feature_indices.value(i) as f64;

            let (grad, feat) = if is_user {
                (
                    lightfm.get_user_feature_gradient(feature, component),
                    lightfm.get_user_feature(feature, component),
                )
            } else {
                (
                    lightfm.get_item_feature_gradient(feature, component),
                    lightfm.get_item_feature(feature, component),
                )
            };

            let local_learning_rate = lightfm.learning_rate / (grad as f64).sqrt();
            let new_feat = (feat as f64 - local_learning_rate * feature_weight * gradient)
                * (1.0 + alpha * local_learning_rate);
            let new_grad = grad as f64 + (gradient * feature_weight).powi(2);

            if is_user {
                lightfm.set_user_feature_gradient(feature, component, new_grad as f32);
                lightfm.set_user_feature(feature, component, new_feat as f32);
            } else {
                lightfm.set_item_feature_gradient(feature, component, new_grad as f32);
                lightfm.set_item_feature(feature, component, new_feat as f32);
            }

            sum_learning_rate += local_learning_rate;
        }
    }

    sum_learning_rate
}

/// Apply gradient update for a single positive interaction (logistic loss)
pub fn update(
    loss: f64,
    item_features: &CSRMatrixView,
    user_features: &CSRMatrixView,
    user_id: i32,
    item_id: i32,
    user_repr: &Repr,
    item_repr: &Repr,
    lightfm: &FastLightFMView,
    item_alpha: f64,
    user_alpha: f64,
) {
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
    for i in 0..lightfm.no_components {
        let user_component = user_repr[i as usize] as f64;
        let item_component = item_repr[i as usize] as f64;

        avg_learning_rate += update_features(
            item_features,
            i,
            item_start,
            item_stop,
            lightfm,
            loss * user_component,
            item_alpha,
            false,
        );
        avg_learning_rate += update_features(
            user_features,
            i,
            user_start,
            user_stop,
            lightfm,
            loss * item_component,
            user_alpha,
            true,
        );
    }

    // Compute average learning rate
    let num_updates = (lightfm.no_components + 1) * (user_stop - user_start)
        + (lightfm.no_components + 1) * (item_stop - item_start);
    if num_updates > 0 {
        avg_learning_rate /= num_updates as f64;
    }

    // Update regularization scales
    lightfm.multiply_item_scale(1.0 + item_alpha * avg_learning_rate);
    lightfm.multiply_user_scale(1.0 + user_alpha * avg_learning_rate);
}

/// Apply gradient update for WARP/BPR (pairwise loss)
pub fn warp_update(
    loss: f64,
    item_features: &CSRMatrixView,
    user_features: &CSRMatrixView,
    user_id: i32,
    positive_item_id: i32,
    negative_item_id: i32,
    user_repr: &Repr,
    pos_item_repr: &Repr,
    neg_item_repr: &Repr,
    lightfm: &FastLightFMView,
    item_alpha: f64,
    user_alpha: f64,
) {
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
    for i in 0..lightfm.no_components {
        let user_component = user_repr[i as usize] as f64;
        let pos_item_component = pos_item_repr[i as usize] as f64;
        let neg_item_component = neg_item_repr[i as usize] as f64;

        avg_learning_rate += update_features(
            item_features,
            i,
            pos_item_start,
            pos_item_stop,
            lightfm,
            -loss * user_component,
            item_alpha,
            false,
        );
        avg_learning_rate += update_features(
            item_features,
            i,
            neg_item_start,
            neg_item_stop,
            lightfm,
            loss * user_component,
            item_alpha,
            false,
        );
        avg_learning_rate += update_features(
            user_features,
            i,
            user_start,
            user_stop,
            lightfm,
            loss * (neg_item_component - pos_item_component),
            user_alpha,
            true,
        );
    }

    // Compute average learning rate
    let num_updates = (lightfm.no_components + 1) * (user_stop - user_start)
        + (lightfm.no_components + 1) * (pos_item_stop - pos_item_start)
        + (lightfm.no_components + 1) * (neg_item_stop - neg_item_start);
    if num_updates > 0 {
        avg_learning_rate /= num_updates as f64;
    }

    // Update regularization scales
    lightfm.multiply_item_scale(1.0 + item_alpha * avg_learning_rate);
    lightfm.multiply_user_scale(1.0 + user_alpha * avg_learning_rate);
}

/// Apply accumulated L2 regularization to all features
pub fn regularize(lightfm: &FastLightFMView) {
    let item_scale = lightfm.get_item_scale();
    let user_scale = lightfm.get_user_scale();

    // Regularize item features
    for i in 0..lightfm.num_item_features {
        for j in 0..lightfm.no_components {
            let val = lightfm.get_item_feature(i, j);
            lightfm.set_item_feature(i, j, (val as f64 / item_scale) as f32);
        }
        let bias = lightfm.get_item_bias(i);
        lightfm.set_item_bias(i, (bias as f64 / item_scale) as f32);
    }

    // Regularize user features
    for i in 0..lightfm.num_user_features {
        for j in 0..lightfm.no_components {
            let val = lightfm.get_user_feature(i, j);
            lightfm.set_user_feature(i, j, (val as f64 / user_scale) as f32);
        }
        let bias = lightfm.get_user_bias(i);
        lightfm.set_user_bias(i, (bias as f64 / user_scale) as f32);
    }

    lightfm.reset_item_scale();
    lightfm.reset_user_scale();
}

/// Check if regularization is needed and apply it (thread-safe)
pub fn check_and_regularize(lightfm: &FastLightFMView) {
    // Use a write lock to ensure only one thread regularizes
    let item_scale = lightfm.get_item_scale();
    let user_scale = lightfm.get_user_scale();

    if item_scale > MAX_REG_SCALE || user_scale > MAX_REG_SCALE {
        regularize(lightfm);
    }
}
