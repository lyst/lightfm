//! FastLightFM model state container

use numpy::{PyReadwriteArray1, PyReadwriteArray2, PyUntypedArrayMethods};
use parking_lot::RwLock;
use pyo3::prelude::*;

/// View of model arrays with raw mutable pointers for parallel computation.
/// This struct is created from FastLightFM for use in training loops.
pub struct FastLightFMView {
    // Item embeddings and gradients
    pub item_features: *mut f32,
    pub item_feature_gradients: *mut f32,
    pub item_feature_momentum: *mut f32,
    pub item_biases: *mut f32,
    pub item_bias_gradients: *mut f32,
    pub item_bias_momentum: *mut f32,

    // User embeddings and gradients
    pub user_features: *mut f32,
    pub user_feature_gradients: *mut f32,
    pub user_feature_momentum: *mut f32,
    pub user_biases: *mut f32,
    pub user_bias_gradients: *mut f32,
    pub user_bias_momentum: *mut f32,

    // Dimensions
    pub no_components: i32,
    pub num_item_features: i32,
    pub num_user_features: i32,

    // Hyperparameters
    pub adadelta: bool,
    pub learning_rate: f64,
    pub rho: f32,
    pub eps: f32,
    pub max_sampled: i32,

    // Regularization scales (thread-safe)
    pub item_scale: RwLock<f64>,
    pub user_scale: RwLock<f64>,
}

// Safety: Raw pointers are only used within parallel regions where
// we ensure non-overlapping access patterns. The RwLock protects scales.
unsafe impl Send for FastLightFMView {}
unsafe impl Sync for FastLightFMView {}

impl FastLightFMView {
    /// Get item feature value at [feature, component]
    #[inline]
    pub fn get_item_feature(&self, feature: i32, component: i32) -> f32 {
        unsafe {
            *self.item_features.offset((feature * self.no_components + component) as isize)
        }
    }

    /// Set item feature value at [feature, component]
    #[inline]
    pub fn set_item_feature(&self, feature: i32, component: i32, value: f32) {
        unsafe {
            *self.item_features.offset((feature * self.no_components + component) as isize) = value;
        }
    }

    /// Get item feature gradient at [feature, component]
    #[inline]
    pub fn get_item_feature_gradient(&self, feature: i32, component: i32) -> f32 {
        unsafe {
            *self.item_feature_gradients.offset((feature * self.no_components + component) as isize)
        }
    }

    /// Set item feature gradient at [feature, component]
    #[inline]
    pub fn set_item_feature_gradient(&self, feature: i32, component: i32, value: f32) {
        unsafe {
            *self.item_feature_gradients.offset((feature * self.no_components + component) as isize) = value;
        }
    }

    /// Get item feature momentum at [feature, component]
    #[inline]
    pub fn get_item_feature_momentum(&self, feature: i32, component: i32) -> f32 {
        unsafe {
            *self.item_feature_momentum.offset((feature * self.no_components + component) as isize)
        }
    }

    /// Set item feature momentum at [feature, component]
    #[inline]
    pub fn set_item_feature_momentum(&self, feature: i32, component: i32, value: f32) {
        unsafe {
            *self.item_feature_momentum.offset((feature * self.no_components + component) as isize) = value;
        }
    }

    /// Get user feature value at [feature, component]
    #[inline]
    pub fn get_user_feature(&self, feature: i32, component: i32) -> f32 {
        unsafe {
            *self.user_features.offset((feature * self.no_components + component) as isize)
        }
    }

    /// Set user feature value at [feature, component]
    #[inline]
    pub fn set_user_feature(&self, feature: i32, component: i32, value: f32) {
        unsafe {
            *self.user_features.offset((feature * self.no_components + component) as isize) = value;
        }
    }

    /// Get user feature gradient at [feature, component]
    #[inline]
    pub fn get_user_feature_gradient(&self, feature: i32, component: i32) -> f32 {
        unsafe {
            *self.user_feature_gradients.offset((feature * self.no_components + component) as isize)
        }
    }

    /// Set user feature gradient at [feature, component]
    #[inline]
    pub fn set_user_feature_gradient(&self, feature: i32, component: i32, value: f32) {
        unsafe {
            *self.user_feature_gradients.offset((feature * self.no_components + component) as isize) = value;
        }
    }

    /// Get user feature momentum at [feature, component]
    #[inline]
    pub fn get_user_feature_momentum(&self, feature: i32, component: i32) -> f32 {
        unsafe {
            *self.user_feature_momentum.offset((feature * self.no_components + component) as isize)
        }
    }

    /// Set user feature momentum at [feature, component]
    #[inline]
    pub fn set_user_feature_momentum(&self, feature: i32, component: i32, value: f32) {
        unsafe {
            *self.user_feature_momentum.offset((feature * self.no_components + component) as isize) = value;
        }
    }

    /// Get item bias
    #[inline]
    pub fn get_item_bias(&self, feature: i32) -> f32 {
        unsafe { *self.item_biases.offset(feature as isize) }
    }

    /// Set item bias
    #[inline]
    pub fn set_item_bias(&self, feature: i32, value: f32) {
        unsafe { *self.item_biases.offset(feature as isize) = value }
    }

    /// Get item bias gradient
    #[inline]
    pub fn get_item_bias_gradient(&self, feature: i32) -> f32 {
        unsafe { *self.item_bias_gradients.offset(feature as isize) }
    }

    /// Set item bias gradient
    #[inline]
    pub fn set_item_bias_gradient(&self, feature: i32, value: f32) {
        unsafe { *self.item_bias_gradients.offset(feature as isize) = value }
    }

    /// Get item bias momentum
    #[inline]
    pub fn get_item_bias_momentum(&self, feature: i32) -> f32 {
        unsafe { *self.item_bias_momentum.offset(feature as isize) }
    }

    /// Set item bias momentum
    #[inline]
    pub fn set_item_bias_momentum(&self, feature: i32, value: f32) {
        unsafe { *self.item_bias_momentum.offset(feature as isize) = value }
    }

    /// Get user bias
    #[inline]
    pub fn get_user_bias(&self, feature: i32) -> f32 {
        unsafe { *self.user_biases.offset(feature as isize) }
    }

    /// Set user bias
    #[inline]
    pub fn set_user_bias(&self, feature: i32, value: f32) {
        unsafe { *self.user_biases.offset(feature as isize) = value }
    }

    /// Get user bias gradient
    #[inline]
    pub fn get_user_bias_gradient(&self, feature: i32) -> f32 {
        unsafe { *self.user_bias_gradients.offset(feature as isize) }
    }

    /// Set user bias gradient
    #[inline]
    pub fn set_user_bias_gradient(&self, feature: i32, value: f32) {
        unsafe { *self.user_bias_gradients.offset(feature as isize) = value }
    }

    /// Get user bias momentum
    #[inline]
    pub fn get_user_bias_momentum(&self, feature: i32) -> f32 {
        unsafe { *self.user_bias_momentum.offset(feature as isize) }
    }

    /// Set user bias momentum
    #[inline]
    pub fn set_user_bias_momentum(&self, feature: i32, value: f32) {
        unsafe { *self.user_bias_momentum.offset(feature as isize) = value }
    }

    /// Get item scale (thread-safe read)
    #[inline]
    pub fn get_item_scale(&self) -> f64 {
        *self.item_scale.read()
    }

    /// Get user scale (thread-safe read)
    #[inline]
    pub fn get_user_scale(&self) -> f64 {
        *self.user_scale.read()
    }

    /// Multiply item scale (thread-safe)
    #[inline]
    pub fn multiply_item_scale(&self, factor: f64) {
        let mut scale = self.item_scale.write();
        *scale *= factor;
    }

    /// Multiply user scale (thread-safe)
    #[inline]
    pub fn multiply_user_scale(&self, factor: f64) {
        let mut scale = self.user_scale.write();
        *scale *= factor;
    }

    /// Reset item scale to 1.0 (thread-safe)
    #[inline]
    pub fn reset_item_scale(&self) {
        let mut scale = self.item_scale.write();
        *scale = 1.0;
    }

    /// Reset user scale to 1.0 (thread-safe)
    #[inline]
    pub fn reset_user_scale(&self) {
        let mut scale = self.user_scale.write();
        *scale = 1.0;
    }
}

/// Python-facing FastLightFM class that holds references to numpy arrays
#[pyclass]
pub struct FastLightFM {
    // Pointers to numpy arrays
    item_features_ptr: *mut f32,
    item_feature_gradients_ptr: *mut f32,
    item_feature_momentum_ptr: *mut f32,
    item_biases_ptr: *mut f32,
    item_bias_gradients_ptr: *mut f32,
    item_bias_momentum_ptr: *mut f32,

    user_features_ptr: *mut f32,
    user_feature_gradients_ptr: *mut f32,
    user_feature_momentum_ptr: *mut f32,
    user_biases_ptr: *mut f32,
    user_bias_gradients_ptr: *mut f32,
    user_bias_momentum_ptr: *mut f32,

    // Dimensions
    #[pyo3(get)]
    no_components: i32,
    num_item_features: i32,
    num_user_features: i32,

    // Hyperparameters
    #[pyo3(get)]
    adadelta: i32,
    #[pyo3(get)]
    learning_rate: f32,
    #[pyo3(get)]
    rho: f32,
    #[pyo3(get)]
    eps: f32,
    #[pyo3(get)]
    max_sampled: i32,

    // Regularization scales
    item_scale: RwLock<f64>,
    user_scale: RwLock<f64>,
}

// Safety: The pointers reference numpy arrays managed by Python.
// Thread safety is ensured by the RwLock on scales.
unsafe impl Send for FastLightFM {}
unsafe impl Sync for FastLightFM {}

#[pymethods]
impl FastLightFM {
    #[new]
    #[pyo3(signature = (item_features, item_feature_gradients, item_feature_momentum, item_biases, item_bias_gradients, item_bias_momentum, user_features, user_feature_gradients, user_feature_momentum, user_biases, user_bias_gradients, user_bias_momentum, no_components, adadelta, learning_rate, rho, epsilon, max_sampled))]
    pub fn new(
        mut item_features: PyReadwriteArray2<f32>,
        mut item_feature_gradients: PyReadwriteArray2<f32>,
        mut item_feature_momentum: PyReadwriteArray2<f32>,
        mut item_biases: PyReadwriteArray1<f32>,
        mut item_bias_gradients: PyReadwriteArray1<f32>,
        mut item_bias_momentum: PyReadwriteArray1<f32>,
        mut user_features: PyReadwriteArray2<f32>,
        mut user_feature_gradients: PyReadwriteArray2<f32>,
        mut user_feature_momentum: PyReadwriteArray2<f32>,
        mut user_biases: PyReadwriteArray1<f32>,
        mut user_bias_gradients: PyReadwriteArray1<f32>,
        mut user_bias_momentum: PyReadwriteArray1<f32>,
        no_components: i32,
        adadelta: i32,
        learning_rate: f32,
        rho: f32,
        epsilon: f32,
        max_sampled: i32,
    ) -> Self {
        let num_item_features = item_features.shape()[0] as i32;
        let num_user_features = user_features.shape()[0] as i32;

        Self {
            item_features_ptr: item_features.as_array_mut().as_mut_ptr(),
            item_feature_gradients_ptr: item_feature_gradients.as_array_mut().as_mut_ptr(),
            item_feature_momentum_ptr: item_feature_momentum.as_array_mut().as_mut_ptr(),
            item_biases_ptr: item_biases.as_array_mut().as_mut_ptr(),
            item_bias_gradients_ptr: item_bias_gradients.as_array_mut().as_mut_ptr(),
            item_bias_momentum_ptr: item_bias_momentum.as_array_mut().as_mut_ptr(),

            user_features_ptr: user_features.as_array_mut().as_mut_ptr(),
            user_feature_gradients_ptr: user_feature_gradients.as_array_mut().as_mut_ptr(),
            user_feature_momentum_ptr: user_feature_momentum.as_array_mut().as_mut_ptr(),
            user_biases_ptr: user_biases.as_array_mut().as_mut_ptr(),
            user_bias_gradients_ptr: user_bias_gradients.as_array_mut().as_mut_ptr(),
            user_bias_momentum_ptr: user_bias_momentum.as_array_mut().as_mut_ptr(),

            no_components,
            num_item_features,
            num_user_features,

            adadelta,
            learning_rate,
            rho,
            eps: epsilon,
            max_sampled,

            item_scale: RwLock::new(1.0),
            user_scale: RwLock::new(1.0),
        }
    }
}

impl FastLightFM {
    /// Create a view for use in compute-heavy functions
    pub fn view(&self) -> FastLightFMView {
        FastLightFMView {
            item_features: self.item_features_ptr,
            item_feature_gradients: self.item_feature_gradients_ptr,
            item_feature_momentum: self.item_feature_momentum_ptr,
            item_biases: self.item_biases_ptr,
            item_bias_gradients: self.item_bias_gradients_ptr,
            item_bias_momentum: self.item_bias_momentum_ptr,

            user_features: self.user_features_ptr,
            user_feature_gradients: self.user_feature_gradients_ptr,
            user_feature_momentum: self.user_feature_momentum_ptr,
            user_biases: self.user_biases_ptr,
            user_bias_gradients: self.user_bias_gradients_ptr,
            user_bias_momentum: self.user_bias_momentum_ptr,

            no_components: self.no_components,
            num_item_features: self.num_item_features,
            num_user_features: self.num_user_features,

            adadelta: self.adadelta != 0,
            learning_rate: self.learning_rate as f64,
            rho: self.rho,
            eps: self.eps,
            max_sampled: self.max_sampled,

            item_scale: RwLock::new(*self.item_scale.read()),
            user_scale: RwLock::new(*self.user_scale.read()),
        }
    }

    /// Sync scales back from view after training
    pub fn sync_scales(&self, view: &FastLightFMView) {
        let mut item_scale = self.item_scale.write();
        *item_scale = view.get_item_scale();

        let mut user_scale = self.user_scale.write();
        *user_scale = view.get_user_scale();
    }
}
