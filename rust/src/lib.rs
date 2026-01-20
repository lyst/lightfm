//! LightFM Rust backend with PyO3 bindings
//!
//! This module provides a Rust implementation of the LightFM recommender system's
//! core computation functions, replacing the Cython implementation for improved
//! performance and easier maintenance.

use pyo3::prelude::*;

mod csr_matrix;
mod loss;
mod model;
mod optim;
mod predict;
mod repr;
mod utils;
mod gpu;

use csr_matrix::{test_in_positives, CSRMatrix};
use loss::{fit_bpr, fit_logistic, fit_warp, fit_warp_kos};
use model::FastLightFM;
use predict::{calculate_auc_from_rank, predict_lightfm, predict_ranks};
use gpu::LightFMwgpu;

/// Test helper function exposed to Python
#[pyfunction]
#[pyo3(name = "__test_in_positives")]
fn py_test_in_positives(row: i32, col: i32, mat: &CSRMatrix) -> bool {
    test_in_positives(row, col, mat)
}

/// Python module definition
#[pymodule]
fn _lightfm_fast_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add classes
    m.add_class::<CSRMatrix>()?;
    m.add_class::<FastLightFM>()?;
    m.add_class::<LightFMwgpu>()?;

    // Add training functions
    m.add_function(wrap_pyfunction!(fit_logistic, m)?)?;
    m.add_function(wrap_pyfunction!(fit_bpr, m)?)?;
    m.add_function(wrap_pyfunction!(fit_warp, m)?)?;
    m.add_function(wrap_pyfunction!(fit_warp_kos, m)?)?;

    // Add prediction functions
    m.add_function(wrap_pyfunction!(predict_lightfm, m)?)?;
    m.add_function(wrap_pyfunction!(predict_ranks, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_auc_from_rank, m)?)?;

    // Add test helper
    m.add_function(wrap_pyfunction!(py_test_in_positives, m)?)?;

    Ok(())
}
