//! CSR Matrix wrapper for zero-copy access to scipy sparse matrices

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

use crate::utils::binary_search;

/// A wrapper around CSR matrix data that borrows from numpy arrays.
/// This struct holds raw pointers for performance-critical inner loops.
pub struct CSRMatrixView {
    /// Column indices for each non-zero element
    pub indices: *const i32,
    /// Pointer to start of each row in indices/data
    pub indptr: *const i32,
    /// Non-zero values
    pub data: *const f32,
    /// Number of rows
    pub rows: i32,
    /// Number of columns
    pub cols: i32,
    /// Number of non-zero elements
    pub nnz: i32,
}

// Safety: CSRMatrixView only reads data, multiple threads can read simultaneously
unsafe impl Send for CSRMatrixView {}
unsafe impl Sync for CSRMatrixView {}

impl CSRMatrixView {
    /// Get the start index of row data in the indices/data arrays
    #[inline]
    pub fn get_row_start(&self, row: i32) -> i32 {
        unsafe { *self.indptr.offset(row as isize) }
    }

    /// Get the end index of row data in the indices/data arrays
    #[inline]
    pub fn get_row_end(&self, row: i32) -> i32 {
        unsafe { *self.indptr.offset((row + 1) as isize) }
    }

    /// Get column index at position i
    #[inline]
    pub fn index(&self, i: i32) -> i32 {
        unsafe { *self.indices.offset(i as isize) }
    }

    /// Get value at position i
    #[inline]
    pub fn value(&self, i: i32) -> f32 {
        unsafe { *self.data.offset(i as isize) }
    }

    /// Check if an item is in the positives for a user (binary search in sorted row)
    #[inline]
    pub fn in_positives(&self, item_id: i32, user_id: i32) -> bool {
        let start = self.get_row_start(user_id) as usize;
        let end = self.get_row_end(user_id) as usize;
        if start >= end {
            return false;
        }

        // Create a slice view of the indices for this row
        let row_indices = unsafe {
            std::slice::from_raw_parts(self.indices.add(start), end - start)
        };

        binary_search(row_indices, item_id)
    }
}

/// Python-facing CSRMatrix class that owns references to numpy arrays
#[pyclass]
pub struct CSRMatrix {
    #[pyo3(get)]
    pub rows: i32,
    #[pyo3(get)]
    pub cols: i32,
    #[pyo3(get)]
    pub nnz: i32,

    // Store raw pointers to the underlying array data
    indices_ptr: *const i32,
    indptr_ptr: *const i32,
    data_ptr: *const f32,
}

// Safety: CSRMatrix only reads data, safe for parallel access
unsafe impl Send for CSRMatrix {}
unsafe impl Sync for CSRMatrix {}

#[pymethods]
impl CSRMatrix {
    #[new]
    pub fn new(csr_matrix: &Bound<'_, PyAny>) -> PyResult<Self> {
        let indices: PyReadonlyArray1<i32> = csr_matrix.getattr("indices")?.extract()?;
        let indptr: PyReadonlyArray1<i32> = csr_matrix.getattr("indptr")?.extract()?;
        let data: PyReadonlyArray1<f32> = csr_matrix.getattr("data")?.extract()?;

        let shape: (i32, i32) = csr_matrix.getattr("shape")?.extract()?;

        // Get raw pointers from the numpy arrays
        let indices_slice = indices.as_slice()?;
        let indptr_slice = indptr.as_slice()?;
        let data_slice = data.as_slice()?;

        let nnz = data_slice.len() as i32;

        let indices_ptr = indices_slice.as_ptr();
        let indptr_ptr = indptr_slice.as_ptr();
        let data_ptr = data_slice.as_ptr();

        Ok(Self {
            rows: shape.0,
            cols: shape.1,
            nnz,
            indices_ptr,
            indptr_ptr,
            data_ptr,
        })
    }
}

impl CSRMatrix {
    /// Create a view for use in compute-heavy functions
    pub fn view(&self) -> CSRMatrixView {
        CSRMatrixView {
            indices: self.indices_ptr,
            indptr: self.indptr_ptr,
            data: self.data_ptr,
            rows: self.rows,
            cols: self.cols,
            nnz: self.nnz,
        }
    }
}

/// Test helper function exposed to Python
pub fn test_in_positives(row: i32, col: i32, mat: &CSRMatrix) -> bool {
    mat.view().in_positives(col, row)
}
