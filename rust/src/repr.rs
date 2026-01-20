//! Representation computation helpers

use crate::csr_matrix::CSRMatrixView;
use crate::model::FastLightFMView;
use wide::f32x4;

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
    let n = lightfm.no_components as usize;

    // Zero out the representation
    // Using fill is often optimized by memset
    repr.fill(0.0);

    for i in start_index..stop_index {
        let feature = features.index(i);
        let feature_weight = features.value(i) * scale;
        
        let embedding_ptr = if is_user {
            unsafe { lightfm.user_features.offset((feature * lightfm.no_components) as isize) }
        } else {
            unsafe { lightfm.item_features.offset((feature * lightfm.no_components) as isize) }
        };

        // SIMD-optimized accumulation
        // Check if no_components is a multiple of 8 for unrolled loop
        let mut j = 0;
        if n % 8 == 0 {
            while j < n {
                unsafe {
                    // Load 8 values (2 x f32x4)
                    let feat_vals0 = f32x4::from(*(embedding_ptr.add(j) as *const [f32; 4]));
                    let feat_vals1 = f32x4::from(*(embedding_ptr.add(j + 4) as *const [f32; 4]));
                    
                    let curr0 = f32x4::from(*(repr.as_ptr().add(j) as *const [f32; 4]));
                    let curr1 = f32x4::from(*(repr.as_ptr().add(j + 4) as *const [f32; 4]));
                    
                    let weight_vec = f32x4::splat(feature_weight);
                    
                    let next0 = curr0 + (feat_vals0 * weight_vec);
                    let next1 = curr1 + (feat_vals1 * weight_vec);
                    
                    // Store back
                    *(repr.as_mut_ptr().add(j) as *mut [f32; 4]) = next0.to_array();
                    *(repr.as_mut_ptr().add(j + 4) as *mut [f32; 4]) = next1.to_array();
                }
                j += 8;
            }
        } else if n % 4 == 0 {
            while j < n {
                unsafe {
                    let feat_vals = f32x4::from(*(embedding_ptr.add(j) as *const [f32; 4]));
                    let curr = f32x4::from(*(repr.as_ptr().add(j) as *const [f32; 4]));
                    let weight_vec = f32x4::splat(feature_weight);
                    let next = curr + (feat_vals * weight_vec);
                    *(repr.as_mut_ptr().add(j) as *mut [f32; 4]) = next.to_array();
                }
                j += 4;
            }
        } else {
            // Fallback for non-divisible sizes
            for k in 0..lightfm.no_components {
                let embedding = unsafe { *embedding_ptr.offset(k as isize) };
                repr[k as usize] += feature_weight * embedding;
            }
        }

        // Add weighted bias
        let bias = if is_user {
            lightfm.get_user_bias(feature)
        } else {
            lightfm.get_item_bias(feature)
        };
        repr[n] += feature_weight * bias;
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
    let mut i = 0;
    if n % 8 == 0 {
        let mut sum0 = f32x4::splat(0.0);
        let mut sum1 = f32x4::splat(0.0);
        
        unsafe {
            let u_ptr = user_repr.as_ptr();
            let i_ptr = item_repr.as_ptr();
            
            while i < n {
                let u0 = f32x4::from(*(u_ptr.add(i) as *const [f32; 4]));
                let u1 = f32x4::from(*(u_ptr.add(i + 4) as *const [f32; 4]));
                let v0 = f32x4::from(*(i_ptr.add(i) as *const [f32; 4]));
                let v1 = f32x4::from(*(i_ptr.add(i + 4) as *const [f32; 4]));
                
                sum0 += u0 * v0;
                sum1 += u1 * v1;
                i += 8;
            }
        }
        result += sum0.reduce_add() + sum1.reduce_add();
    } else if n % 4 == 0 {
        let mut sum = f32x4::splat(0.0);
        unsafe {
            let u_ptr = user_repr.as_ptr();
            let i_ptr = item_repr.as_ptr();
            while i < n {
                let u = f32x4::from(*(u_ptr.add(i) as *const [f32; 4]));
                let v = f32x4::from(*(i_ptr.add(i) as *const [f32; 4]));
                sum += u * v;
                i += 4;
            }
        }
        result += sum.reduce_add();
    } else {
        // Fallback
        result += user_repr[0..n]
            .iter()
            .zip(item_repr[0..n].iter())
            .map(|(u, i)| u * i)
            .sum::<f32>();
    }

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
