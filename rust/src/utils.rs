//! Utility functions for LightFM Rust implementation

use rand::Rng;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

/// Thread-local RNG wrapper for reproducible randomness
pub struct ThreadRng {
    rng: Xoshiro256PlusPlus,
}

impl ThreadRng {
    /// Create a new RNG seeded with the given value
    pub fn new(seed: u64) -> Self {
        Self {
            rng: Xoshiro256PlusPlus::seed_from_u64(seed),
        }
    }

    /// Generate a random integer in range [0, max)
    #[inline]
    pub fn randint(&mut self, max: i32) -> i32 {
        self.rng.gen_range(0..max)
    }

    /// Generate a random integer in range [min, max)
    #[inline]
    pub fn sample_range(&mut self, min: i32, max: i32) -> i32 {
        self.rng.gen_range(min..max)
    }
}

/// Binary search to check if a value exists in a sorted slice
#[inline]
pub fn binary_search(arr: &[i32], target: i32) -> bool {
    arr.binary_search(&target).is_ok()
}

/// Compute sigmoid function
#[inline]
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Maximum regularization scale before full regularization is applied
pub const MAX_REG_SCALE: f64 = 1_000_000.0;

/// Maximum loss value for gradient clipping
pub const MAX_LOSS: f64 = 10.0;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-10);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn test_binary_search() {
        let arr = vec![1, 3, 5, 7, 9];
        assert!(binary_search(&arr, 5));
        assert!(!binary_search(&arr, 4));
    }

    #[test]
    fn test_rng() {
        let mut rng = ThreadRng::new(42);
        let val = rng.randint(100);
        assert!(val >= 0 && val < 100);
    }
}
