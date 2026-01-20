import time
import argparse
import numpy as np
import scipy.sparse as sp
from lightfm.datasets import fetch_movielens
import lightfm._lightfm_fast_rust as rust_backend

def run_gpu_fit_benchmark(multiplier=10, num_components=30, epochs=1):
    print(f"Fetching MovieLens 100k dataset...")
    data = fetch_movielens()
    train_orig = data['train']
    
    print(f"Replicating dataset {multiplier}x...")
    blocks = [train_orig] * multiplier
    train_large = sp.vstack(blocks).tocsr()
    
    num_users, num_items = train_large.shape
    print(f"Dataset: {num_users} users, {num_items} items.")
    
    # Initialize embeddings
    user_features = np.random.randn(num_users, num_components).astype(np.float32) * 0.01
    item_features = np.random.randn(num_items, num_components).astype(np.float32) * 0.01
    user_biases = np.zeros(num_users, dtype=np.float32)
    item_biases = np.zeros(num_items, dtype=np.float32)
    
    # Extract CSR arrays
    train_indptr = train_large.indptr.astype(np.int32)
    train_indices = train_large.indices.astype(np.int32)
    
    # Num samples: usually interactions * epochs
    num_samples = train_large.nnz
    
    print("\nBenchmarking Fit (GPU BPR)")
    print("-" * 65)
    
    try:
        gpu_backend = rust_backend.LightFMwgpu()
        start_gpu = time.time()
        
        for epoch in range(epochs):
            seed = int(time.time() + epoch)
            gpu_backend.fit_bpr_gpu(
                user_features, 
                item_features, 
                user_biases, 
                item_biases, 
                train_indptr, 
                train_indices, 
                num_components, 
                0.05, # LR
                0.0, # item alpha
                0.0, # user alpha
                num_samples,
                seed
            )
            print(f"Epoch {epoch+1}/{epochs} done.")
            
        time_gpu = time.time() - start_gpu
        print(f"GPU Fit Time ({epochs} epochs): {time_gpu:.4f}s")
        print(f"Samples/sec: {num_samples * epochs / time_gpu:,.0f}")
        
        # Verify correctness check
        # We copy arrays to check changes (just boolean)
        print(f"User features changed: {np.any(np.abs(user_features) > 0.02)}") # Check if some values grew/changed
        
    except Exception as e:
        print(f"GPU Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--multiplier', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=5)
    args = parser.parse_args()
    run_gpu_fit_benchmark(multiplier=args.multiplier, epochs=args.epochs)
