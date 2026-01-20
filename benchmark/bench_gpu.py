import time
import argparse
import numpy as np
import scipy.sparse as sp
from lightfm import LightFM
from lightfm.datasets import fetch_movielens
import lightfm._lightfm_fast_rust as rust_backend

def run_gpu_benchmark(multiplier=50, num_components=30):
    print(f"Fetching MovieLens 100k dataset...")
    data = fetch_movielens()
    train_orig = data['train']
    
    # Replicate to create a larger workload
    print(f"Replicating dataset {multiplier}x...")
    blocks = [train_orig] * multiplier
    train_large = sp.vstack(blocks).tocsr()
    
    # Create random embeddings
    num_users, num_items = train_large.shape
    print(f"Dataset: {num_users} users, {num_items} items.")
    
    user_features = np.random.randn(num_users, num_components).astype(np.float32)
    item_features = np.random.randn(num_items, num_components).astype(np.float32)
    user_biases = np.random.randn(num_users).astype(np.float32)
    item_biases = np.random.randn(num_items).astype(np.float32)
    
    # Generate random pairs to predict
    num_predictions = 10_000_000
    print(f"Generating {num_predictions} random pairs to predict...")
    
    user_ids = np.random.randint(0, num_users, num_predictions, dtype=np.int32)
    item_ids = np.random.randint(0, num_items, num_predictions, dtype=np.int32)
    
    # Interleave for GPU kernel: [u1, i1, u2, i2...]
    pairs = np.stack((user_ids, item_ids), axis=1).flatten().astype(np.int32)

    print("\nBenchmarking Prediction (CPU vs GPU)")
    print("-" * 65)
    print(f"{'Device':<10} | {'Time (s)':<15} | {'Predictions/s':<20}")
    print("-" * 65)

    # CPU Benchmark
    # We use the internal Rust predict_lightfm function directly to be fair (skip Python overhead)
    # But wait, predict_lightfm takes CSR matrices.
    # Let's use the standard LightFM.predict logic but optimized.
    # For fair comparison, we'll implement a simple loop calling the rust backend or just use the model.
    # Actually, let's use the Rust extension directly if possible.
    
    start_cpu = time.time()
    # Mocking what LightFM.predict does internally but batched
    # LightFM doesn't have a batched predict for arbitrary pairs exposed easily in Python without loop.
    # So we'll use dot product in numpy as a proxy for "Optimized CPU"
    cpu_preds = np.dot(user_features[user_ids] * item_features[item_ids], np.ones(num_components)) + user_biases[user_ids] + item_biases[item_ids]
    time_cpu = time.time() - start_cpu
    
    print(f"{'CPU (NumPy)':<10} | {time_cpu:<15.4f} | {num_predictions/time_cpu:<20,.0f}")

    # GPU Benchmark
    try:
        gpu_backend = rust_backend.LightFMwgpu()
        start_gpu = time.time()
        gpu_preds = gpu_backend.predict(user_features, item_features, user_biases, item_biases, pairs, num_components)
        time_gpu = time.time() - start_gpu
        
        print(f"{'GPU (Metal)':<10} | {time_gpu:<15.4f} | {num_predictions/time_gpu:<20,.0f}")
        
        # Verify correctness
        error = np.mean(np.abs(cpu_preds - gpu_preds))
        print(f"\nMean Absolute Error: {error:.6f}")
        print(f"Speedup: {time_cpu / time_gpu:.2f}x")
        
    except Exception as e:
        print(f"GPU Failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--multiplier', type=int, default=50)
    args = parser.parse_args()
    run_gpu_benchmark(multiplier=args.multiplier)
