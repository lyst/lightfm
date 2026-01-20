import time
import argparse
import numpy as np
import scipy.sparse as sp
from lightfm import LightFM
from lightfm.datasets import fetch_movielens
from lightfm.evaluation import precision_at_k
import lightfm._lightfm_fast_rust as rust_backend

def run_gpu_fit_benchmark(multiplier=10, num_components=30, epochs=1):
    print(f"Fetching MovieLens 100k dataset...")
    data = fetch_movielens()
    train_orig = data['train']
    test_data = data['test']
    
    print(f"Replicating dataset {multiplier}x...")
    blocks = [train_orig] * multiplier
    train_large = sp.vstack(blocks).tocsr()
    
    # Prepare test data (padded)
    extra_rows = train_large.shape[0] - test_data.shape[0]
    empty_block = sp.csr_matrix((extra_rows, test_data.shape[1]), dtype=test_data.dtype)
    test_large = sp.vstack([test_data, empty_block]).tocsr()
    
    num_users, num_items = train_large.shape
    print(f"Dataset: {num_users} users, {num_items} items.")
    
    # -------------------------------------------------------------------------
    # CPU Benchmark
    # -------------------------------------------------------------------------
    print("\nBenchmarking Fit (CPU BPR - 8 threads)")
    print("-" * 65)
    
    model_cpu = LightFM(no_components=num_components, loss='bpr', learning_rate=0.05)
    
    start_cpu = time.time()
    model_cpu.fit(train_large, epochs=epochs, num_threads=8)
    time_cpu = time.time() - start_cpu
    
    print(f"CPU Fit Time: {time_cpu:.4f}s")
    
    # Evaluate CPU
    print("Evaluating CPU model...")
    p_k_cpu = precision_at_k(model_cpu, test_large, k=10, num_threads=8)[:test_data.shape[0]].mean()
    print(f"CPU Precision@10: {p_k_cpu:.4f}")

    # -------------------------------------------------------------------------
    # GPU Benchmark
    # -------------------------------------------------------------------------
    print("\nBenchmarking Fit (GPU BPR)")
    print("-" * 65)
    
    # Initialize arrays for GPU (similar initialization to LightFM)
    # LightFM uses (rand - 0.5) / no_components
    user_features = (np.random.rand(num_users, num_components).astype(np.float32) - 0.5) / num_components
    item_features = (np.random.rand(num_items, num_components).astype(np.float32) - 0.5) / num_components
    user_biases = np.zeros(num_users, dtype=np.float32)
    item_biases = np.zeros(num_items, dtype=np.float32)
    
    train_indptr = train_large.indptr.astype(np.int32)
    train_indices = train_large.indices.astype(np.int32)
    
    num_samples = train_large.nnz
    
    try:
        gpu_backend = rust_backend.LightFMwgpu()
        start_gpu = time.time()
        
        # GPU training loop
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
                0.01, # LR (Lower for GPU due to race conditions)
                0.0, # item alpha
                0.0, # user alpha
                num_samples,
                seed
            )
            
        time_gpu = time.time() - start_gpu
        print(f"GPU Fit Time: {time_gpu:.4f}s")
        print(f"Samples/sec: {num_samples * epochs / time_gpu:,.0f}")
        
        # Inject into LightFM model for evaluation
        model_gpu = LightFM(no_components=num_components, loss='bpr')
        # We need to set internal state to trick LightFM into thinking it's fitted
        model_gpu.user_embeddings = user_features
        model_gpu.item_embeddings = item_features
        model_gpu.user_biases = user_biases
        model_gpu.item_biases = item_biases
        
        # We also need to set gradients/momentum to satisfy _check_initialized
        model_gpu.user_embedding_gradients = np.zeros_like(user_features)
        model_gpu.user_embedding_momentum = np.zeros_like(user_features)
        model_gpu.user_bias_gradients = np.zeros_like(user_biases)
        model_gpu.user_bias_momentum = np.zeros_like(user_biases)
        
        model_gpu.item_embedding_gradients = np.zeros_like(item_features)
        model_gpu.item_embedding_momentum = np.zeros_like(item_features)
        model_gpu.item_bias_gradients = np.zeros_like(item_biases)
        model_gpu.item_bias_momentum = np.zeros_like(item_biases)
        
        print("Evaluating GPU model...")
        p_k_gpu = precision_at_k(model_gpu, test_large, k=10, num_threads=8)[:test_data.shape[0]].mean()
        print(f"GPU Precision@10: {p_k_gpu:.4f}")
        
        print("-" * 65)
        print(f"Speedup: {time_cpu / time_gpu:.2f}x")
        print(f"Quality Ratio (GPU/CPU): {p_k_gpu / p_k_cpu:.2f}")
        
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
