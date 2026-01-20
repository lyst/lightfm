import time
import argparse
import numpy as np
import scipy.sparse as sp
from lightfm import LightFM
from lightfm.datasets import fetch_movielens

def run_large_benchmark(multiplier=10, epochs=10, num_components=30, threads=8):
    print(f"Fetching MovieLens 100k dataset...")
    data = fetch_movielens()
    train_orig = data['train']
    
    print(f"Original dataset shape: {train_orig.shape} with {train_orig.nnz} interactions.")
    
    # Replicate the dataset vertically (simulating N times more users)
    print(f"Replicating dataset {multiplier}x vertically...")
    blocks = [train_orig] * multiplier
    train_large = sp.vstack(blocks).tocsr()
    
    print(f"Large dataset shape: {train_large.shape} with {train_large.nnz} interactions.")
    
    print(f"\nBenchmarking LightFM on {multiplier}x Dataset (Loss: WARP, Epochs: {epochs}, Components: {num_components})")
    print("-" * 65)
    print(f"{ 'Threads':<10} | {'Total Time (s)':<15} | {'Interactions/s':<15}")
    print("-" * 65)

    # Benchmark with 1 thread for baseline
    print("Running with 1 thread (Baseline)...")
    model_single = LightFM(no_components=num_components, loss='warp', learning_rate=0.05)
    start_time_single = time.time()
    model_single.fit(train_large, epochs=epochs, num_threads=1)
    time_single = time.time() - start_time_single
    
    ips_single = (train_large.nnz * epochs) / time_single
    print(f"{1:<10} | {time_single:<15.4f} | {ips_single:,.0f}")

    # Benchmark with N threads
    print(f"Running with {threads} threads...")
    model_multi = LightFM(no_components=num_components, loss='warp', learning_rate=0.05)
    start_time_multi = time.time()
    model_multi.fit(train_large, epochs=epochs, num_threads=threads)
    time_multi = time.time() - start_time_multi
    
    ips_multi = (train_large.nnz * epochs) / time_multi
    print(f"{threads:<10} | {time_multi:<15.4f} | {ips_multi:,.0f}")
    
    print("-" * 65)
    print(f"Speedup: {time_single / time_multi:.2f}x")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark LightFM on a large replicated dataset.')
    parser.add_argument('--multiplier', type=int, default=50, help='Dataset replication factor (default: 50)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--components', type=int, default=30, help='Number of components')
    parser.add_argument('--threads', type=int, default=8, help='Number of threads for multi-threaded test')
    args = parser.parse_args()

    run_large_benchmark(multiplier=args.multiplier, epochs=args.epochs, num_components=args.components, threads=args.threads)
