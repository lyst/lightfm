import time
import numpy as np
import argparse
import multiprocessing
from lightfm import LightFM
from lightfm.datasets import fetch_movielens

def run_benchmark(epochs=10, num_components=30):
    print("Fetching MovieLens 100k dataset...")
    data = fetch_movielens()
    train = data['train']
    
    # Test thread counts: 1, 2, 4, and up to physical cores if > 4
    max_cores = multiprocessing.cpu_count()
    thread_counts = sorted(list(set([1, 2, 4, 8, max_cores])))
    thread_counts = [t for t in thread_counts if t <= max_cores]
    
    # Ensure 1 is present
    if 1 not in thread_counts:
        thread_counts.insert(0, 1)

    print(f"\nBenchmarking Thread Scaling (Loss: WARP, Epochs: {epochs}, Components: {num_components})")
    print("-" * 65)
    print(f"{ 'Threads':<10} | {'Total Time (s)':<15} | {'Speedup':<10}")
    print("-" * 65)

    baseline_time = None

    for num_threads in thread_counts:
        model = LightFM(no_components=num_components, loss='warp', learning_rate=0.05)
        
        start_time = time.time()
        model.fit(train, epochs=epochs, num_threads=num_threads)
        total_time = time.time() - start_time
        
        if num_threads == 1:
            baseline_time = total_time
            speedup = 1.0
        else:
            speedup = baseline_time / total_time
        
        print(f"{num_threads:<10} | {total_time:<15.4f} | {speedup:<10.2f}x")

    print("-" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark LightFM thread scaling.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--components', type=int, default=30, help='Number of latent components')
    args = parser.parse_args()

    run_benchmark(epochs=args.epochs, num_components=args.components)
