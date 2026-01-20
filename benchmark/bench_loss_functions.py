import time
import numpy as np
import argparse
from lightfm import LightFM
from lightfm.datasets import fetch_movielens

def run_benchmark(epochs=10, num_components=30):
    print("Fetching MovieLens 100k dataset...")
    data = fetch_movielens()
    train = data['train']
    
    losses = ['logistic', 'bpr', 'warp', 'warp-kos']
    results = {}

    print(f"\nBenchmarking Loss Functions (Epochs: {epochs}, Components: {num_components}, Threads: 4)")
    print("-" * 65)
    print(f"{ 'Loss Function':<15} | { 'Total Time (s)':<15} | { 'Time/Epoch (s)':<15}")
    print("-" * 65)

    for loss in losses:
        # k-OS requires k and n parameters, using defaults implicitly or setting reasonable ones if needed
        model = LightFM(no_components=num_components, loss=loss, learning_rate=0.05)
        
        start_time = time.time()
        model.fit(train, epochs=epochs, num_threads=4)
        total_time = time.time() - start_time
        
        time_per_epoch = total_time / epochs
        results[loss] = total_time
        
        print(f"{loss:<15} | {total_time:<15.4f} | {time_per_epoch:<15.4f}")

    print("-" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark LightFM loss functions on MovieLens.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--components', type=int, default=30, help='Number of latent components')
    args = parser.parse_args()

    run_benchmark(epochs=args.epochs, num_components=args.components)
