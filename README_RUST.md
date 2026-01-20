# LightFM Rust Backend Development

This project uses a Rust backend for high-performance computation, managed by `maturin`.

## Prerequisites

- Python 3.9+
- Rust (latest stable)
- `uv` (recommended) or `pip`

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/lyst/lightfm.git
    cd lightfm
    ```

2.  **Create a virtual environment:**
    ```bash
    uv venv
    source .venv/bin/activate
    ```

3.  **Install development dependencies:**
    ```bash
    uv pip install maturin pytest numpy scipy requests scikit-learn
    ```

## Building

To build the Rust extension and install it in editable mode:

**Debug Build (Fast compilation, slower execution):**
```bash
uv run maturin develop
```

**Release Build (Optimized for performance):**
```bash
uv run maturin develop --release
```

*Note: Always use `--release` when benchmarking or deploying.*

## Testing

Run the Python test suite:
```bash
uv run pytest tests
```

## Benchmarking

To run the thread scaling benchmark:
```bash
# Ensure release build first
uv run maturin develop --release

# Run benchmark
uv run benchmark/bench_threads.py
```
