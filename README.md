# Bellman–Ford HPC: OpenMP + CUDA (Modular Skeleton)

This project implements the Bellman–Ford shortest path algorithm in four variants, demonstrating High-Performance Computing (HPC) techniques with OpenMP, CUDA, and a hybrid CPU–GPU design.

Variants:

- `BF_serial` — single-threaded baseline
- `BF_openmp` — CPU parallel (OpenMP), ping-pong arrays
- `BF_cuda` — GPU parallel (CUDA), edge-parallel + `atomicMin`
- `BF_hybrid` — CPU+GPU split (partition by destination; tune ratio)

## Build

Requires GCC with OpenMP and the NVIDIA CUDA Toolkit.

```bash
make
```

Builds: BF_serial, BF_openmp, BF_cuda, BF_hybrid.

## Run

Examples (graph with 5000 vertices, edge weights in [-30,30]):

# Serial

```bash
./BF_serial  5000 -30 30 0.001
OMP_NUM_THREADS=8
```

# OpenMP (multi-core CPU, 8 threads)

```bash
./BF_openmp 5000 -30 30 0.001 8
```

# CUDA (GPU acceleration)

```bash
./BF_cuda     5000 -30 30 0.001
OMP_NUM_THREADS=8
```

# Hybrid (CPU+GPU, 60% edges on GPU)

```bash
./BF_hybrid 5000 -30 30 0.6 0.001 8
```

Graphs auto-save as `data/graph_<V>_<max>_<min>.txt`.  
Outputs: `serial/openmp/cuda/hybrid_output__V_max_min.txt`.

## Validate

Compare each variant’s output against the serial baseline with RMSE:

```bash
python3 scripts/compare_rmse.py serial_output__5000_30_-30.txt openmp_output__5000_30_-30.txt
python3 scripts/compare_rmse.py serial_output__5000_30_-30.txt cuda_output__5000_30_-30.txt
python3 scripts/compare_rmse.py serial_output__5000_30_-30.txt hybrid_output__5000_30_-30.txt
```

RMSE near 0.0 confirms equivalence.

## Notes & Insights

OpenMP

- Uses two arrays (ping-pong) to avoid heavy locking; only minimal critical region for safe updates.
- Experiment with schedule(static) vs. schedule(guided) for load balance.

CUDA

- Launches one thread per edge; uses atomicMin to update dist[v].
- Tune block size (128–512) and grid size for your GPU.
- Each iteration checks a global “updated” flag for early stopping.

Hybrid

- Splits work between CPU (OpenMP) and GPU (CUDA).
- Partition ratio configurable (e.g., 0.6 = 60% edges on GPU).
- Overlaps CPU relaxation with GPU kernel; merges results each iteration.

Algorithm semantics

- Stops early when no updates occur in an iteration.

- If updates occur in the V-th iteration, reports a negative weight cycle.
