<a href="https://colab.research.google.com/github/UchihaIthachi/bellman-ford-hpc-openmp-cuda/blob/main/SSSP-APSP-HPC-Analysis.ipynb" target="_blank" rel="noopener noreferrer">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

# SSSP/APSP HPC: A Comparative Analysis

This project provides and analyzes different implementations of single-source shortest path (SSSP) and all-pairs shortest path (APSP) algorithms, showcasing a range of high-performance computing (HPC) techniques.

## Algorithms Implemented

- **SSSP (Single-Source Shortest Path)**
  - **Dijkstra:** For graphs with non-negative edge weights.
  - **Bellman-Ford:** For graphs that may include negative edge weights; also detects negative cycles.
- **APSP (All-Pairs Shortest Path)**
  - **Floyd-Warshall:** A classic dynamic programming algorithm for dense graphs.
  - **Johnson's Algorithm:** For sparse graphs; uses Bellman-Ford for re-weighting and then runs Dijkstra from each vertex.

## Variants

Each algorithm is implemented in several variants where applicable:

- **Serial:** A standard, single-threaded baseline implementation.
- **OpenMP:** A parallel implementation using OpenMP for multi-core CPUs.
- **CUDA:** A massively parallel implementation using CUDA for NVIDIA GPUs.
- **Hybrid:** A combined CPU+GPU approach using both OpenMP and CUDA.

## Quickstart: Performance Analysis with Jupyter

The best way to explore this project is through the interactive Jupyter Notebook. It will guide you through building the code, running benchmarks, and visualizing the performance results for all algorithms.

**>> [Open the Performance Analysis Notebook](SSSP-APSP-HPC-Analysis.ipynb) <<**

## Manual Build and Run

If you prefer to work from the command line, you can build and run the executables manually.

### Build

You will need GCC (with OpenMP support) to build the CPU-based variants. For the CUDA and hybrid variants, you will also need the NVIDIA CUDA Toolkit (`nvcc`).

- **Build all available targets:**

  ```bash
  make all
  ```

  This will compile all sources and place the executables in the `bin/` directory. If `nvcc` is not found, CUDA targets will be skipped automatically.

- **Clean the build artifacts:**
  ```bash
  make clean
  ```

### Run

The executables will automatically generate graph data files in the `data/` directory if they don't already exist. Output files will be placed in the project root.

Here are some examples of how to run each variant.

#### SSSP Algorithms

- **BF_serial / dijkstra_serial:**
  ```bash
  # Usage: <exe> <V> <min_w> <max_w> [density]
  ./bin/BF_serial 5000 -30 30 0.001
  ```
- **BF_openmp / dijkstra_openmp:**
  ```bash
  # Usage: <exe> <V> <min_w> <max_w> [density] [threads]
  ./bin/BF_openmp 5000 -30 30 0.001 8
  ```
- **BF_cuda / dijkstra_cuda:**
  ```bash
  # Usage: <exe> <V> <min_w> <max_w> [density]
  ./bin/BF_cuda 5000 -30 30 0.001
  ```
- **BF_hybrid / dijkstra_hybrid:**
  ```bash
  # Usage: <exe> <V> <min_w> <max_w> <split_ratio> [density] [threads]
  ./bin/BF_hybrid 5000 -30 30 0.6 0.001 8
  ```

#### APSP Algorithms

- **floyd_serial / johnson_serial:**
  ```bash
  # Usage: <exe> <V> <min_w> <max_w> [density]
  ./bin/floyd_serial 1000 -10 50 0.01
  ```
- **floyd_openmp / johnson_openmp:**
  ```bash
  # Usage: <exe> <V> <min_w> <max_w> [density] [threads]
  ./bin/johnson_openmp 1000 -10 50 0.01 8
  ```
- **floyd_cuda / johnson_cuda (stubs):**
  ```bash
  # Usage: <exe> <V> <min_w> <max_w> [density]
  ./bin/floyd_cuda 1000 -10 50 0.01
  ```

## Validate Results

After running the benchmarks, you can compare the output of the parallel SSSP implementations against the serial baseline to ensure correctness. The Root Mean Square Error (RMSE) should be close to 0.

_Note: This script is for SSSP algorithms that produce distance vectors. APSP algorithms produce distance matrices and must be compared differently._

```bash
# Example for a graph with 5000 vertices
python3 scripts/compare_rmse.py output_BF_serial__5000_30_-30.txt output_BF_openmp__5000_30_-30.txt
```
