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

The analysis is split into several Jupyter Notebooks, located in the `notebooks/` directory.

**1. Start Here: `notebooks/00_setup_build.ipynb`**

<a href="https://colab.research.google.com/github/UchihaIthachi/sssp-apsp-hpc-openmp-cuda/blob/main/notebooks/00_setup_build.ipynb" target="_blank" rel="noopener noreferrer">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

Run this notebook first to clone the repository, install dependencies, and compile all the executables.

**2. Per-Algorithm Benchmarks**

Once the setup is complete, you can run the individual benchmark notebooks:

- `notebooks/01_bellman_ford_sssp.ipynb`
- `notebooks/02_dijkstra_sssp.ipynb`
- `notebooks/03_floyd_warshall_apsp.ipynb`
- `notebooks/04_johnson_apsp.ipynb`

**3. Combined Analysis**

- `notebooks/05_compare_rollup.ipynb`: This notebook loads the results from the other notebooks and creates combined comparison plots.

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
