<a href="https://colab.research.google.com/github/UchihaIthachi/bellman-ford-hpc-openmp-cuda/blob/main/Bellman-Ford-HPC-Analysis.ipynb" target="_blank" rel="noopener noreferrer">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

# Bellman-Ford HPC: A Comparative Analysis

This project provides and analyzes four different implementations of the Bellman-Ford shortest path algorithm, showcasing a range of high-performance computing (HPC) techniques.

- **Serial:** A standard, single-threaded baseline implementation.
- **OpenMP:** A parallel implementation using OpenMP for multi-core CPUs.
- **CUDA:** A massively parallel implementation using CUDA for NVIDIA GPUs.
- **Hybrid:** A combined CPU+GPU approach using both OpenMP and CUDA.

## Quickstart: Performance Analysis with Jupyter

The best way to explore this project is through the interactive Jupyter Notebook. It will guide you through building the code, running benchmarks, and visualizing the performance results.

**>> [Open the Performance Analysis Notebook](Bellman-Ford-HPC-Analysis.ipynb) <<**

## Manual Build and Run

If you prefer to work from the command line, you can build and run the executables manually.

### Build

You will need GCC (with OpenMP) and the NVIDIA CUDA Toolkit (`nvcc`).

```bash
make
```

This will create four executables: `BF_serial`, `BF_openmp`, `BF_cuda`, and `BF_hybrid`.

### Run

Here are some examples of how to run each variant. The executables will automatically generate graph data if it doesn't exist.

**Serial**

```bash
./bin/BF_serial 5000 -30 30 0.001
```

**OpenMP (8 threads)**

```bash
OMP_NUM_THREADS=8 ./bin/BF_openmp 5000 -30 30 0.001 8
```

**CUDA**

```bash
./bin/BF_cuda 5000 -30 30 0.001
```

**Hybrid (60% of work on GPU)**

```bash
OMP_NUM_THREADS=8 ./bin/BF_hybrid 5000 -30 30 0.6 0.001 8
```

## Validate Results

After running the benchmarks, you can compare the output of the parallel implementations against the serial baseline to ensure correctness. The Root Mean Square Error (RMSE) should be close to 0.

```bash
# Example for a graph with 5000 vertices
python3 scripts/compare_rmse.py serial_output__5000_30_-30.txt openmp_output__5000_30_-30.txt
python3 scripts/compare_rmse.py serial_output__5000_30_-30.txt cuda_output__5000_30_-30.txt
python3 scripts/compare_rmse.py serial_output__5000_30_-30.txt hybrid_output__5000_30_-30.txt
```
