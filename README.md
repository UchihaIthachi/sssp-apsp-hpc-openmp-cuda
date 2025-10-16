<p align="center">
  <img src="https://img.shields.io/github/license/UchihaIthachi/sssp-apsp-hpc-openmp-cuda?style=for-the-badge" alt="License">
  <img src="https://img.shields.io/github/languages/top/UchihaIthachi/sssp-apsp-hpc-openmp-cuda?style=for-the-badge" alt="Top Language">
  <img src="https://img.shields.io/badge/PRs-Welcome-brightgreen.svg?style=for-the-badge" alt="PRs Welcome">
</p>

# 🚀 High-Performance SSSP & APSP Algorithms with OpenMP + CUDA

A research-grade performance study of classic shortest-path algorithms (SSSP & APSP) using Serial, OpenMP (CPU), and CUDA (GPU). This repo includes well-optimized C++ kernels, reproducible Jupyter benchmarks, and HPC profiling tools—all tailored for both teaching and advanced performance engineering.

---

## 🔍 Algorithms Implemented

| Algorithm          | Type | Description                                                                   |
| ------------------ | ---- | ----------------------------------------------------------------------------- |
| **Bellman-Ford**   | SSSP | Supports negative weights, detects cycles. Frontier variant discussed.        |
| **Dijkstra**       | SSSP | Classic non-negative paths. Includes OpenMP variant and Δ-stepping analysis.  |
| **Floyd-Warshall** | APSP | Dynamic programming; efficient on dense graphs; includes cache-aware version. |
| **Johnson's**      | APSP | Combines BF & Dijkstra. Efficient for sparse graphs with negative weights.    |

---

## ⚙️ Parallel Variants

Each algorithm is implemented in the following forms to explore parallelism trade-offs:

- `serial`: Clean single-threaded baseline.
- `openmp`: Loop-parallelized CPU kernels with scheduling and locking optimizations.
- `cuda`: Optimized GPU kernels using thread-level parallelism and memory coalescing.
- `hybrid`: (optional) Proof-of-concept split execution between CPU and GPU.

---

## 🧠 What You'll Learn

This repo is not just code—it's an HPC lab in a box. Each notebook dives deep into a key HPC concept:

| Topic                      | Techniques & Insights                                                                                |
| -------------------------- | ---------------------------------------------------------------------------------------------------- |
| **Work vs Parallelism**    | When to prefer Bellman-Ford over Dijkstra? What trade-offs do Δ-stepping and frontier methods offer? |
| **Memory Optimization**    | Blocked Floyd-Warshall for cache locality; pinned host memory & CUDA streams for async transfers.    |
| **Thread Scheduling**      | OpenMP `static`, `dynamic`, and `guided` schedules benchmarked and visualized.                       |
| **Profiling & Analysis**   | Nsight Systems CLI traces of GPU kernels, memory copies, and overlap timing.                         |
| **GPU vs CPU Tradeoffs**   | Real-world scenarios where GPU underperforms (e.g., small graph, low compute intensity).             |
| **cuGraph Comparison**     | Benchmarks against RAPIDS cuGraph to set baselines for CUDA kernel quality.                          |
| **Real vs Synthetic Data** | Benchmarks with synthetic ER graphs and real-world road/social/web networks from SNAP & DIMACS.      |

---

## 📁 Project Layout

```
notebooks/
00_setup_build.ipynb         # 📦 Setup: clone, build, and detect environment
01_bellman_ford_sssp.ipynb   # 🔁 BF variants benchmarked on SSSP with negative weights
02_dijkstra_sssp.ipynb       # ⏩ Dijkstra variants on non-negative graphs, incl. cuGraph
03_floyd_warshall_apsp.ipynb # 🧮 Dense graph APSP + Blocked FW
04_johnson_apsp.ipynb        # 🔀 Sparse graph APSP + parallel Dijkstra passes
05_compare_rollup.ipynb      # 📊 Rollup notebook for plotting speedup, crossover points
src/
*.cpp, *.cu, *.h             # 🧩 All algorithm implementations
bin/
executables after `make`    # 🧵 CPU/GPU targets (e.g., `BF_openmp`, `dijkstra_cuda`)
Makefile                      # 🔧 Customizable build logic (auto-detects CUDA)
```

---

## ⚡ Quickstart on Google Colab

| Notebook                  | Link                                                                                                                                                                                                            |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Setup (build, env detect) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/UchihaIthachi/sssp-apsp-hpc-openmp-cuda/blob/main/notebooks/00_setup_build.ipynb)         |
| Bellman–Ford SSSP         | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/UchihaIthachi/sssp-apsp-hpc-openmp-cuda/blob/main/notebooks/01_bellman_ford_sssp.ipynb)   |
| Dijkstra SSSP             | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/UchihaIthachi/sssp-apsp-hpc-openmp-cuda/blob/main/notebooks/02_dijkstra_sssp.ipynb)       |
| Floyd–Warshall APSP       | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/UchihaIthachi/sssp-apsp-hpc-openmp-cuda/blob/main/notebooks/03_floyd_warshall_apsp.ipynb) |
| Johnson's APSP            | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/UchihaIthachi/sssp-apsp-hpc-openmp-cuda/blob/main/notebooks/04_johnson_apsp.ipynb)        |
| Compare & Rollup          | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/UchihaIthachi/sssp-apsp-hpc-openmp-cuda/blob/main/notebooks/05_compare_rollup.ipynb)      |

---

## 🛠 Manual Usage

### 🔨 Build

Requires:

- **GCC** with OpenMP support
- **CUDA Toolkit** (for CUDA/hybrid variants)

```bash
# Clone the repository
git clone https://github.com/UchihaIthachi/sssp-apsp-hpc-openmp-cuda.git
cd sssp-apsp-hpc-openmp-cuda

# Build all available executables
make all

# Clean build artifacts
make clean
```

Environment variables:

```bash
# Override GPU architecture if needed
export GPU_ARCH=sm_75
# Skip CUDA build
export DISABLE_CUDA=1
```

---

### ▶️ Run

Each executable takes the following CLI format:

```bash
./bin/BF_openmp <V> <min_w> <max_w> [density=0.01] [threads=4]
./bin/dijkstra_cuda <V> <min_w> <max_w> [density=0.01]
```

Outputs:

- Saves `.csv` or `.txt` of shortest-path results per variant.
- Prints time in `s` to stdout.

---

## 📈 Example Results

| Vertices | Dijkstra CUDA | BF OpenMP | FW OpenMP | Johnson CUDA |
| -------- | ------------- | --------- | --------- | ------------ |
| 500      | 18.2×         | 1.12×     | 5.7×      | 12.8×        |
| 1000     | 69.3×         | 1.86×     | 11.3×     | 30.4×        |
| 2000     | 442.9×        | 2.37×     | 22.4×     | 65.2×        |
| 5000     | 3802×         | 2.61×     | 60.9×     | 131.7×       |

(CUDA outperforms CPU significantly on large graphs; OpenMP gives modest gains depending on algorithm and system.)

---

## 💡 Credits & References

- CUDA streams and pinned memory: [mups.etf.rs](https://mups.etf.rs)
- cuGraph benchmarking: [RAPIDS.ai](https://rapids.ai)
- Floyd–Warshall blocked cache-aware layout: [arXiv 2009.11873](https://arxiv.org/abs/2009.11873)
- Johnson algorithm review: [Moore & Kalapos, 2010](https://cs.cmu.edu/~calos/johnson-apsp)
- Δ-Stepping SSSP: [Meyer & Sanders, JEA 2003](https://doi.org/10.1145/857082.857123)

---

## 🙌 Contributions Welcome

Have an idea to extend the kernels, add distributed memory (MPI), or optimize memory coalescing? Feel free to open an issue or pull request.

> If you're using this for coursework or research, feel free to cite or fork with attribution.

---

## 📄 License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for details.
