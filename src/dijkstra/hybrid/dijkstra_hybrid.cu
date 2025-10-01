/*
 * dijkstra_hybrid.cu
 *
 * Hybrid CPU+GPU implementation of Dijkstra’s algorithm. The set of edges
 * is partitioned between CPU threads (OpenMP) and GPU threads (CUDA). Each
 * iteration performs a parallel relaxation on both partitions and then
 * synchronizes. This continues until no distances are updated.
 *
 * The algorithm assumes non-negative weights. It is structurally similar
 * to a hybrid Bellman–Ford.
 */

#include "graph.h"
#include "utils.h"
#include "graphGen.h"
#include <cuda_runtime.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

/* GPU relaxation kernel: processes edges in the GPU partition. */
__global__ void dijkstra_relax_gpu(const Edge *edges, int E, int *dist, int *changed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= E) return;
    int u = edges[idx].src;
    int v = edges[idx].dest;
    int w = edges[idx].weight;
    int du = dist[u];
    if (du != INT_MAX && du + w < dist[v]) {
        int newDist = du + w;
        int old = atomicMin(&dist[v], newDist);
        if (newDist < old) {
            // Non-atomically setting changed flag is fine; any thread seeing a change is enough.
            *changed = 1;
        }
    }
}

/* CPU relaxation function: processes edges in the CPU partition. */
void dijkstra_relax_cpu(const Edge *edges, int E, int *dist, int *changed_flag) {
    int local_changed = 0;
    #pragma omp parallel for reduction(||:local_changed)
    for (int i = 0; i < E; i++) {
        int u = edges[i].src;
        int v = edges[i].dest;
        int w = edges[i].weight;
        int du = dist[u];
        if (du != INT_MAX && du + w < dist[v]) {
            int newDist = du + w;
            #pragma omp critical
            {
                if (newDist < dist[v]) {
                    dist[v] = newDist;
                    local_changed = 1;
                }
            }
        }
    }
    if (local_changed) {
        *changed_flag = 1;
    }
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "Usage: %s <V> <min_w> <max_w> <gpu_ratio> [density=0.005] [threads=0]\n", argv[0]);
        return 1;
    }
    int V = atoi(argv[1]);
    int min_w = atoi(argv[2]);
    int max_w = atoi(argv[3]);
    double gpu_ratio = atof(argv[4]);
    double density = (argc > 5) ? atof(argv[5]) : 0.005;
    int numThreads = (argc > 6) ? atoi(argv[6]) : 0;

    if (V <= 0 || max_w < min_w || min_w < 0 || gpu_ratio < 0.0 || gpu_ratio > 1.0) {
        fprintf(stderr, "Invalid arguments. V>0, min_w>=0, max_w>=min_w, 0<=gpu_ratio<=1\n");
        return 1;
    }
    if (numThreads > 0) {
        omp_set_num_threads(numThreads);
    }

    Graph *g = get_or_create_graph(V, max_w, min_w, density);
    if (!g) return 1;

    int *h_dist = (int*)malloc(sizeof(int) * g->V);
    if (!h_dist) { perror("malloc"); free_graph(g); return 1; }
    for (int i = 0; i < g->V; i++) h_dist[i] = INT_MAX;
    h_dist[0] = 0;

    int gpu_edges_count = (int)(g->E * gpu_ratio);
    int cpu_edges_count = g->E - gpu_edges_count;
    Edge *cpu_edges = g->edges;
    Edge *gpu_edges = g->edges + cpu_edges_count;

    Edge *d_gpu_edges;
    int *d_dist;
    int *d_changed;
    cudaMalloc((void**)&d_gpu_edges, sizeof(Edge) * gpu_edges_count);
    cudaMalloc((void**)&d_dist, sizeof(int) * g->V);
    cudaMalloc((void**)&d_changed, sizeof(int));

    int blockSize = 256;
    int gridSize = (gpu_edges_count + blockSize - 1) / blockSize;

    double t0 = omp_get_wtime();
    bool updated;
    do {
        updated = false;
        int h_changed = 0;
        int d_changed_val = 0;

        cudaMemcpy(d_dist, h_dist, sizeof(int) * g->V, cudaMemcpyHostToDevice);
        cudaMemcpy(d_gpu_edges, gpu_edges, sizeof(Edge) * gpu_edges_count, cudaMemcpyHostToDevice);
        cudaMemset(d_changed, 0, sizeof(int));
        
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                dijkstra_relax_cpu(cpu_edges, cpu_edges_count, h_dist, &h_changed);
            }
            #pragma omp section
            {
                dijkstra_relax_gpu<<<gridSize, blockSize>>>(d_gpu_edges, gpu_edges_count, d_dist, d_changed);
                cudaDeviceSynchronize();
                cudaMemcpy(&d_changed_val, d_changed, sizeof(int), cudaMemcpyDeviceToHost);
            }
        }

        // After both sections complete, merge results from GPU
        cudaMemcpy(h_dist, d_dist, sizeof(int) * g->V, cudaMemcpyDeviceToHost);
        
        if (h_changed || d_changed_val) {
            updated = true;
        }

    } while (updated);
    double t1 = omp_get_wtime();
    printf("[dijkstra_hybrid] time: %.6f s (gpu_ratio=%.2f, threads=%d)\n", t1 - t0, gpu_ratio, (numThreads ? numThreads : omp_get_max_threads()));

    save_distance_vector("dijkstra_hybrid", g->V, max_w, min_w, h_dist, g->V, false);

    free(h_dist);
    free_graph(g);
    cudaFree(d_gpu_edges);
    cudaFree(d_dist);
    cudaFree(d_changed);
    return 0;
}