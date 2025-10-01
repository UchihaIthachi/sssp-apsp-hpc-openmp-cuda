/*
 * dijkstra_hybrid.cu
 *
 * Hybrid CPU+GPU implementation of Dijkstra’s algorithm.  The set of edges
 * is partitioned between CPU threads (OpenMP) and GPU threads (CUDA).  Each
 * iteration performs a parallel relaxation on both partitions and then
 * synchronizes.  This continues until no distances are updated.
 *
 * The algorithm assumes non‑negative weights.  It is structurally similar
 * to a hybrid Bellman–Ford: though it still uses repeated relaxation
 * instead of a priority queue, it demonstrates how CPU and GPU can
 * collaborate on the same data.
 */

#include "../../include/graph.h"
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
        if (newDist < old) *changed = 1;
    }
}

/* CPU relaxation function: processes edges in the CPU partition.  Uses
 * OpenMP to parallelize the loop. */
void dijkstra_relax_cpu(const Edge *edges, int E, int *dist, int *changed) {
    int local_changed = 0;
    #pragma omp parallel for
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
    if (local_changed) *changed = 1;
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "Usage: %s <V> <minWeight> <maxWeight> <gpu_ratio> [tol]\n", argv[0]);
        fprintf(stderr, " gpu_ratio: fraction of edges to process on GPU (0.0–1.0)\n");
        return 1;
    }
    int V = atoi(argv[1]);
    int minW = atoi(argv[2]);
    int maxW = atoi(argv[3]);
    double gpu_ratio = atof(argv[4]);
    if (V <= 0 || maxW < minW || minW < 0 || gpu_ratio < 0.0 || gpu_ratio > 1.0) {
        fprintf(stderr, "Invalid arguments.  V>0, minWeight>=0, maxWeight>=minWeight, 0<=gpu_ratio<=1\n");
        return 1;
    }
    Graph *g = load_graph(V, minW, maxW);
    if (!g) return 1;
    int *dist = (int*)malloc(sizeof(int) * g->V);
    if (!dist) { perror("malloc"); free_graph(g); return 1; }
    for (int i = 0; i < g->V; i++) dist[i] = INT_MAX;
    dist[0] = 0;
    /* Partition edges into CPU and GPU sets by simple ratio */
    int gpu_edges_count = (int)(g->E * gpu_ratio);
    int cpu_edges_count = g->E - gpu_edges_count;
    Edge *cpu_edges = (Edge*)malloc(sizeof(Edge) * cpu_edges_count);
    Edge *gpu_edges = (Edge*)malloc(sizeof(Edge) * gpu_edges_count);
    if (!cpu_edges || !gpu_edges) { fprintf(stderr, "Allocation failed\n"); free(dist); free_graph(g); return 1; }
    /* Simple partition: first gpu_edges_count edges to GPU, rest to CPU */
    for (int i = 0; i < gpu_edges_count; i++) gpu_edges[i] = g->edges[i];
    for (int i = 0; i < cpu_edges_count; i++) cpu_edges[i] = g->edges[gpu_edges_count + i];
    /* Allocate GPU memory for GPU edges and distances */
    Edge *d_gpu_edges;
    int *d_dist;
    int *d_changed;
    cudaMalloc((void**)&d_gpu_edges, sizeof(Edge) * gpu_edges_count);
    cudaMalloc((void**)&d_dist, sizeof(int) * g->V);
    cudaMalloc((void**)&d_changed, sizeof(int));
    cudaMemcpy(d_gpu_edges, gpu_edges, sizeof(Edge) * gpu_edges_count, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dist, dist, sizeof(int) * g->V, cudaMemcpyHostToDevice);
    int blockSize = 256;
    int gridSize = (gpu_edges_count + blockSize - 1) / blockSize;
    bool updated;
    do {
        updated = false;
        /* GPU relaxation */
        cudaMemset(d_changed, 0, sizeof(int));
        dijkstra_relax_gpu<<<gridSize, blockSize>>>(d_gpu_edges, gpu_edges_count, d_dist, d_changed);
        cudaDeviceSynchronize();
        /* CPU relaxation */
        int host_changed = 0;
        dijkstra_relax_cpu(cpu_edges, cpu_edges_count, dist, &host_changed);
        /* Copy GPU distances back to host */
        cudaMemcpy(dist, d_dist, sizeof(int) * g->V, cudaMemcpyDeviceToHost);
        /* OR copy host distances to device before next iteration */
        cudaMemcpy(d_dist, dist, sizeof(int) * g->V, cudaMemcpyHostToDevice);
        int gpu_changed_flag;
        cudaMemcpy(&gpu_changed_flag, d_changed, sizeof(int), cudaMemcpyDeviceToHost);
        if (gpu_changed_flag || host_changed) updated = true;
    } while (updated);
    /* Write results */
    write_output("dijkstra", "hybrid", g->V, maxW, minW, dist);
    /* Cleanup */
    free(dist);
    free(cpu_edges);
    free(gpu_edges);
    free_graph(g);
    cudaFree(d_gpu_edges);
    cudaFree(d_dist);
    cudaFree(d_changed);
    return 0;
}