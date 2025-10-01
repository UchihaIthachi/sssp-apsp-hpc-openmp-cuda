/*
 * floyd_cuda.cu
 *
 * Naïve CUDA implementation of the Floyd–Warshall algorithm for all–pairs
 * shortest paths. For each intermediate vertex k, a kernel updates all
 * pairs (i,j) in parallel.
 */

#include "graph.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

/* Kernel that relaxes all (i,j) pairs given a fixed k. */
__global__ void floyd_kernel(int *d, int k, int n, const int INF) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        int ik = d[row * n + k];
        int kj = d[k * n + col];
        if (ik != INF && kj != INF) {
            int new_dist = ik + kj;
            if (new_dist < d[row * n + col]) {
                d[row * n + col] = new_dist;
            }
        }
    }
}

void floyd_warshall_cuda(const Graph* g, int* h_dist) {
    int V = g->V;
    const int INF = INT_MAX / 2;
    size_t bytes = (size_t)V * V * sizeof(int);

    // Initialize host matrix
    for (int i = 0; i < V * V; i++) h_dist[i] = INF;
    for (int i = 0; i < V; i++) h_dist[i * V + i] = 0;
    for (int i = 0; i < g->E; i++) {
        int u = g->edges[i].src;
        int v = g->edges[i].dest;
        int w = g->edges[i].weight;
        if (w < h_dist[u * V + v]) {
            h_dist[u * V + v] = w;
        }
    }

    // Allocate and copy to device
    int *d_dist;
    cudaMalloc((void**)&d_dist, bytes);
    cudaMemcpy(d_dist, h_dist, bytes, cudaMemcpyHostToDevice);

    // Kernel launch configuration
    dim3 block(16, 16);
    dim3 grid((V + block.x - 1) / block.x, (V + block.y - 1) / block.y);

    // Main loop
    for (int k = 0; k < V; k++) {
        floyd_kernel<<<grid, block>>>(d_dist, k, V, INF);
        cudaDeviceSynchronize();
    }

    // Copy results back
    cudaMemcpy(h_dist, d_dist, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_dist);
}

int main(int argc, char **argv) {
    if (argc < 4) {
        printf("Usage: %s <V> <min_w> <max_w> [density=0.005]\n", argv[0]);
        return 1;
    }
    int V = atoi(argv[1]);
    int min_w = atoi(argv[2]);
    int max_w = atoi(argv[3]);
    double density = (argc > 4) ? atof(argv[4]) : 0.005;

    Graph *g = get_or_create_graph(V, max_w, min_w, density);
    if (!g) return 1;

    size_t bytes = (size_t)g->V * g->V * sizeof(int);
    int* dist_matrix = (int*)malloc(bytes);
    if (!dist_matrix) {
        perror("Failed to allocate distance matrix");
        free_graph(g);
        return 1;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    floyd_warshall_cuda(g, dist_matrix);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("[floyd_cuda] time: %.6f s\n", ms / 1000.0f);

    save_distance_matrix("floyd_cuda", g->V, max_w, min_w, dist_matrix);

    free(dist_matrix);
    free_graph(g);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}