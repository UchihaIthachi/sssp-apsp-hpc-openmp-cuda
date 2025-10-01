/*
 * floyd_cuda.cu
 *
 * Naïve CUDA implementation of the Floyd–Warshall algorithm for all–pairs
 * shortest paths.  For each intermediate vertex k, a kernel updates all
 * pairs (i,j) in parallel.  This implementation is primarily illustrative
 * and does not exploit shared memory or tiling.  Graph data is loaded
 * using the common graph utilities and converted to a dense matrix.
 */

#include "../../include/graph.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

/* Kernel that relaxes all (i,j) pairs given a fixed k.  Updates d[i*n+j] if
 * going through k yields a shorter path. */
__global__ void floyd_kernel(int *d, int k, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        int ik = d[row * n + k];
        int kj = d[k * n + col];
        int current = d[row * n + col];
        int via = ik + kj;
        if (via < current) d[row * n + col] = via;
    }
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <V> <minWeight> <maxWeight>\n", argv[0]);
        return 1;
    }
    int V = atoi(argv[1]);
    int minW = atoi(argv[2]);
    int maxW = atoi(argv[3]);
    if (V <= 0 || maxW < minW) {
        fprintf(stderr, "Invalid arguments.\n");
        return 1;
    }
    Graph *g = load_graph(V, minW, maxW);
    if (!g) return 1;
    /* Allocate host matrix */
    const int INF = INT_MAX / 4;
    size_t bytes = (size_t)V * V * sizeof(int);
    int *h_dist = (int*)malloc(bytes);
    if (!h_dist) { perror("malloc"); free_graph(g); return 1; }
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
            if (i == j) h_dist[i * V + j] = 0;
            else h_dist[i * V + j] = INF;
        }
    }
    /* Incorporate edges */
    for (int e = 0; e < g->E; e++) {
        int u = g->edges[e].src;
        int v = g->edges[e].dest;
        int w = g->edges[e].weight;
        if (w < h_dist[u * V + v]) h_dist[u * V + v] = w;
    }
    /* Allocate device matrix and copy */
    int *d_dist;
    cudaMalloc((void**)&d_dist, bytes);
    cudaMemcpy(d_dist, h_dist, bytes, cudaMemcpyHostToDevice);
    /* Kernel launch configuration: 16×16 threads per block */
    dim3 block(16, 16);
    dim3 grid((V + block.x - 1) / block.x, (V + block.y - 1) / block.y);
    for (int k = 0; k < V; k++) {
        floyd_kernel<<<grid, block>>>(d_dist, k, V);
        cudaDeviceSynchronize();
    }
    /* Copy results back */
    cudaMemcpy(h_dist, d_dist, bytes, cudaMemcpyDeviceToHost);
    /* Write output */
    char filename[256];
    snprintf(filename, sizeof(filename), "floyd_cuda__%d_%d_%d.txt", V, maxW, minW);
    FILE *fp = fopen(filename, "w");
    if (!fp) { perror("fopen"); free(h_dist); free_graph(g); cudaFree(d_dist); return 1; }
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
            int d = h_dist[i * V + j];
            if (d > INT_MAX/8) fprintf(fp, "INF");
            else fprintf(fp, "%d", d);
            if (j < V - 1) fprintf(fp, " ");
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
    /* Cleanup */
    free(h_dist);
    free_graph(g);
    cudaFree(d_dist);
    return 0;
}