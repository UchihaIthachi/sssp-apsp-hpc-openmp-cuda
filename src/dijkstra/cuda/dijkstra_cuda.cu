/*
 * dijkstra_cuda.cu
 *
 * Simplified CUDA implementation of Dijkstra’s algorithm.  This version
 * assumes non‑negative weights.  It uses a parallel relaxation scheme
 * similar to Bellman–Ford: all edges are processed in parallel and
 * distances are updated via atomic operations until no updates occur.
 *
 * NOTE: This is a pedagogical example.  True Dijkstra on the GPU
 * typically uses more advanced data structures (e.g. delta‑stepping or
 * bucketed priority queues) for better performance.
 */

#include "../../include/graph.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <limits.h>

/* CUDA kernel: relax all edges once.  If a shorter path is found,
 * atomically update the destination distance and set the `changed` flag. */
__global__ void relax_edges_kernel(const Edge *edges, int E, int *dist, int *changed) {
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
            *changed = 1;
        }
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
    if (V <= 0 || maxW < minW || minW < 0) {
        fprintf(stderr, "Invalid arguments.  V>0, minWeight>=0, maxWeight>=minWeight\n");
        return 1;
    }
    /* Load or generate graph */
    Graph *g = load_graph(V, minW, maxW);
    if (!g) return 1;
    /* Allocate distances on host and initialize */
    int *h_dist = (int*)malloc(sizeof(int) * g->V);
    if (!h_dist) { perror("malloc"); free_graph(g); return 1; }
    for (int i = 0; i < g->V; i++) h_dist[i] = INT_MAX;
    h_dist[0] = 0;
    /* Allocate device memory */
    Edge *d_edges;
    int *d_dist;
    int *d_changed;
    cudaMalloc((void**)&d_edges, sizeof(Edge) * g->E);
    cudaMalloc((void**)&d_dist, sizeof(int) * g->V);
    cudaMalloc((void**)&d_changed, sizeof(int));
    cudaMemcpy(d_edges, g->edges, sizeof(Edge) * g->E, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dist, h_dist, sizeof(int) * g->V, cudaMemcpyHostToDevice);
    /* Kernel launch configuration */
    int blockSize = 256;
    int gridSize = (g->E + blockSize - 1) / blockSize;
    bool host_changed;
    /* Main relaxation loop: iterate until no changes */
    do {
        host_changed = false;
        cudaMemset(d_changed, 0, sizeof(int));
        relax_edges_kernel<<<gridSize, blockSize>>>(d_edges, g->E, d_dist, d_changed);
        cudaDeviceSynchronize();
        int changedFlag;
        cudaMemcpy(&changedFlag, d_changed, sizeof(int), cudaMemcpyDeviceToHost);
        if (changedFlag != 0) host_changed = true;
    } while (host_changed);
    /* Copy result back */
    cudaMemcpy(h_dist, d_dist, sizeof(int) * g->V, cudaMemcpyDeviceToHost);
    /* Write output */
    write_output("dijkstra", "cuda", g->V, maxW, minW, h_dist);
    /* Clean up */
    free(h_dist);
    free_graph(g);
    cudaFree(d_edges);
    cudaFree(d_dist);
    cudaFree(d_changed);
    return 0;
}