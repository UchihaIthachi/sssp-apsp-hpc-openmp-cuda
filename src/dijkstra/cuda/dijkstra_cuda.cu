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

#include "graph.h"
#include "graph_io.h"
#include "graphGen.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
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
        fprintf(stderr, "Usage: %s <V> <min_w> <max_w> [density=0.005]\n", argv[0]);
        return 1;
    }
    int V = atoi(argv[1]);
    int min_w = atoi(argv[2]);
    int max_w = atoi(argv[3]);
    double density = (argc > 4) ? atof(argv[4]) : 0.005;

    if (V <= 0 || max_w < min_w || min_w < 0) {
        fprintf(stderr, "Invalid arguments. V > 0, min_w >= 0, max_w >= min_w\n");
        return 1;
    }

    Graph *g = get_or_create_graph(V, max_w, min_w, density);
    if (!g) return 1;

    int *h_dist = (int*)malloc(sizeof(int) * g->V);
    if (!h_dist) { perror("malloc"); free_graph(g); return 1; }
    for (int i = 0; i < g->V; i++) h_dist[i] = INT_MAX;
    h_dist[0] = 0;

    Edge *d_edges;
    int *d_dist;
    int *d_changed;
    cudaMalloc((void**)&d_edges, sizeof(Edge) * g->E);
    cudaMalloc((void**)&d_dist, sizeof(int) * g->V);
    cudaMalloc((void**)&d_changed, sizeof(int));
    cudaMemcpy(d_edges, g->edges, sizeof(Edge) * g->E, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dist, h_dist, sizeof(int) * g->V, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (g->E + blockSize - 1) / blockSize;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    bool host_changed;
    do {
        host_changed = false;
        cudaMemset(d_changed, 0, sizeof(int));
        relax_edges_kernel<<<gridSize, blockSize>>>(d_edges, g->E, d_dist, d_changed);
        cudaDeviceSynchronize();
        int changedFlag;
        cudaMemcpy(&changedFlag, d_changed, sizeof(int), cudaMemcpyDeviceToHost);
        if (changedFlag != 0) host_changed = true;
    } while (host_changed);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("[dijkstra_cuda] time: %.6f s\n", ms / 1000.0f);

    cudaMemcpy(h_dist, d_dist, sizeof(int) * g->V, cudaMemcpyDeviceToHost);
    save_distance_vector("dijkstra_cuda", g->V, max_w, min_w, h_dist, g->V, false);

    free(h_dist);
    free_graph(g);
    cudaFree(d_edges);
    cudaFree(d_dist);
    cudaFree(d_changed);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}