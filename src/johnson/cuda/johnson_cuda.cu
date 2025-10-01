// Placeholder CUDA implementation of Johnson's algorithm.

#include "graph.h"
#include "graph_io.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

void johnson_cuda_stub(const Graph* g, int* dist_matrix) {
    printf("johnson_cuda: not yet implemented.\n");
    // Fill the distance matrix with INF
    for (int i = 0; i < g->V * g->V; i++) {
        dist_matrix[i] = INT_MAX;
    }
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
    johnson_cuda_stub(g, dist_matrix);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("[johnson_cuda] time: %.6f s\n", ms / 1000.0f);

    // Save the dummy matrix (all INF)
    save_distance_matrix("johnson_cuda", g->V, max_w, min_w, dist_matrix, false);

    free(dist_matrix);
    free_graph(g);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}