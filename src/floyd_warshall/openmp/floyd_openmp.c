/*
 * floyd_openmp.c
 *
 * OpenMP implementation of the Floyd–Warshall algorithm. Parallelizes
 * the inner two loops (i,j) for each intermediate vertex k.
 */

#include "graph.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <omp.h>

void floyd_warshall_openmp(const Graph* g, int* dist) {
    int V = g->V;
    const int INF = INT_MAX / 2;

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
            dist[i * V + j] = (i == j) ? 0 : INF;
        }
    }

    for (int i = 0; i < g->E; i++) {
        int u = g->edges[i].src;
        int v = g->edges[i].dest;
        int w = g->edges[i].weight;
        if (w < dist[u * V + v]) {
            dist[u * V + v] = w;
        }
    }

    for (int k = 0; k < V; k++) {
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < V; i++) {
            for (int j = 0; j < V; j++) {
                if (dist[i * V + k] != INF && dist[k * V + j] != INF) {
                    int new_dist = dist[i * V + k] + dist[k * V + j];
                    if (new_dist < dist[i * V + j]) {
                        dist[i * V + j] = new_dist;
                    }
                }
            }
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 4) {
        printf("Usage: %s <V> <min_w> <max_w> [density=0.005] [threads=0]\n", argv[0]);
        return 1;
    }
    int V = atoi(argv[1]);
    int min_w = atoi(argv[2]);
    int max_w = atoi(argv[3]);
    double density = (argc > 4) ? atof(argv[4]) : 0.005;
    int numThreads = (argc > 5) ? atoi(argv[5]) : 0;

    if (numThreads > 0) {
        omp_set_num_threads(numThreads);
    }

    Graph *g = get_or_create_graph(V, max_w, min_w, density);
    if (!g) return 1;

    int* dist_matrix = (int*)malloc(sizeof(int) * g->V * g->V);
    if (!dist_matrix) {
        perror("Failed to allocate distance matrix");
        free_graph(g);
        return 1;
    }

    double t0 = omp_get_wtime();
    floyd_warshall_openmp(g, dist_matrix);
    double t1 = omp_get_wtime();
    printf("[floyd_openmp] time: %.6f s (threads=%d)\n", t1 - t0, (numThreads ? numThreads : omp_get_max_threads()));

    save_distance_matrix("floyd_openmp", g->V, max_w, min_w, dist_matrix, false);

    free(dist_matrix);
    free_graph(g);
    return 0;
}