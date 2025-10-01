/*
 * floyd_serial.c
 *
 * Serial implementation of the Floyd–Warshall algorithm. Computes
 * all–pairs shortest paths on a weighted directed graph.
 */

#include "graph.h"
#include "graph_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>

void floyd_warshall_serial(const Graph* g, int* dist) {
    int V = g->V;
    const int INF = INT_MAX / 2; // Prevent overflow

    // Initialize distance matrix
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
            dist[i * V + j] = (i == j) ? 0 : INF;
        }
    }

    // Initialize with edge weights
    for (int i = 0; i < g->E; i++) {
        int u = g->edges[i].src;
        int v = g->edges[i].dest;
        int w = g->edges[i].weight;
        if (w < dist[u * V + v]) {
            dist[u * V + v] = w;
        }
    }

    // Floyd-Warshall main loop
    for (int k = 0; k < V; k++) {
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
        printf("Usage: %s <V> <min_w> <max_w> [density=0.005]\n", argv[0]);
        return 1;
    }
    int V = atoi(argv[1]);
    int min_w = atoi(argv[2]);
    int max_w = atoi(argv[3]);
    double density = (argc > 4) ? atof(argv[4]) : 0.005;

    Graph *g = get_or_create_graph(V, max_w, min_w, density);
    if (!g) return 1;

    int* dist_matrix = (int*)malloc(sizeof(int) * g->V * g->V);
    if (!dist_matrix) {
        perror("Failed to allocate distance matrix");
        free_graph(g);
        return 1;
    }

    clock_t t0 = clock();
    floyd_warshall_serial(g, dist_matrix);
    clock_t t1 = clock();
    double secs = (double)(t1 - t0) / CLOCKS_PER_SEC;
    printf("[floyd_serial] time: %.6f s\n", secs);

    save_distance_matrix("floyd_serial", g->V, max_w, min_w, dist_matrix, false);

    free(dist_matrix);
    free_graph(g);
    return 0;
}