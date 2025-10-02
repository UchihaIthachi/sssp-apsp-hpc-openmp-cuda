/*
 * floyd_openmp.c
 *
 * Optimized OpenMP implementation of the Floyd–Warshall algorithm.
 * Uses a single parallel region to reduce thread overhead and improve
 * cache performance.
 */

#include "graph.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <stdbool.h>
#include <omp.h>

void floyd_warshall_openmp(const Graph* g, int* __restrict dist) {
    const int V = g->V;
    const int INF = INT_MAX / 2;

    // init matrix
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
            dist[i*V + j] = (i == j) ? 0 : INF;
        }
    }

    // seed edges (sequential is fine; parallelizing needs atomic min)
    for (int e = 0; e < g->E; e++) {
        int u = g->edges[e].src;
        int v = g->edges[e].dest;
        int w = g->edges[e].weight;
        if (w < dist[u*V + v]) dist[u*V + v] = w;
    }

    // one team reused across all k
    #pragma omp parallel
    {
        for (int k = 0; k < V; k++) {
            const int * __restrict row_k = &dist[k*V];

            #pragma omp for schedule(static)
            for (int i = 0; i < V; i++) {
                int * __restrict row_i = &dist[i*V];
                const int dik = row_i[k];
                if (dik == INF) continue;  // no path i->k, skip whole row

                // Vectorizable inner loop
                #pragma omp simd
                for (int j = 0; j < V; j++) {
                    int dkj = row_k[j];
                    // both INF check folded into arithmetic: avoid overflow
                    int via = (dkj >= INF - dik) ? INF : (dik + dkj);
                    if (via < row_i[j]) row_i[j] = via;
                }
            }
            // Barrier to ensure all threads finish an iteration `k` before any start `k+1`
            #pragma omp barrier
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

    bool has_neg_cycle = false;
    for (int i = 0; i < V; ++i) {
        if (dist_matrix[i*V + i] < 0) {
            has_neg_cycle = true;
            break;
        }
    }

    save_distance_matrix("floyd_openmp", g->V, max_w, min_w, dist_matrix, has_neg_cycle);

    free(dist_matrix);
    free_graph(g);
    return 0;
}