/*
 * floyd_openmp.c
 *
 * OpenMP implementation of the Floyd–Warshall algorithm.  Parallelizes
 * the inner two loops (i,j) for each intermediate vertex k.  Uses the
 * same input and output conventions as the serial version.
 */

#include "../../include/graph.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <omp.h>

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
    const int INF = INT_MAX / 4;
    int *dist = (int*)malloc(sizeof(int) * V * V);
    if (!dist) { perror("malloc"); free_graph(g); return 1; }
    /* Initialize matrix */
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
            if (i == j) dist[i * V + j] = 0;
            else dist[i * V + j] = INF;
        }
    }
    /* Set edges */
    for (int e = 0; e < g->E; e++) {
        int u = g->edges[e].src;
        int v = g->edges[e].dest;
        int w = g->edges[e].weight;
        if (w < dist[u * V + v]) dist[u * V + v] = w;
    }
    /* Floyd–Warshall with OpenMP: parallelize i,j loops */
    for (int k = 0; k < V; k++) {
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < V; i++) {
            for (int j = 0; j < V; j++) {
                int viaK = dist[i * V + k] + dist[k * V + j];
                if (viaK < dist[i * V + j]) dist[i * V + j] = viaK;
            }
        }
    }
    /* Write output */
    char filename[256];
    snprintf(filename, sizeof(filename), "floyd_openmp__%d_%d_%d.txt", V, maxW, minW);
    FILE *fp = fopen(filename, "w");
    if (!fp) { perror("fopen"); free(dist); free_graph(g); return 1; }
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
            int d = dist[i * V + j];
            if (d > INT_MAX/8) fprintf(fp, "INF");
            else fprintf(fp, "%d", d);
            if (j < V - 1) fprintf(fp, " ");
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
    free(dist);
    free_graph(g);
    return 0;
}