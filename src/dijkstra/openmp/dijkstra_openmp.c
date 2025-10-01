#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <stdbool.h>
#include <omp.h>

#include "../../include/graph.h"
#include "../../utils/graph_io.h"

/* Build adjacency lists from an edge list (same as serial version) */
static void build_adjacency(const Graph *g, int **head, int **to, int **weight, int **next)
{
    int V = g->V;
    int E = g->E;
    *head = (int*)malloc(sizeof(int) * V);
    *to = (int*)malloc(sizeof(int) * E);
    *weight = (int*)malloc(sizeof(int) * E);
    *next = (int*)malloc(sizeof(int) * E);
    for (int i = 0; i < V; i++) (*head)[i] = -1;
    for (int i = 0; i < E; i++) {
        int u = g->edges[i].src;
        int v = g->edges[i].dest;
        int w = g->edges[i].weight;
        (*to)[i] = v;
        (*weight)[i] = w;
        (*next)[i] = (*head)[u];
        (*head)[u] = i;
    }
}

static void dijkstra_openmp(int V, int *head, int *to, int *weight, int *next, int src, int *dist)
{
    bool *visited = (bool*)malloc(sizeof(bool) * V);
    for (int i = 0; i < V; i++) {
        dist[i] = INT_MAX;
        visited[i] = false;
    }
    dist[src] = 0;
    for (int iter = 0; iter < V; iter++) {
        int u = -1;
        int best = INT_MAX;
        /* Parallel search for unvisited vertex with minimum distance */
        #pragma omp parallel
        {
            int local_u = -1;
            int local_best = INT_MAX;
            #pragma omp for nowait
            for (int i = 0; i < V; i++) {
                if (!visited[i] && dist[i] < local_best) {
                    local_best = dist[i];
                    local_u = i;
                }
            }
            #pragma omp critical
            {
                if (local_u != -1 && local_best < best) {
                    best = local_best;
                    u = local_u;
                }
            }
        }
        if (u == -1 || best == INT_MAX) break;
        visited[u] = true;
        /* Relax outgoing edges of u */
        #pragma omp parallel for
        for (int e = head[u]; e != -1; e = next[e]) {
            int v = to[e];
            int w = weight[e];
            int du = dist[u];
            if (du != INT_MAX && du + w < dist[v]) {
                #pragma omp critical
                {
                    if (du + w < dist[v]) {
                        dist[v] = du + w;
                    }
                }
            }
        }
    }
    free(visited);
}

int main(int argc, char **argv)
{
    int V = 10;
    int min_wt = 0;
    int max_wt = 10;
    int numThreads = 0;
    if (argc > 1) V = atoi(argv[1]);
    if (argc > 2) min_wt = atoi(argv[2]);
    if (argc > 3) max_wt = atoi(argv[3]);
    if (argc > 4) numThreads = atoi(argv[4]);
    if (min_wt < 0) {
        fprintf(stderr, "Error: Dijkstra requires non‑negative weights\n");
        return 1;
    }
    if (numThreads > 0) omp_set_num_threads(numThreads);
    Graph *g = load_graph(V, min_wt, max_wt);
    if (!g) return 1;
    int *head, *to, *weight, *next;
    build_adjacency(g, &head, &to, &weight, &next);
    int *dist = (int*)malloc(sizeof(int) * V);
    if (!dist) {
        fprintf(stderr, "Out of memory\n");
        free_graph(g);
        free(head); free(to); free(weight); free(next);
        return 1;
    }
    dijkstra_openmp(V, head, to, weight, next, 0, dist);
    write_output("dijkstra", "openmp", V, max_wt, min_wt, dist);
    free(dist);
    free(head); free(to); free(weight); free(next);
    free_graph(g);
    return 0;
}