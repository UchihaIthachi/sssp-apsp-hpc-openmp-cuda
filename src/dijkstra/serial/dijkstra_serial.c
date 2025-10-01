#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <stdbool.h>
#include <time.h>

#include "graph.h"
#include "graph_io.h"
#include "graphGen.h"

/*
 * Simple serial Dijkstra algorithm using adjacency lists built from the
 * edge list.  This implementation uses a linear search for the next
 * closest vertex (O(V^2) time) and does not employ a priority queue.  It
 * assumes all edge weights are non‑negative.  The resulting distance
 * array will contain INT_MAX for unreachable vertices.
 */
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

static void dijkstra_serial(int V, int *head, int *to, int *weight, int *next, int src, int *dist)
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
        /* Find the unvisited vertex with minimum distance */
        for (int i = 0; i < V; i++) {
            if (!visited[i] && dist[i] < best) {
                best = dist[i];
                u = i;
            }
        }
        if (u == -1) break; /* no more reachable vertices */
        visited[u] = true;
        /* Relax outgoing edges of u */
        for (int e = head[u]; e != -1; e = next[e]) {
            int v = to[e];
            int w = weight[e];
            if (dist[u] != INT_MAX && dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
            }
        }
    }
    free(visited);
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        printf("Usage: %s <V> <min_w> <max_w> [density=0.005]\n", argv[0]);
        return 1;
    }
    int V = atoi(argv[1]);
    int min_w = atoi(argv[2]);
    int max_w = atoi(argv[3]);
    double density = (argc > 4) ? atof(argv[4]) : 0.005;

    /* Non-negative weights required */
    if (min_w < 0) {
        fprintf(stderr, "Error: Dijkstra requires non-negative weights\n");
        return 1;
    }

    Graph *g = get_or_create_graph(V, max_w, min_w, density);
    if (!g) return 1;

    int *head, *to, *weight, *next;
    build_adjacency(g, &head, &to, &weight, &next);

    int *dist = (int*)malloc(sizeof(int) * g->V);
    if (!dist) {
        fprintf(stderr, "Out of memory\n");
        free_graph(g);
        free(head); free(to); free(weight); free(next);
        return 1;
    }

    clock_t t0 = clock();
    dijkstra_serial(g->V, head, to, weight, next, 0, dist);
    clock_t t1 = clock();
    double secs = (double)(t1 - t0) / CLOCKS_PER_SEC;
    printf("[dijkstra_serial] time: %.6f s\n", secs);

    save_distance_vector("dijkstra_serial", g->V, max_w, min_w, dist, g->V, false);

    free(dist);
    free(head); free(to); free(weight); free(next);
    free_graph(g);
    return 0;
}