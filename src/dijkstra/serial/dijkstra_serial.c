#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <stdbool.h>

#include "../../include/graph.h"
#include "../../utils/graph_io.h"

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
    int V = 10;
    int min_wt = 0;
    int max_wt = 10;
    if (argc > 1) V = atoi(argv[1]);
    if (argc > 2) min_wt = atoi(argv[2]);
    if (argc > 3) max_wt = atoi(argv[3]);
    /* Non‑negative weights required */
    if (min_wt < 0) {
        fprintf(stderr, "Error: Dijkstra requires non‑negative weights\n");
        return 1;
    }
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
    dijkstra_serial(V, head, to, weight, next, 0, dist);
    write_output("dijkstra", "serial", V, max_wt, min_wt, dist);
    free(dist);
    free(head); free(to); free(weight); free(next);
    free_graph(g);
    return 0;
}