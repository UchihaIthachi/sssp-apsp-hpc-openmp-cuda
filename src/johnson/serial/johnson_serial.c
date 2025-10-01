// Serial implementation of Johnson's algorithm for all-pairs shortest paths.

#include "graph.h"
#include "graph_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <stdbool.h>
#include <time.h>

// Simple Bellman-Ford for reweighting. Returns 0 on success, -1 on negative cycle.
static int bellman_ford_reweight(const Graph *g, int *h) {
    int V = g->V;
    for (int i = 0; i < V; i++) h[i] = INT_MAX;
    h[0] = 0; // Assuming source is 0 for the augmented graph

    for (int i = 0; i < V - 1; i++) {
        for (int j = 0; j < g->E; j++) {
            int u = g->edges[j].src;
            int v = g->edges[j].dest;
            int w = g->edges[j].weight;
            if (h[u] != INT_MAX && h[u] + w < h[v]) {
                h[v] = h[u] + w;
            }
        }
    }

    for (int j = 0; j < g->E; j++) {
        int u = g->edges[j].src;
        int v = g->edges[j].dest;
        int w = g->edges[j].weight;
        if (h[u] != INT_MAX && h[u] + w < h[v]) {
            return -1; // Negative cycle detected
        }
    }
    return 0;
}

typedef struct AdjNode {
    int to;
    int w;
    struct AdjNode *next;
} AdjNode;

static AdjNode** build_adj_list(const Graph *g) {
    AdjNode **adj = calloc(g->V, sizeof(AdjNode*));
    for (int i = 0; i < g->E; i++) {
        AdjNode *node = malloc(sizeof(AdjNode));
        node->to = g->edges[i].dest;
        node->w = g->edges[i].weight;
        node->next = adj[g->edges[i].src];
        adj[g->edges[i].src] = node;
    }
    return adj;
}

static void free_adj_list(AdjNode **adj, int V) {
    for (int i = 0; i < V; i++) {
        AdjNode *curr = adj[i];
        while (curr) {
            AdjNode *tmp = curr;
            curr = curr->next;
            free(tmp);
        }
    }
    free(adj);
}

// Simple Dijkstra using an array (O(V^2))
static void dijkstra(int V, AdjNode **adj, int src, int *dist) {
    bool *visited = calloc(V, sizeof(bool));
    for (int i = 0; i < V; i++) dist[i] = INT_MAX;
    dist[src] = 0;

    for (int i = 0; i < V; i++) {
        int u = -1;
        int min_dist = INT_MAX;
        for (int j = 0; j < V; j++) {
            if (!visited[j] && dist[j] < min_dist) {
                min_dist = dist[j];
                u = j;
            }
        }

        if (u == -1) break;
        visited[u] = true;

        for (AdjNode *edge = adj[u]; edge; edge = edge->next) {
            if (dist[u] != INT_MAX && dist[u] + edge->w < dist[edge->to]) {
                dist[edge->to] = dist[u] + edge->w;
            }
        }
    }
    free(visited);
}

void johnson_serial(const Graph* g, int* dist_matrix, bool* has_neg_cycle) {
    int V = g->V;
    int* h = malloc(V * sizeof(int));
    if (bellman_ford_reweight(g, h) != 0) {
        *has_neg_cycle = true;
        free(h);
        return;
    }
    *has_neg_cycle = false;

    Graph reweighted_g = *g;
    reweighted_g.edges = malloc(g->E * sizeof(Edge));
    for (int i = 0; i < g->E; i++) {
        reweighted_g.edges[i] = g->edges[i];
        int u = g->edges[i].src;
        int v = g->edges[i].dest;
        reweighted_g.edges[i].weight += h[u] - h[v];
    }

    AdjNode** adj = build_adj_list(&reweighted_g);
    
    int* temp_dist = malloc(V * sizeof(int));
    for (int i = 0; i < V; i++) {
        dijkstra(V, adj, i, temp_dist);
        for (int j = 0; j < V; j++) {
            if (temp_dist[j] != INT_MAX) {
                dist_matrix[i * V + j] = temp_dist[j] - h[i] + h[j];
            } else {
                dist_matrix[i * V + j] = INT_MAX;
            }
        }
    }

    free(temp_dist);
    free_adj_list(adj, V);
    free(reweighted_g.edges);
    free(h);
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

    int* dist_matrix = malloc(sizeof(int) * g->V * g->V);
    bool has_neg_cycle = false;

    clock_t t0 = clock();
    johnson_serial(g, dist_matrix, &has_neg_cycle);
    clock_t t1 = clock();
    double secs = (double)(t1 - t0) / CLOCKS_PER_SEC;
    printf("[johnson_serial] time: %.6f s\n", secs);

    save_distance_matrix("johnson_serial", g->V, max_w, min_w, dist_matrix, has_neg_cycle);

    free(dist_matrix);
    free_graph(g);
    return 0;
}