// OpenMP implementation of Johnson's algorithm for all‑pairs shortest paths.
//
// This is a simple parallel version: after reweighting the graph with
// Bellman–Ford, it runs Dijkstra from each source in parallel using
// OpenMP.  Each thread maintains its own distance array to avoid false
// sharing.  See johnson_serial.c for the sequential logic.

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <omp.h>
#include "../../include/graph.h"

typedef struct AdjNode {
    int to;
    int w;
    struct AdjNode *next;
} AdjNode;

static AdjNode **build_adj(const Graph *g) {
    int V = g->V;
    AdjNode **adj = (AdjNode **)calloc(V, sizeof(AdjNode *));
    for (int i = 0; i < g->E; i++) {
        int u = g->edges[i].src;
        int v = g->edges[i].dest;
        int w = g->edges[i].weight;
        AdjNode *node = (AdjNode *)malloc(sizeof(AdjNode));
        node->to = v;
        node->w = w;
        node->next = adj[u];
        adj[u] = node;
    }
    return adj;
}

static void free_adj(int V, AdjNode **adj) {
    for (int i = 0; i < V; i++) {
        AdjNode *cur = adj[i];
        while (cur) {
            AdjNode *tmp = cur;
            cur = cur->next;
            free(tmp);
        }
    }
    free(adj);
}

static int bellman_ford(const Graph *g, int *h) {
    int V = g->V;
    int E = g->E;
    for (int i = 0; i < V; i++) h[i] = 0;
    for (int iter = 0; iter < V - 1; iter++) {
        int updated = 0;
        for (int e = 0; e < E; e++) {
            int u = g->edges[e].src;
            int v = g->edges[e].dest;
            int w = g->edges[e].weight;
            if (h[u] != INT_MAX && h[u] + w < h[v]) {
                h[v] = h[u] + w;
                updated = 1;
            }
        }
        if (!updated) break;
    }
    for (int e = 0; e < E; e++) {
        int u = g->edges[e].src;
        int v = g->edges[e].dest;
        int w = g->edges[e].weight;
        if (h[u] != INT_MAX && h[u] + w < h[v]) {
            return -1;
        }
    }
    return 0;
}

static void dijkstra(int V, AdjNode **adj, int src, int *dist) {
    char *visited = (char *)calloc(V, sizeof(char));
    for (int i = 0; i < V; i++) dist[i] = INT_MAX;
    dist[src] = 0;
    for (int iter = 0; iter < V; iter++) {
        int u = -1;
        int best = INT_MAX;
        for (int i = 0; i < V; i++) {
            if (!visited[i] && dist[i] < best) {
                best = dist[i];
                u = i;
            }
        }
        if (u == -1) break;
        visited[u] = 1;
        for (AdjNode *edge = adj[u]; edge; edge = edge->next) {
            int v = edge->to;
            int w = edge->w;
            if (dist[u] != INT_MAX && dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
            }
        }
    }
    free(visited);
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s V minWeight maxWeight [threads]\n", argv[0]);
        return 1;
    }
    int V = atoi(argv[1]);
    int minW = atoi(argv[2]);
    int maxW = atoi(argv[3]);
    int threads = (argc > 4) ? atoi(argv[4]) : 1;
    if (threads > 0) omp_set_num_threads(threads);
    Graph *g = load_graph(V, minW, maxW);
    if (!g) return 1;
    int *h = (int *)malloc(g->V * sizeof(int));
    if (bellman_ford(g, h) != 0) {
        printf("Negative weight cycle detected.\n");
        {
            char filename[256];
            snprintf(filename, sizeof(filename), "johnson_openmp__%d_%d_%d.txt", V, maxW, minW);
            FILE *fp = fopen(filename, "w");
            if (fp) {
                fprintf(fp, "Negative weight cycle detected.\n");
                fclose(fp);
            }
        }
        free_graph(g);
        free(h);
        return 0;
    }
    for (int i = 0; i < g->E; i++) {
        int u = g->edges[i].src;
        int v = g->edges[i].dest;
        int w = g->edges[i].weight;
        long newW = (long)w + h[u] - h[v];
        if (newW > INT_MAX) newW = INT_MAX;
        if (newW < INT_MIN) newW = INT_MIN;
        g->edges[i].weight = (int)newW;
    }
    AdjNode **adj = build_adj(g);
    int Vcount = g->V;
    int total = Vcount * Vcount;
    int *res = (int *)malloc(total * sizeof(int));
    #pragma omp parallel for schedule(static)
    for (int s = 0; s < Vcount; s++) {
        int *dist = (int *)malloc(Vcount * sizeof(int));
        dijkstra(Vcount, adj, s, dist);
        for (int v = 0; v < Vcount; v++) {
            long d = dist[v];
            if (d != INT_MAX) {
                long val = d - h[s] + h[v];
                if (val > INT_MAX) val = INT_MAX;
                if (val < INT_MIN) val = INT_MIN;
                res[s * Vcount + v] = (int)val;
            } else {
                res[s * Vcount + v] = INT_MAX;
            }
        }
        free(dist);
    }
    // Write the full distance matrix to an output file.  Each row of V
    // integers (or INF) is written on its own line, separated by spaces.
    {
        char filename[256];
        snprintf(filename, sizeof(filename), "johnson_openmp__%d_%d_%d.txt", Vcount, maxW, minW);
        FILE *fp = fopen(filename, "w");
        if (!fp) {
            perror("fopen");
        } else {
            for (int i = 0; i < Vcount; i++) {
                for (int j = 0; j < Vcount; j++) {
                    int d = res[i * Vcount + j];
                    if (d == INT_MAX) {
                        fprintf(fp, "INF");
                    } else {
                        fprintf(fp, "%d", d);
                    }
                    if (j < Vcount - 1) fprintf(fp, " ");
                }
                fprintf(fp, "\n");
            }
            fclose(fp);
            printf("Output saved to %s\n", filename);
        }
    }
    free(res);
    free_adj(Vcount, adj);
    free_graph(g);
    free(h);
    return 0;
}