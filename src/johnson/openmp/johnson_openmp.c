// OpenMP implementation of Johnson's algorithm for all-pairs shortest paths.
// Optimized based on expert code review.

#include "graph.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <stdbool.h>
#include <omp.h>

// Helper to clamp a 64-bit integer to a 32-bit integer range
static inline int clamp64_to_int(long long x) {
    if (x > INT_MAX) return INT_MAX;
    if (x < INT_MIN) return INT_MIN;
    return (int)x;
}

// Serial Bellman-Ford for reweighting with "super-source" trick
static int bellman_ford_reweight(const Graph *g, int *h) {
    const int V = g->V;
    // Super-source: h[v]=0 for all v
    for (int i = 0; i < V; ++i) h[i] = 0;

    // Relax V-1 times
    for (int it = 0; it < V - 1; ++it) {
        int changed = 0;
        for (int e = 0; e < g->E; ++e) {
            int u = g->edges[e].src, v = g->edges[e].dest, w = g->edges[e].weight;
            if (h[u] != INT_MAX) { // Avoid relaxing from an unreachable node
                long long cand = (long long)h[u] + (long long)w;
                if (cand < h[v]) { h[v] = (int)cand; changed = 1; }
            }
        }
        if (!changed) break;
    }

    // Detect any negative cycle in the whole graph
    for (int e = 0; e < g->E; ++e) {
        int u = g->edges[e].src, v = g->edges[e].dest, w = g->edges[e].weight;
        if (h[u] != INT_MAX && (long long)h[u] + (long long)w < (long long)h[v]) return -1;
    }
    return 0;
}

// Local Adjacency List for Dijkstra
typedef struct AdjNode {
    int to;
    int w;
    struct AdjNode *next;
} AdjNode;

static AdjNode** build_adj_list(const Graph *g) {
    AdjNode **adj = calloc(g->V, sizeof(AdjNode*));
    if (!adj) {
        perror("Failed to allocate adjacency list");
        return NULL;
    }
    for (int i = 0; i < g->E; i++) {
        AdjNode *node = malloc(sizeof(AdjNode));
        if (!node) {
            perror("Failed to allocate adjacency node");
            // In a real scenario, we'd free the partially built list
            return NULL;
        }
        node->to = g->edges[i].dest;
        node->w = g->edges[i].weight;
        node->next = adj[g->edges[i].src];
        adj[g->edges[i].src] = node;
    }
    return adj;
}

static void free_adj_list(AdjNode **adj, int V) {
    if (!adj) return;
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

// Local Dijkstra (serial) with safe addition
static inline int safe_add_int(int a, int b) {
    if (a == INT_MAX) return INT_MAX;
    long long s = (long long)a + (long long)b;
    return (s > INT_MAX) ? INT_MAX : (int)s;
}

static void dijkstra(int V, AdjNode **adj, int src, int *dist) {
    bool *vis = (bool*)calloc(V, sizeof(bool));
    if (!vis) return; // OOM
    for (int i=0;i<V;i++) dist[i]=INT_MAX;
    dist[src]=0;

    for (int it=0; it<V; ++it) {
        int u=-1, best=INT_MAX;
        for (int v=0; v<V; ++v)
            if (!vis[v] && dist[v] < best) { best=dist[v]; u=v; }
        if (u==-1 || best==INT_MAX) break;
        vis[u]=true;
        for (AdjNode *e=adj[u]; e; e=e->next) {
            int nd = safe_add_int(dist[u], e->w);
            if (nd < dist[e->to]) dist[e->to] = nd;
        }
    }
    free(vis);
}

void johnson_openmp(const Graph* g, int* dist_matrix, bool* has_neg_cycle) {
    const int V = g->V;
    int* h = (int*)malloc(V*sizeof(int));
    if (!h) { *has_neg_cycle=true; return; }

    if (bellman_ford_reweight(g, h) != 0) {
        *has_neg_cycle = true;
        free(h);
        return;
    }
    *has_neg_cycle = false;

    Graph rw = *g;
    rw.edges = (Edge*)malloc(g->E*sizeof(Edge));
    if (!rw.edges) { *has_neg_cycle = true; free(h); return; }

    for (int i=0;i<g->E;i++) {
        rw.edges[i] = g->edges[i];
        int u = g->edges[i].src, v = g->edges[i].dest;
        long long wprime = (long long)g->edges[i].weight + (long long)h[u] - (long long)h[v];
        rw.edges[i].weight = clamp64_to_int(wprime);
    }

    AdjNode** adj = build_adj_list(&rw);
    if (!adj) {
        *has_neg_cycle = true;
        free(h);
        free(rw.edges);
        return;
    }

    #pragma omp parallel
    {
        int *tmp_dist = (int*)malloc(sizeof(int)*V);
        if (tmp_dist) {
            #pragma omp for schedule(dynamic)
            for (int s=0; s<V; ++s) {
                dijkstra(V, adj, s, tmp_dist);
                for (int t=0; t<V; ++t) {
                    if (tmp_dist[t] != INT_MAX) {
                        long long orig = (long long)tmp_dist[t] - (long long)h[s] + (long long)h[t];
                        dist_matrix[s*V + t] = clamp64_to_int(orig);
                    } else {
                        dist_matrix[s*V + t] = INT_MAX;
                    }
                }
            }
            free(tmp_dist);
        }
    }

    free_adj_list(adj, V);
    free(rw.edges);
    free(h);
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

    int* dist_matrix = malloc(sizeof(int) * g->V * g->V);
    if (!dist_matrix) {
        perror("Failed to allocate distance matrix");
        free_graph(g);
        return 1;
    }
    bool has_neg_cycle = false;

    double t0 = omp_get_wtime();
    johnson_openmp(g, dist_matrix, &has_neg_cycle);
    double t1 = omp_get_wtime();
    printf("[johnson_openmp] time: %.6f s (threads=%d)\n", t1 - t0, (numThreads ? numThreads : omp_get_max_threads()));

    save_distance_matrix("johnson_openmp", g->V, max_w, min_w, dist_matrix, has_neg_cycle);

    free(dist_matrix);
    free_graph(g);
    return 0;
}