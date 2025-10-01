#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>
#include "graph.h"
#include "graph_io.h"

static inline int relax_add(int a, int b) {
    if (a == INT_MAX) return INT_MAX;
    long long s = (long long)a + (long long)b;
    if (s > INT_MAX) return INT_MAX;
    if (s < INT_MIN) return INT_MIN;
    return (int)s;
}

int bellman_ford_serial(const Graph* g, int src, int* dist) {
    int V = g->V, E = g->E;
    for (int i=0;i<V;i++) dist[i] = INT_MAX;
    dist[src] = 0;
    int updated = 0;
    for (int it=0; it<V; ++it) {
        updated = 0;
        for (int j=0;j<E;j++) {
            int u = g->edges[j].src;
            int v = g->edges[j].dest;
            int w = g->edges[j].weight;
            int cand = relax_add(dist[u], w);
            if (cand < dist[v]) {
                if (it == V-1) return -1;
                dist[v] = cand;
                updated = 1;
            }
        }
        if (!updated) break;
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        printf("Usage: %s <V> <min_w> <max_w> [density=0.005]\n", argv[0]);
        return 1;
    }
    int V = atoi(argv[1]);
    int min_w = atoi(argv[2]);
    int max_w = atoi(argv[3]);
    double density = (argc>4)? atof(argv[4]) : 0.005;

    Graph* g = get_or_create_graph(V, max_w, min_w, density);
    if (!g) return 1;

    int* dist = (int*)malloc(sizeof(int)*g->V);
    clock_t t0 = clock();
    int rc = bellman_ford_serial(g, 0, dist);
    clock_t t1 = clock();
    double secs = (double)(t1 - t0) / CLOCKS_PER_SEC;
    printf("[serial] time: %.6f s\n", secs);

    save_distance_vector("serial", g->V, max_w, min_w, dist, g->V, (rc==-1));
    free(dist);
    free_graph(g);
    return 0;
}
