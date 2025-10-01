#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <omp.h>
#include "graph.h"
#include "utils.h"

static inline int relax_add(int a, int b) {
    if (a == INT_MAX) return INT_MAX;
    long long s = (long long)a + (long long)b;
    if (s > INT_MAX) return INT_MAX;
    if (s < INT_MIN) return INT_MIN;
    return (int)s;
}

int bellman_ford_openmp(const Graph* g, int src, int* out) {
    int V = g->V, E = g->E;
    int *dist_in = (int*)malloc(V*sizeof(int));
    int *dist_out = (int*)malloc(V*sizeof(int));
    if (!dist_in || !dist_out) { fprintf(stderr,"OOM\n"); return -2; }
    for (int i=0;i<V;i++) dist_in[i] = INT_MAX;
    dist_in[src] = 0;

    int any_update = 0;
    for (int it=0; it<V; ++it) {
        any_update = 0;
        #pragma omp parallel for schedule(static)
        for (int i=0;i<V;i++) dist_out[i] = dist_in[i];

        #pragma omp parallel for schedule(static) reduction(||:any_update)
        for (int j=0;j<E;j++) {
            int u = g->edges[j].src;
            int v = g->edges[j].dest;
            int w = g->edges[j].weight;
            int cand = relax_add(dist_in[u], w);
            if (cand < dist_out[v]) {
                #pragma omp critical
                {
                    if (cand < dist_out[v]) { dist_out[v] = cand; any_update = 1; }
                }
            }
        }
        if (it == V-1 && any_update) { free(dist_in); free(dist_out); return -1; }
        if (!any_update) { break; }
        int* tmp = dist_in; dist_in = dist_out; dist_out = tmp;
    }
    for (int i=0;i<V;i++) out[i] = dist_in[i];
    free(dist_in); free(dist_out);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        printf("Usage: %s <V> <min_w> <max_w> [density=0.005] [threads]\n", argv[0]);
        return 1;
    }
    int V = atoi(argv[1]);
    int min_w = atoi(argv[2]);
    int max_w = atoi(argv[3]);
    double density = (argc>4)? atof(argv[4]) : 0.005;
    if (argc>5) omp_set_num_threads(atoi(argv[5]));

    Graph* g = get_or_create_graph(V, max_w, min_w, density);
    if (!g) return 1;

    int* dist = (int*)malloc(sizeof(int)*g->V);
    double t0 = omp_get_wtime();
    int rc = bellman_ford_openmp(g, 0, dist);
    double t1 = omp_get_wtime();
    printf("[openmp] time: %.6f s  (threads=%d)\n", t1-t0, omp_get_max_threads());

    save_distance_vector("openmp", g->V, max_w, min_w, dist, g->V, (rc==-1));
    free(dist);
    free_graph(g);
    return 0;
}
