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
    const int V = g->V, E = g->E;
    int *dist_in = (int*)malloc(V*sizeof(int));
    int *dist_out= (int*)malloc(V*sizeof(int));
    omp_lock_t* locks = (omp_lock_t*)malloc(V*sizeof(omp_lock_t));
    if (!dist_in || !dist_out || !locks) {
        fprintf(stderr, "OOM\n");
        free(dist_in); free(dist_out); free(locks);
        return -2;
    }

    #pragma omp parallel for
    for (int i=0;i<V;i++) { dist_in[i]=INT_MAX; omp_init_lock(&locks[i]); }
    dist_in[src]=0;

    int stop=0, neg=0;
    int any_update; // Shared variable for reduction

    #pragma omp parallel
    {
        for (int it=0; it<V && !stop && !neg; ++it) {
            #pragma omp single
            {
                any_update = 0; // Reset shared flag by a single thread
            }

            #pragma omp for schedule(static)
            for (int i=0;i<V;i++) dist_out[i]=dist_in[i];

            #pragma omp for schedule(static) reduction(|:any_update)
            for (int j=0;j<E;j++) {
                int u=g->edges[j].src, v=g->edges[j].dest, w=g->edges[j].weight;
                int cand = relax_add(dist_in[u], w);
                omp_set_lock(&locks[v]);
                if (cand < dist_out[v]) {
                    dist_out[v]=cand;
                    any_update=1;
                }
                omp_unset_lock(&locks[v]);
            }

            #pragma omp single
            {
                if (it==V-1 && any_update) neg=1;
                else if (!any_update) stop=1;
                int* tmp=dist_in; dist_in=dist_out; dist_out=tmp;
            }
            #pragma omp barrier
        }

        #pragma omp for
        for (int i=0;i<V;i++) out[i]=dist_in[i];

        #pragma omp for
        for (int i=0;i<V;i++) omp_destroy_lock(&locks[i]);
    }

    free(locks); free(dist_in); free(dist_out);
    return neg? -1 : 0;
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