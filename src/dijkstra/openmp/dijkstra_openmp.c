#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <stdbool.h>
#include <omp.h>

#include "graph.h"
#include "utils.h"
#include "graphGen.h"

/* Build adjacency lists from an edge list (same as serial version) */
static void build_adjacency(const Graph *g, int **head, int **to, int **weight, int **next)
{
    int V = g->V;
    int E = g->E;
    *head = (int*)malloc(sizeof(int) * V);
    *to = (int*)malloc(sizeof(int) * E);
    *weight = (int*)malloc(sizeof(int) * E);
    *next = (int*)malloc(sizeof(int) * E);
    if (!*head || !*to || !*weight || !*next) {
        fprintf(stderr, "Adjacency list OOM\n");
        free(*head); free(*to); free(*weight); free(*next);
        *head = NULL; // Signal error
        return;
    }
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

static inline int safe_add(int a, int b) {
    if (a == INT_MAX) return INT_MAX;
    long long s = (long long)a + (long long)b;
    if (s > INT_MAX) return INT_MAX;
    return (int)s; // non-negative weights assumed
}

static void dijkstra_openmp(int V, int * __restrict head,
                            int * __restrict to,
                            int * __restrict weight,
                            int * __restrict next,
                            int src, int * __restrict dist)
{
    bool *visited = (bool*)malloc(sizeof(bool)*V);
    for (int i=0;i<V;i++){ dist[i]=INT_MAX; visited[i]=false; }
    dist[src]=0;

    bool stop = false;
    int best;

    #pragma omp parallel shared(stop, best, dist, visited, head, to, weight, next)
    {
        for (int iter=0; iter<V && !stop; ++iter) {
            #pragma omp single
            best = INT_MAX;

            #pragma omp for reduction(min:best)
            for (int i=0; i<V; ++i) {
                if (!visited[i] && dist[i] < best) {
                    best = dist[i];
                }
            }

            #pragma omp single
            {
                if (best == INT_MAX) {
                    stop = true;
                } else {
                    int u = -1;
                    for (int i=0; i<V; ++i) {
                        if (!visited[i] && dist[i] == best) {
                            u = i;
                            break;
                        }
                    }
                    if (u != -1) {
                        visited[u] = true;
                        for (int e = head[u]; e != -1; e = next[e]) {
                            int v = to[e];
                            int w = weight[e];
                            if (dist[u] != INT_MAX) {
                                int new_dist = safe_add(dist[u], w);
                                if (new_dist < dist[v]) {
                                    dist[v] = new_dist;
                                }
                            }
                        }
                    }
                }
            }
            #pragma omp barrier
        }
    }
    free(visited);
}


int main(int argc, char **argv)
{
    if (argc < 4) {
        printf("Usage: %s <V> <min_w> <max_w> [density=0.005] [threads=0]\n", argv[0]);
        return 1;
    }
    int V = atoi(argv[1]);
    int min_w = atoi(argv[2]);
    int max_w = atoi(argv[3]);
    double density = (argc > 4) ? atof(argv[4]) : 0.005;
    int numThreads = (argc > 5) ? atoi(argv[5]) : 0;

    if (min_w < 0) {
        fprintf(stderr, "Error: Dijkstra requires non-negative weights\n");
        return 1;
    }
    if (numThreads > 0) {
        omp_set_num_threads(numThreads);
    }

    Graph *g = get_or_create_graph(V, max_w, min_w, density);
    if (!g) return 1;

    int *head, *to, *weight, *next;
    build_adjacency(g, &head, &to, &weight, &next);
    if (!head) { free_graph(g); return 1; }

    int *dist = (int*)malloc(sizeof(int) * g->V);
    if (!dist) {
        fprintf(stderr, "Out of memory\n");
        free_graph(g);
        free(head); free(to); free(weight); free(next);
        return 1;
    }

    double t0 = omp_get_wtime();
    dijkstra_openmp(g->V, head, to, weight, next, 0, dist);
    double t1 = omp_get_wtime();
    printf("[dijkstra_openmp] time: %.6f s (threads=%d)\n", t1 - t0, (numThreads ? numThreads : omp_get_max_threads()));

    save_distance_vector("dijkstra_openmp", g->V, max_w, min_w, dist, g->V, false);

    free(dist);
    free(head); free(to); free(weight); free(next);
    free_graph(g);
    return 0;
}