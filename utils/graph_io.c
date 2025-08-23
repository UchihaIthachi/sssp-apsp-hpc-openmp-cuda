#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "graph.h"

int generate_graph_file(const char* filename, int V, int E, int min_w, int max_w, unsigned int seed) {
    FILE* fp = fopen(filename, "w");
    if (!fp) { perror("fopen"); return -1; }
    fprintf(fp, "%d %d\n", V, E);
    srand(seed);
    for (int i=0;i<E;i++) {
        int u = rand() % V;
        int v = rand() % V;
        if (u==v) v = (v+1)%V;
        int w = min_w + rand() % (max_w - min_w + 1);
        fprintf(fp, "%d %d %d\n", u, v, w);
    }
    fclose(fp);
    return 0;
}

Graph* load_graph(const char* filename) {
    FILE* fp = fopen(filename, "r");
    if (!fp) { perror("fopen"); return NULL; }
    int V,E;
    if (fscanf(fp, "%d %d", &V, &E) != 2) { fprintf(stderr,"bad header\n"); fclose(fp); return NULL; }
    Graph* g = (Graph*)malloc(sizeof(Graph));
    g->V = V; g->E = E;
    g->edges = (Edge*)malloc(E * sizeof(Edge));
    if (!g->edges) { fprintf(stderr,"OOM\n"); fclose(fp); free(g); return NULL; }
    for (int i=0;i<E;i++) {
        int u,v,w;
        if (fscanf(fp, "%d %d %d", &u, &v, &w) != 3) { fprintf(stderr,"bad edge\n"); free(g->edges); free(g); fclose(fp); return NULL; }
        g->edges[i].src = u; g->edges[i].dest = v; g->edges[i].weight = w;
    }
    fclose(fp);
    return g;
}

Graph* ensure_and_load_graph(const char* filename, int V, int min_w, int max_w) {
    return load_graph(filename);
}

void free_graph(Graph* g) {
    if (!g) return;
    free(g->edges);
    free(g);
}
