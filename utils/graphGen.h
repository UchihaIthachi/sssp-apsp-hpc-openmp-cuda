#ifndef GRAPHGEN_H
#define GRAPHGEN_H
#include "graph.h"

#ifdef __cplusplus
extern "C" {
#endif

// Helper to build canonical name: graph_<V>_<max_w>_<min_w>.txt
void build_graph_filename(char* out, size_t out_sz, int V, int max_w, int min_w);

// Reads "graph_V_max_min.txt" based on parameters (generate if missing)
Graph* get_or_create_graph(int V, int max_w, int min_w, double density);

// Save distance vector to file with scheme: <tag>_output__V_max_min.txt
int save_distance_vector(const char* tag, int V, int max_w, int min_w, const int* dist, int n, int neg_cycle);

#ifdef __cplusplus
}
#endif
#endif
