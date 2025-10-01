#ifndef UTILS_H
#define UTILS_H

#include "graph.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// from graphGen.h
int generate_graph(const char* filename, int V, int max_w, int min_w, double density);

// from graph_io.h
Graph* get_or_create_graph(int V, int max_w, int min_w, double density);
void free_graph(Graph* g);
void save_distance_vector(const char* variant, int V, int max_w, int min_w, const int* dist, int num_nodes, bool has_neg_cycle);
void save_distance_matrix(const char* variant, int V, int max_w, int min_w, const int* dist_matrix, bool has_neg_cycle);

#ifdef __cplusplus
}
#endif

#endif /* UTILS_H */