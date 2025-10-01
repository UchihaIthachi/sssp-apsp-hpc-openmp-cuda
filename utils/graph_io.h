#ifndef GRAPH_IO_H
#define GRAPH_IO_H

#include "graph.h"
#include <stdbool.h>

// Attempts to load a graph from a standard file path. If the file doesn't
// exist, it generates a new graph, saves it, and then loads it.
Graph* get_or_create_graph(int V, int max_w, int min_w, double density);

// Frees the memory allocated for the graph structure.
void free_graph(Graph* g);

// Saves a 1D distance vector (for SSSP algorithms) to a file.
void save_distance_vector(const char* variant, int V, int max_w, int min_w, const int* dist, int num_nodes, bool has_neg_cycle);

// Saves a 2D distance matrix (for APSP algorithms) to a file.
// If has_neg_cycle is true, it writes a message instead of the matrix.
void save_distance_matrix(const char* variant, int V, int max_w, int min_w, const int* dist_matrix, bool has_neg_cycle);

#endif /* GRAPH_IO_H */