#ifndef GRAPH_H
#define GRAPH_H

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int src;
    int dest;
    int weight;
} Edge;

typedef struct {
    int V;       // vertices
    int E;       // edges
    Edge* edges; // edge list
} Graph;

// Ensure graph file exists (generates if missing), then load.
// File format:
//   V E
//   u v w   (E lines)
// Weights can be negative.
Graph* ensure_and_load_graph(const char* filename, int V, int min_w, int max_w);

// Load graph (edge list) from filename. Returns heap-allocated Graph*.
Graph* load_graph(const char* filename);

// Generate and save random directed graph (Erdos-Renyi style) with weights [min_w, max_w].
int generate_graph_file(const char* filename, int V, int E, int min_w, int max_w, unsigned int seed);

// Free Graph*
void free_graph(Graph* g);

#ifdef __cplusplus
}
#endif

#endif // GRAPH_H
