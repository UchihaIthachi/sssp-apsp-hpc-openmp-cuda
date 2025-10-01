#ifndef GRAPH_IO_H
#define GRAPH_IO_H

#include "../include/graph.h"

/*
 * Loads a graph with the given parameters.  If the file does not exist,
 * generate_graph_file() from graph_gen.c will be called.  Returns a pointer
 * to a Graph structure or NULL on failure.
 */
Graph* load_or_generate_graph(int V, int minWeight, int maxWeight);

/*
 * Writes the distance array (length V) to a file named
 * "<algorithm>_<variant>__V_maxWeight_minWeight.txt" in the current working
 * directory.  Returns 0 on success or -1 on failure.
 */
int write_distances(const char* algorithm, const char* variant,
                    int V, int maxWeight, int minWeight,
                    const int* distance);

#endif /* GRAPH_IO_H */