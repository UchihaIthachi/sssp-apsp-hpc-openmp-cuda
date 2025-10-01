#ifndef GRAPHGEN_H
#define GRAPHGEN_H

// Generates a graph with the specified parameters and saves it to a file.
// Returns 0 on success, -1 on failure.
int generate_graph(const char* filename, int V, int max_w, int min_w, double density);

#endif /* GRAPHGEN_H */