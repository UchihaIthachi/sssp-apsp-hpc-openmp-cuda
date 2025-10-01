#include "graphGen.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int generate_graph(const char* filename, int V, int max_w, int min_w, double density) {
    if (V <= 0 || density <= 0.0 || density > 1.0) {
        fprintf(stderr, "Invalid parameters for graph generation.\n");
        return -1;
    }

    long long max_E = (long long)V * (V - 1);
    long long E_ll = (long long)(density * max_E);
    if (E_ll > 2000000000) { // Practical limit to avoid excessive file size
        fprintf(stderr, "Warning: Calculated number of edges is very large. Capping at 2B.\n");
        E_ll = 2000000000;
    }
    int E = (int)E_ll;
    if (E == 0 && V > 1) E = 1; // Ensure at least one edge in a non-trivial graph

    FILE* fp = fopen(filename, "w");
    if (!fp) {
        perror("Failed to open file for graph generation");
        return -1;
    }

    fprintf(fp, "%d %d\n", V, E);
    srand(time(NULL));

    for (int i = 0; i < E; i++) {
        int u = rand() % V;
        int v = rand() % V;
        if (u == v) {
            v = (v + 1) % V;
        }
        int w = min_w + rand() % (max_w - min_w + 1);
        fprintf(fp, "%d %d %d\n", u, v, w);
    }

    fclose(fp);
    return 0;
}