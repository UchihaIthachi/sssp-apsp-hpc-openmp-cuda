// Placeholder CUDA implementation of Johnson's algorithm.
//
// A full GPU implementation of Johnson's algorithm requires implementing
// a parallel Bellman–Ford followed by many parallel Dijkstra or SSSP
// kernels, which is beyond the scope of this skeleton.  This stub
// simply loads the graph and reports that the CUDA version is not yet
// implemented.

#include <cstdio>
#include "../../include/graph.h"
#include <climits>

int main(int argc, char **argv) {
    if (argc < 4) {
        printf("Usage: %s V minWeight maxWeight\n", argv[0]);
        return 1;
    }
    int V = atoi(argv[1]);
    int minW = atoi(argv[2]);
    int maxW = atoi(argv[3]);
    Graph *g = load_graph(V, minW, maxW);
    if (!g) return 1;
    printf("johnson_cuda: not yet implemented.\n");
    // Write an empty V×V result (all INF) to the output file.  Each row
    // contains V entries separated by spaces.
    int total = V * V;
    int *res = (int *)malloc(total * sizeof(int));
    for (int i = 0; i < total; i++) res[i] = INT_MAX;
    {
        char filename[256];
        snprintf(filename, sizeof(filename), "johnson_cuda__%d_%d_%d.txt", V, maxW, minW);
        FILE *fp = fopen(filename, "w");
        if (fp) {
            for (int i = 0; i < V; i++) {
                for (int j = 0; j < V; j++) {
                    int d = res[i * V + j];
                    if (d == INT_MAX) {
                        fprintf(fp, "INF");
                    } else {
                        fprintf(fp, "%d", d);
                    }
                    if (j < V - 1) fprintf(fp, " ");
                }
                fprintf(fp, "\n");
            }
            fclose(fp);
            printf("Output saved to %s\n", filename);
        }
    }
    free(res);
    free_graph(g);
    return 0;
}