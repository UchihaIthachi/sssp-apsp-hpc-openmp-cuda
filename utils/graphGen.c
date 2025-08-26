#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <limits.h>
#include <sys/stat.h>
#include "graphGen.h"
#include "graph.h"

void build_graph_filename(char* out, size_t out_sz, int V, int max_w, int min_w) {
    snprintf(out, out_sz, "data/graph_%d_%d_%d.txt", V, max_w, min_w);
}

static int file_exists(const char* path) {
    FILE* f = fopen(path, "r");
    if (f) { fclose(f); return 1; }
    return 0;
}

Graph* get_or_create_graph(int V, int max_w, int min_w, double density) {
    char fname[256];
    build_graph_filename(fname, sizeof(fname), V, max_w, min_w);
    if (!file_exists(fname)) {
        long long maxE = (long long)V * (V-1);
        long long E = (long long)(density * (double)maxE);
        if (E < V) E = V;
        printf("[gen] creating %s with ~%lld edges (density=%.3f)\n", fname, E, density);
        if (generate_graph_file(fname, V, (int)E, min_w, max_w, (unsigned int)time(NULL)) != 0) {
            fprintf(stderr, "Failed generating graph\n");
            return NULL;
        }
    }
    return load_graph(fname);
}

int save_distance_vector(const char* tag, int V, int max_w, int min_w, const int* dist, int n, int neg_cycle) {
    char outname[256];
    mkdir("reports", 0777);
    snprintf(outname, sizeof(outname), "reports/%s_output__%d_%d_%d.txt", tag, V, max_w, min_w);
    FILE* fp = fopen(outname, "w");
    if (!fp) { perror("fopen"); return -1; }
    if (neg_cycle) {
        fprintf(fp, "Negative weight cycle detected.\n");
    } else {
        for (int i = 0; i < n; i++) {
            if (dist[i] == 2147483647) fprintf(fp, "INF\n");
            else fprintf(fp, "%d\n", dist[i]);
        }
    }
    fclose(fp);
    printf("[out] wrote %s\n", outname);
    return 0;
}
