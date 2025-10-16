#include "graph_io.h"
#include "graphGen.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <sys/stat.h>

// Helper to check if a file exists
static bool file_exists(const char *filename) {
    struct stat buffer;
    return (stat(filename, &buffer) == 0);
}

// Helper to create a directory if it doesn't exist
static void ensure_dir_exists(const char *dir) {
    struct stat st = {0};
    if (stat(dir, &st) == -1) {
        mkdir(dir, 0700);
    }
}

Graph* get_or_create_graph(int V, int max_w, int min_w, double density) {
    ensure_dir_exists("data");
    char filename[256];
    sprintf(filename, "data/graph_%d_%d_%d_%.3f.txt", V, max_w, min_w, density);

    if (!file_exists(filename)) {
        printf("Graph file not found, generating a new one: %s\n", filename);
        if (generate_graph(filename, V, max_w, min_w, density) != 0) {
            fprintf(stderr, "Failed to generate graph file.\n");
            return NULL;
        }
    }

    FILE* fp = fopen(filename, "r");
    if (!fp) {
        perror("Failed to open graph file");
        return NULL;
    }

    int file_V, file_E;
    if (fscanf(fp, "%d %d", &file_V, &file_E) != 2) {
        fprintf(stderr, "Error: Invalid graph file header.\n");
        fclose(fp);
        return NULL;
    }

    Graph* g = (Graph*)malloc(sizeof(Graph));
    g->V = file_V;
    g->E = file_E;
    g->edges = (Edge*)malloc(g->E * sizeof(Edge));
    if (!g->edges) {
        fprintf(stderr, "Failed to allocate memory for edges.\n");
        fclose(fp);
        free(g);
        return NULL;
    }

    for (int i = 0; i < g->E; i++) {
        if (fscanf(fp, "%d %d %d", &g->edges[i].src, &g->edges[i].dest, &g->edges[i].weight) != 3) {
            fprintf(stderr, "Error reading edge #%d from file.\n", i);
            fclose(fp);
            free(g->edges);
            free(g);
            return NULL;
        }
    }
    fclose(fp);
    return g;
}

void free_graph(Graph* g) {
    if (g) {
        free(g->edges);
        free(g);
    }
}

void save_distance_vector(const char* variant, int V, int max_w, int min_w, const int* dist, int num_nodes, bool has_neg_cycle) {
    char filename[256];
    sprintf(filename, "data/output_%s__%d_%d_%d.txt", variant, V, max_w, min_w);
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        perror("Failed to open output file for distance vector");
        return;
    }

    if (has_neg_cycle) {
        fprintf(fp, "NEGATIVE_CYCLE_DETECTED\n");
    } else {
        for (int i = 0; i < num_nodes; i++) {
            if (dist[i] == INT_MAX) {
                fprintf(fp, "%d INF\n", i);
            } else {
                fprintf(fp, "%d %d\n", i, dist[i]);
            }
        }
    }
    fclose(fp);
}

void save_distance_matrix(const char* variant, int V, int max_w, int min_w, const int* dist_matrix, bool has_neg_cycle) {
    char filename[256];
    sprintf(filename, "data/output_%s__%d_%d_%d.txt", variant, V, max_w, min_w);
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        perror("Failed to open output file for distance matrix");
        return;
    }

    if (has_neg_cycle) {
        fprintf(fp, "NEGATIVE_CYCLE_DETECTED\n");
    } else {
        for (int i = 0; i < V; i++) {
            for (int j = 0; j < V; j++) {
                int d = dist_matrix[i * V + j];
                if (d >= INT_MAX / 2) { // Use a threshold for infinity
                    fprintf(fp, "INF");
                } else {
                    fprintf(fp, "%d", d);
                }
                if (j < V - 1) {
                    fprintf(fp, " ");
                }
            }
            fprintf(fp, "\n");
        }
    }
    fclose(fp);
}