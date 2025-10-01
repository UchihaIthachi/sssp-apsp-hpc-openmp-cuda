#include <cstdio>
#include <cstdlib>
#include <climits>
#include <omp.h>
#include <cuda_runtime.h>
#include "graph.h"
#include "utils.h"

static inline int relax_add_host(int a, int b) {
    if (a == INT_MAX) return INT_MAX;
    long long s = (long long)a + (long long)b;
    if (s > INT_MAX) return INT_MAX;
    if (s < INT_MIN) return INT_MIN;
    return (int)s;
}

__device__ __forceinline__ int relax_add_dev(int a, int b) {
    if (a == INT_MAX) return INT_MAX;
    long long s = (long long)a + (long long)b;
    if (s > INT_MAX) return INT_MAX;
    if (s < INT_MIN) return INT_MIN;
    return (int)s;
}

__global__ void relax_edges_gpu(const Edge* __restrict__ edges, int E,
                                const int* __restrict__ d_in, int* __restrict__ d_out,
                                int* __restrict__ d_updated)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (j >= E) return;
    int u = edges[j].src;
    int v = edges[j].dest;
    int w = edges[j].weight;
    int cand = relax_add_dev(d_in[u], w);
    if (cand < d_out[v]) {
        int old = atomicMin(&d_out[v], cand);
        if (cand < old) atomicExch(d_updated, 1);
    }
}

static void gpu_check(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error %s: %s\n", msg, cudaGetErrorString(e));
        std::exit(1);
    }
}

static void partition_edges(const Graph* g, int split, Edge** cpu_edges, int* cpu_E,
                            Edge** gpu_edges, int* gpu_E) {
    int E = g->E;
    int c=0, d=0;
    for (int i=0;i<E;i++) {
        if (g->edges[i].dest < split) c++; else d++;
    }
    *cpu_edges = (Edge*)malloc(c*sizeof(Edge));
    *gpu_edges = (Edge*)malloc(d*sizeof(Edge));
    int ci=0, gi=0;
    for (int i=0;i<E;i++) {
        if (g->edges[i].dest < split) (*cpu_edges)[ci++] = g->edges[i];
        else (*gpu_edges)[gi++] = g->edges[i];
    }
    *cpu_E = c; *gpu_E = d;
}

int bellman_ford_hybrid(const Graph* g, int src, int split, int threads, int* out) {
    int V = g->V;
    int *h_in = (int*)malloc(V*sizeof(int));
    int *h_out = (int*)malloc(V*sizeof(int));
    for (int i=0;i<V;i++) h_in[i] = INT_MAX;
    h_in[src] = 0;

    Edge *cpu_edges=nullptr, *gpu_edges=nullptr;
    int cpu_E=0, gpu_E=0;
    partition_edges(g, split, &cpu_edges, &cpu_E, &gpu_edges, &gpu_E);

    Edge* d_edges=nullptr; int *d_in=nullptr, *d_out_vec=nullptr, *d_updated=nullptr;
    gpu_check(cudaMalloc(&d_edges, gpu_E*sizeof(Edge)), "malloc edges");
    gpu_check(cudaMemcpy(d_edges, gpu_edges, gpu_E*sizeof(Edge), cudaMemcpyHostToDevice), "cpy edges");
    gpu_check(cudaMalloc(&d_in, V*sizeof(int)), "malloc in");
    gpu_check(cudaMalloc(&d_out_vec, V*sizeof(int)), "malloc out");
    gpu_check(cudaMalloc(&d_updated, sizeof(int)), "malloc flag");

    dim3 blk(256), grd((gpu_E + blk.x -1)/blk.x);
    if (threads>0) omp_set_num_threads(threads);

    for (int it=0; it<V; ++it) {
        #pragma omp parallel for schedule(static)
        for (int i=0;i<V;i++) h_out[i] = h_in[i];
        gpu_check(cudaMemcpy(d_in, h_in, V*sizeof(int), cudaMemcpyHostToDevice), "cpy h_in->d_in");
        gpu_check(cudaMemcpy(d_out_vec, h_in, V*sizeof(int), cudaMemcpyHostToDevice), "cpy in->out");
        int zero=0; gpu_check(cudaMemcpy(d_updated, &zero, sizeof(int), cudaMemcpyHostToDevice), "reset flag");

        int cpu_updated = 0;

        #pragma omp parallel sections
        {
            #pragma omp section
            {
                #pragma omp parallel for schedule(static) reduction(||:cpu_updated)
                for (int j=0;j<cpu_E;j++) {
                    int u = cpu_edges[j].src;
                    int v = cpu_edges[j].dest;
                    int w = cpu_edges[j].weight;
                    int cand = relax_add_host(h_in[u], w);
                    if (cand < h_out[v]) {
                        #pragma omp critical
                        {
                            if (cand < h_out[v]) { h_out[v] = cand; cpu_updated = 1; }
                        }
                    }
                }
            }
            #pragma omp section
            {
                relax_edges_gpu<<<grd, blk>>>(d_edges, gpu_E, d_in, d_out_vec, d_updated);
                gpu_check(cudaDeviceSynchronize(), "kernel sync");
            }
        }

        gpu_check(cudaMemcpy(h_out + split, d_out_vec + split, (V - split) * sizeof(int), cudaMemcpyDeviceToHost), "cpy back");
        int gpu_updated=0; gpu_check(cudaMemcpy(&gpu_updated, d_updated, sizeof(int), cudaMemcpyDeviceToHost), "cpy flag");

        if (it == V-1 && (cpu_updated || gpu_updated)) {
            free(cpu_edges); free(gpu_edges);
            cudaFree(d_edges); cudaFree(d_in); cudaFree(d_out_vec); cudaFree(d_updated);
            free(h_in); free(h_out);
            return -1;
        }
        if (!(cpu_updated || gpu_updated)) break;

        int* tmp = h_in; h_in = h_out; h_out = tmp;
    }

    for (int i=0;i<V;i++) out[i] = h_in[i];

    free(cpu_edges); free(gpu_edges);
    cudaFree(d_edges); cudaFree(d_in); cudaFree(d_out_vec); cudaFree(d_updated);
    free(h_in); free(h_out);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        printf("Usage: %s <V> <min_w> <max_w> <split_ratio_0to1> [density=0.005] [threads]\n", argv[0]);
        return 1;
    }
    int V = atoi(argv[1]);
    int min_w = atoi(argv[2]);
    int max_w = atoi(argv[3]);
    double split_ratio = atof(argv[4]);
    double density = (argc>5)? atof(argv[5]) : 0.005;
    int threads = (argc>6)? atoi(argv[6]) : 0;

    int split = (int)(split_ratio * V);
    if (split < 0) split = 0;
    if (split > V) split = V;

    Graph* g = get_or_create_graph(V, max_w, min_w, density);
    if (!g) return 1;

    int* dist = (int*)malloc(sizeof(int)*g->V);

    double t0 = omp_get_wtime();
    int rc = bellman_ford_hybrid(g, 0, split, threads, dist);
    double t1 = omp_get_wtime();
    printf("[hybrid] time: %.6f s (split=%d/%d, threads=%d)\n", t1-t0, split, g->V, (threads?threads:omp_get_max_threads()));

    save_distance_vector("hybrid", g->V, max_w, min_w, dist, g->V, (rc==-1));
    free(dist);
    free_graph(g);
    return 0;
}
