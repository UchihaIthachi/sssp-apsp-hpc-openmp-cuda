#include <cstdio>
#include <cstdlib>
#include <climits>
#include <cuda_runtime.h>
#include "utils.h"
#include "graph_io.h"

__device__ __forceinline__ int relax_add_dev(int a, int b) {
    if (a == INT_MAX) return INT_MAX;
    long long s = (long long)a + (long long)b;
    if (s > INT_MAX) return INT_MAX;
    if (s < INT_MIN) return INT_MIN;
    return (int)s;
}

__global__ void init_dist(int n, int src, int* d_dist) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) d_dist[i] = (i==src)? 0 : INT_MAX;
}

__global__ void relax_edges(const Edge* __restrict__ edges, int E,
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

int bellman_ford_cuda(const Graph* g, int src, int* h_out) {
    int V = g->V, E = g->E;
    Edge* d_edges=nullptr;
    int *d_in=nullptr, *d_out=nullptr, *d_updated=nullptr;
    gpu_check(cudaMalloc(&d_edges, E*sizeof(Edge)), "malloc edges");
    gpu_check(cudaMemcpy(d_edges, g->edges, E*sizeof(Edge), cudaMemcpyHostToDevice), "cpy edges");
    gpu_check(cudaMalloc(&d_in, V*sizeof(int)), "malloc in");
    gpu_check(cudaMalloc(&d_out, V*sizeof(int)), "malloc out");
    gpu_check(cudaMalloc(&d_updated, sizeof(int)), "malloc flag");

    dim3 blk(256), grd_edges((E+blk.x-1)/blk.x), grd_nodes((V+blk.x-1)/blk.x);

    init_dist<<<grd_nodes, blk>>>(V, src, d_in);
    gpu_check(cudaDeviceSynchronize(), "init sync");

    for (int it=0; it<V; ++it) {
        gpu_check(cudaMemcpy(d_out, d_in, V*sizeof(int), cudaMemcpyDeviceToDevice), "copy in->out");
        int zero = 0; gpu_check(cudaMemcpy(d_updated, &zero, sizeof(int), cudaMemcpyHostToDevice), "reset flag");
        relax_edges<<<grd_edges, blk>>>(d_edges, E, d_in, d_out, d_updated);
        gpu_check(cudaDeviceSynchronize(), "relax sync");
        int h_flag=0; gpu_check(cudaMemcpy(&h_flag, d_updated, sizeof(int), cudaMemcpyDeviceToHost), "cpy flag");
        if (it == V-1 && h_flag) { cudaFree(d_edges); cudaFree(d_in); cudaFree(d_out); cudaFree(d_updated); return -1; }
        if (!h_flag) break;
        int* tmp = d_in; d_in = d_out; d_out = tmp;
    }
    gpu_check(cudaMemcpy(h_out, d_in, V*sizeof(int), cudaMemcpyDeviceToHost), "cpy result");

    cudaFree(d_edges); cudaFree(d_in); cudaFree(d_out); cudaFree(d_updated);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        printf("Usage: %s <V> <min_w> <max_w> [density=0.005]\n", argv[0]);
        return 1;
    }
    int V = atoi(argv[1]);
    int min_w = atoi(argv[2]);
    int max_w = atoi(argv[3]);
    double density = (argc>4)? atof(argv[4]) : 0.005;

    Graph* g = get_or_create_graph(V, max_w, min_w, density);
    if (!g) return 1;

    int* dist = (int*)malloc(sizeof(int)*g->V);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    int rc = bellman_ford_cuda(g, 0, dist);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms=0; cudaEventElapsedTime(&ms, start, stop);
    printf("[cuda] time: %.6f s\n", ms/1000.0f);

    save_distance_vector("cuda", g->V, max_w, min_w, dist, g->V, (rc==-1));
    free(dist);
    free_graph(g);
    return 0;
}
