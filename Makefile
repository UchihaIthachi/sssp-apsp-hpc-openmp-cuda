CC=gcc
NVCC=nvcc
CFLAGS=-O2 -Iinclude
OMP=-fopenmp

all: BF_serial BF_openmp BF_cuda BF_hybrid

BF_serial: src/BF_serial.c utils/graph_io.c utils/graphGen.c include/graph.h utils/graphGen.h
	$(CC) $(CFLAGS) src/BF_serial.c utils/graph_io.c utils/graphGen.c -o $@

BF_openmp: src/BF_openmp.c utils/graph_io.c utils/graphGen.c include/graph.h utils/graphGen.h
	$(CC) $(CFLAGS) $(OMP) src/BF_openmp.c utils/graph_io.c utils/graphGen.c -o $@

BF_cuda: src/BF_cuda.cu utils/graph_io.c utils/graphGen.c include/graph.h utils/graphGen.h
	$(NVCC) -O2 -Iinclude src/BF_cuda.cu utils/graph_io.c utils/graphGen.c -o $@

BF_hybrid: src/BF_hybrid.cu utils/graph_io.c utils/graphGen.c include/graph.h utils/graphGen.h
	$(NVCC) -O2 -Iinclude -Xcompiler -fopenmp src/BF_hybrid.cu utils/graph_io.c utils/graphGen.c -o $@

clean:
	rm -f BF_serial BF_openmp BF_cuda BF_hybrid
	rm -f *_output__*.txt
	rm -f data/graph_*.txt

.PHONY: all clean
