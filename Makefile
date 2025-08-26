CC=gcc
NVCC=nvcc
CFLAGS=-O2 -Iinclude -Iutils
OMP=-fopenmp
BIN_DIR=bin
MKDIR_P=mkdir -p

TARGETS=BF_serial BF_openmp BF_cuda BF_hybrid

all: $(patsubst %,$(BIN_DIR)/%,$(TARGETS))

$(BIN_DIR)/BF_serial: src/BF_serial.c utils/graph_io.c utils/graphGen.c include/graph.h utils/graphGen.h
	@$(MKDIR_P) $(BIN_DIR)
	$(CC) $(CFLAGS) $^ -o $@

$(BIN_DIR)/BF_openmp: src/BF_openmp.c utils/graph_io.c utils/graphGen.c include/graph.h utils/graphGen.h
	@$(MKDIR_P) $(BIN_DIR)
	$(CC) $(CFLAGS) $(OMP) $^ -o $@

$(BIN_DIR)/BF_cuda: src/BF_cuda.cu utils/graph_io.c utils/graphGen.c include/graph.h utils/graphGen.h
	@$(MKDIR_P) $(BIN_DIR)
	$(NVCC) -O2 -Iinclude $^ -o $@

$(BIN_DIR)/BF_hybrid: src/BF_hybrid.cu utils/graph_io.c utils/graphGen.c include/graph.h utils/graphGen.h
	@$(MKDIR_P) $(BIN_DIR)
	$(NVCC) -O2 -Iinclude -Xcompiler -fopenmp $^ -o $@

clean:
	rm -rf $(BIN_DIR) reports
	rm -f $(TARGETS)
	rm -f *_output__*.txt
	rm -f data/graph_*.txt

.PHONY: all clean
