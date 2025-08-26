# Makefile for Bellman-Ford HPC Project

# Compiler and Flags
NVCC := nvcc
CXX := g++

# Auto-detect GPU architecture (sm_XY)
ARCH ?= $(shell \
    TMP_FILE="/tmp/get_arch.cu"; \
    BIN_FILE="/tmp/get_arch"; \
    echo '                                                        \
      #include <cuda_runtime.h>                                   \
      #include <iostream>                                         \
      int main(){                                                 \
        cudaDeviceProp prop; int dev;                             \
        cudaGetDevice(&dev);                                      \
        cudaGetDeviceProperties(&prop, dev);                      \
        std::cout << "sm_" << prop.major << prop.minor << "\n";   \
        return 0;                                                 \
      }                                                           \
    ' > $$TMP_FILE; \
    if $(NVCC) -o $$BIN_FILE $$TMP_FILE >/dev/null 2>&1; then \
        $$BIN_FILE; \
    else \
        echo "sm_50"; \
    fi; \
    rm -f $$TMP_FILE $$BIN_FILE \
)

# Extract compute capability (e.g. 75 from sm_75)
COMPUTE_CAP := $(subst sm_,,$(ARCH))

# Compiler flags
NVCCFLAGS := -O2 -std=c++14 \
             -gencode arch=compute_$(COMPUTE_CAP),code=$(ARCH)
CXXFLAGS := -O2 -std=c++14 -Wall -fopenmp
INCLUDE := -I./include -I./utils
BIN_DIR := ./bin
MKDIR_P := mkdir -p

# Targets
TARGETS := BF_serial BF_openmp BF_cuda BF_hybrid
BIN_TARGETS := $(patsubst %,$(BIN_DIR)/%,$(TARGETS))

# Default rule
all: $(BIN_TARGETS)

# Build rules
$(BIN_DIR)/BF_serial: src/BF_serial.c utils/graph_io.c utils/graphGen.c
	$(MKDIR_P) $(BIN_DIR)
	$(CXX) $(filter-out -fopenmp,$(CXXFLAGS)) $(INCLUDE) $^ -o $@

$(BIN_DIR)/BF_openmp: src/BF_openmp.c utils/graph_io.c utils/graphGen.c
	$(MKDIR_P) $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDE) $^ -o $@

$(BIN_DIR)/BF_cuda: src/BF_cuda.cu utils/graph_io.c utils/graphGen.c
	$(MKDIR_P) $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) $^ -o $@

$(BIN_DIR)/BF_hybrid: src/BF_hybrid.cu utils/graph_io.c utils/graphGen.c
	$(MKDIR_P) $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fopenmp $(INCLUDE) $^ -o $@

# Clean rule
clean:
	rm -rf $(BIN_DIR) reports
	rm -f $(TARGETS)
	rm -f *_output__*.txt
	rm -f data/graph_*.txt

.PHONY: all clean
