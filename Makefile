# Makefile for SSSP/APSP HPC Project

.PHONY: all clean

# Compilers
CC := gcc
NVCC := $(shell command -v nvcc 2>/dev/null)

# Directories
BIN_DIR := bin
SRC_DIR := src
INCLUDE_DIR := include
UTILS_DIR := utils

# Flags
CFLAGS_BASE := -O2 -Wall
CFLAGS := $(CFLAGS_BASE) -I$(INCLUDE_DIR) -I$(UTILS_DIR)
OMPFLAGS := -fopenmp
LDFLAGS := -lm
NVCCFLAGS_BASE := -O2
NVCCFLAGS := $(NVCCFLAGS_BASE) -I$(INCLUDE_DIR) -I$(UTILS_DIR)

# --- Utility Objects ---
UTIL_SRCS := $(wildcard $(UTILS_DIR)/*.c)
UTIL_OBJS := $(patsubst %.c,%.o,$(UTIL_SRCS))

# --- Source Discovery ---
SERIAL_SRCS := $(wildcard $(SRC_DIR)/*/serial/*.c)
OMP_SRCS := $(wildcard $(SRC_DIR)/*/openmp/*.c)

# --- Target Definitions ---
SERIAL_TARGETS := $(patsubst %.c,$(BIN_DIR)/%,$(notdir $(SERIAL_SRCS)))
OMP_TARGETS := $(patsubst %.c,$(BIN_DIR)/%,$(notdir $(OMP_SRCS)))
TARGETS := $(SERIAL_TARGETS) $(OMP_TARGETS)

# --- CUDA Target Handling ---
ifeq ($(NVCC),)
    $(info nvcc not found. CUDA targets will be skipped.)
else
    $(info nvcc found. Adding CUDA targets.)
    CUDA_SRCS := $(wildcard $(SRC_DIR)/*/cuda/*.cu)
    HYBRID_SRCS := $(wildcard $(SRC_DIR)/*/hybrid/*.cu)

    CUDA_TARGETS := $(patsubst %.cu,$(BIN_DIR)/%,$(notdir $(CUDA_SRCS)))
    HYBRID_TARGETS := $(patsubst %.cu,$(BIN_DIR)/%,$(notdir $(HYBRID_SRCS)))
    TARGETS += $(CUDA_TARGETS) $(HYBRID_TARGETS)

    # Auto-detect GPU architecture
    ARCH_DETECT := $(shell $(NVCC) --query-gpu-info --short 2>/dev/null | grep "SM" | head -n 1 | sed 's/SM_//' || echo "")
    ifeq ($(ARCH_DETECT),)
        ARCH := 60 # Fallback architecture
    else
        ARCH := $(ARCH_DETECT)
    endif
    NVCCFLAGS += -gencode arch=compute_$(ARCH),code=sm_$(ARCH)
endif

# --- Main Rules ---
all: $(TARGETS)
	echo "Build process initiated. See output above for details."

clean:
	echo "Cleaning up..."
	rm -rf $(BIN_DIR) $(UTILS_DIR)/*.o
	echo "Done."

# --- Rule Generation ---

# Rule for utility objects
$(UTILS_DIR)/%.o: $(UTILS_DIR)/%.c
	echo "Compiling utility object $@"
	$(CC) $(CFLAGS) -c $< -o $@

# Function to define a build rule for a C file
define C_RULE
  $(eval TARGET := $(patsubst %.c,$(BIN_DIR)/%,$(notdir $(1))))
  $(eval $(TARGET): $(1) $(UTIL_OBJS) ; \
    mkdir -p $$(@D) ; \
    echo "Compiling executable $$@"; \
    $(CC) $$(CFLAGS) $(2) $$^ -o $$@ $$(LDFLAGS))
endef

# Function to define a build rule for a CUDA file
define CUDA_RULE
  $(eval TARGET := $(patsubst %.cu,$(BIN_DIR)/%,$(notdir $(1))))
  $(eval $(TARGET): $(1) $(UTIL_OBJS) ; \
    mkdir -p $$(@D) ; \
    echo "Compiling executable $$@ with NVCC"; \
    $(NVCC) $$(NVCCFLAGS) $(2) $$^ -o $$@ $$(LDFLAGS))
endef

# Generate all the rules
$(foreach src,$(SERIAL_SRCS),$(call C_RULE,$(src)))
$(foreach src,$(OMP_SRCS),$(call C_RULE,$(src),$(OMPFLAGS)))

ifneq ($(NVCC),)
    $(foreach src,$(CUDA_SRCS),$(call CUDA_RULE,$(src)))
    $(foreach src,$(HYBRID_SRCS),$(call CUDA_RULE,$(src),-Xcompiler "$(OMPFLAGS)"))
endif