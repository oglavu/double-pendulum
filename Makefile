
SRC_DIR := ./src
BUILD_DIR := ./build
BIN_DIR := ./bin
TST_DIR := ./test
LIB_DIR := ./lib
INC_DIR := ./inc
TMP_DIR := ./tmp

SSTORAGE_DIR := $(SRC_DIR)/storage
BSTORAGE_DIR := $(BUILD_DIR)/storage
SVISUAL_DIR  := $(SRC_DIR)/visual
BVISUAL_DIR  := $(BUILD_DIR)/visual

LIBS := glfw GL GLEW
LIBS_FLAGS := $(addprefix -l,$(LIBS))

CC   := gcc
CXX  := g++
NVCC := nvcc
CURL := curl
7Z   := 7z


REAL_TYPE ?= 0 # 0-float, 1-double

LDFLAGS   := $(LIBS_FLAGS)
NVCCFLAGS := --x cu -Wno-deprecated-gpu-targets -DREAL_TYPE=$(REAL_TYPE)
CXXFLAGS  := -x c++ -DREAL_TYPE=$(REAL_TYPE)

PYTHON := $(shell python3 --version >/dev/null 2>&1 && echo python3 || echo python)
PREPROCESSOR := $(PYTHON) ./replace_greek.py

# Find all .cu and .cuh files
SHARED_SRCS := $(wildcard $(SRC_DIR)/*.cu) $(wildcard $(SRC_DIR)/*.cpp) $(wildcard $(SRC_DIR)/*.c)
SHARED_HEDS := $(wildcard $(SRC_DIR)/*.cuh) $(wildcard $(SRC_DIR)/*.hpp) $(wildcard $(SRC_DIR)/*.h)
SHARED_OBJS := $(SHARED_SRCS:$(SRC_DIR)/%=$(BUILD_DIR)/%.o)
SHARED_PP_H := $(SHARED_HEDS:$(SRC_DIR)/%=$(BUILD_DIR)/%)

# Default target
all: storage gen int visual

# Create dirs
$(BUILD_DIR):
	mkdir -p $@

$(BIN_DIR):
	mkdir -p $@

$(LIB_DIR):
	mkdir -p $@

$(INC_DIR):
	mkdir -p $@

$(TMP_DIR):
	mkdir -p $@

$(SSTORAGE_DIR):
	mkdir -p $@

$(BSTORAGE_DIR):
	mkdir -p $@

$(SVISUAL_DIR):
	mkdir -p $@

$(BVISUAL_DIR):
	mkdir -p $@

### Build storage.exe
STORAGE_EXE := $(BIN_DIR)/storage.exe

STORAGE_SRCS = $(wildcard $(SSTORAGE_DIR)/*.cu) $(wildcard $(SSTORAGE_DIR)/*.cpp) $(wildcard $(SSTORAGE_DIR)/*.c)
STORAGE_HEDS = $(wildcard $(SSTORAGE_DIR)/*.cuh) $(wildcard $(SSTORAGE_DIR)/*.hpp) $(wildcard $(SSTORAGE_DIR)/*.h)
STORAGE_OBJS = $(STORAGE_SRCS:$(SSTORAGE_DIR)/%=$(BSTORAGE_DIR)/%.o) $(SHARED_OBJS)
STORAGE_PP_H = $(STORAGE_HEDS:$(SSTORAGE_DIR)/%=$(BSTORAGE_DIR)/%) $(SHARED_PP_H)

# Preprocess src/* and src/storage/* files
$(BUILD_DIR)/%.pp: $(SRC_DIR)/%
	$(PREPROCESSOR) $< $@

# Prepare header .cuh, .hpp and .h from /build and /build/storage
$(BUILD_DIR)/%: $(BUILD_DIR)/%.pp
	rename "s/\.pp$$//" $<

# Compile preprocessed .c, .cpp and .cu files from /build and /build/storage
$(BUILD_DIR)/%.cu.o: $(BUILD_DIR)/%.cu.pp
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
$(BUILD_DIR)/%.cpp.o: $(BUILD_DIR)/%.cpp.pp
	$(CXX) $(CXXFLAGS) -c $< -o $@
$(BUILD_DIR)/%.c.o: $(BUILD_DIR)/%.c.pp
	$(CC) $(GCCFLAGS) -c $< -o $@

# Link into final executable
$(STORAGE_EXE): $(BIN_DIR) $(BUILD_DIR) $(BSTORAGE_DIR) $(STORAGE_PP_H) $(STORAGE_OBJS) 
	$(NVCC) $(STORAGE_OBJS) -o $@ $(LDFLAGS)
	rm -rf $(BIN_DIR)/*.exp $(BIN_DIR)/*.lib

storage: $(STORAGE_EXE)

# Random generator output binary 
RG_EXE := $(BIN_DIR)/rand_generator.exe
RG_PP  := $(BUILD_DIR)/rand_generator.cpp.pp

gen: $(RG_EXE)

$(BUILD_DIR)/%.cpp.pp: $(TST_DIR)/%.cpp
	$(PREPROCESSOR) $< $@

$(RG_EXE): $(BUILD_DIR) $(BIN_DIR) $(RG_PP)
	$(CXX) $(CXXFLAGS) $(RG_PP) -o $@


# Interpreter output binary
INT_EXE := $(BIN_DIR)/bin_interpret.exe
INT_PP  := $(BUILD_DIR)/bin_interpret.cpp.pp

int: $(INT_EXE)

$(INT_EXE): $(BUILD_DIR) $(BIN_DIR) $(INT_PP)
	$(CXX) $(CXXFLAGS) $(INT_PP) -o $@


# Visual output binary

VISUAL_EXE := $(BIN_DIR)/visual.exe

VISUAL_SRCS := $(wildcard $(SVISUAL_DIR)/*.cu) $(wildcard $(SVISUAL_DIR)/*.cpp) $(wildcard $(SVISUAL_DIR)/*.c)
VISUAL_HEDS := $(wildcard $(SVISUAL_DIR)/*.cuh) $(wildcard $(SVISUAL_DIR)/*.hpp) $(wildcard $(SVISUAL_DIR)/*.h) $(SHARED_HEDS)
VISUAL_OBJS := $(VISUAL_SRCS:$(SVISUAL_DIR)/%=$(BVISUAL_DIR)/%.o) $(SHARED_OBJS)

$(VISUAL_EXE): $(BVISUAL_DIR) $(BIN_DIR) $(SHARED_PP_H) $(VISUAL_OBJS)
	$(NVCC) $(VISUAL_OBJS) -o $@ $(LDFLAGS)

visual: $(VISUAL_EXE)

# Clean
.PHONY: clean-all
clean-all:
	rm -rf $(BIN_DIR) $(BUILD_DIR) *.bin

.PHONY: clean-setup
clean-setup:
	rm -rf $(TMP_DIR) $(INC_DIR) $(LIB_DIR)

.PHONY: print-vars
print-vars:
	@echo Shared srcs: $(SHARED_SRCS)
	@echo Shared heds: $(SHARED_HEDS)
	@echo Storage srcs: $(STORAGE_SRCS)
	@echo Storage heds: $(STORAGE_HEDS)
	@echo Storage objs: $(STORAGE_OBJS)
	@echo Storage pp h: $(STORAGE_PP_H)