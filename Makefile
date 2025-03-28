
SRC_DIR := ./src
BUILD_DIR := ./build
BIN_DIR := ./bin
TST_DIR := ./test

CXX := g++
NVCC := nvcc

NVCCFLAGS := -Wno-deprecated-gpu-targets

PREPROCESSOR := python $(SRC_DIR)/replace_greek.py

# Find all .cu and .cuh files
PREH := $(patsubst $(SRC_DIR)/%.cuh, $(BUILD_DIR)/%.cuh,    $(wildcard $(SRC_DIR)/*.cuh))
OBJS  = $(patsubst $(SRC_DIR)/%.cu,  $(BUILD_DIR)/pp_%.obj, $(wildcard $(SRC_DIR)/*.cu)) 

# Output binary
EXECUTABLE := $(BIN_DIR)/main.exe

# Default target
all: $(EXECUTABLE)

# Create dirs
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Preprocess src/*.cu and src/*.cuh files
$(BUILD_DIR)/pp_%.cu: $(SRC_DIR)/%.cu	
	$(PREPROCESSOR) $< $@

$(BUILD_DIR)/%.cuh: $(SRC_DIR)/%.cuh	
	$(PREPROCESSOR) $< $@

# Compile preprocessed build/*.cu files
$(BUILD_DIR)/pp_%.obj: $(BUILD_DIR)/pp_%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Link into final executable
$(EXECUTABLE): $(BIN_DIR) $(BUILD_DIR) $(PREH) $(OBJS)
	$(NVCC) $(NVCCFLAGS) $(OBJS) -o $@


# Random generator output binary 
RG_EXE := $(BIN_DIR)/rand_generator.exe
RG_PP  := $(BUILD_DIR)/pp_rand_generator.cpp

gen: $(RG_EXE)

$(BUILD_DIR)/pp_%.cpp: */%.cpp
	$(PREPROCESSOR) $< $@

$(RG_EXE): $(BUILD_DIR) $(BIN_DIR) $(RG_PP)
	$(CXX) $(word 3, $^) -o $@


# Interpreter output binary
INT_EXE := $(BIN_DIR)/bin_interpret.exe
INT_PP  := $(BUILD_DIR)/pp_bin_interpret.cpp

int: $(INT_EXE)

$(INT_EXE): $(BUILD_DIR) $(BIN_DIR) $(INT_PP)
	$(CXX) $(word 3, $^) -o $@


# Clean
.PHONY: clean
clean:
	rm -rf $(BIN_DIR) $(BUILD_DIR) *.bin