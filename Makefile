
SRC_DIR := ./src
BUILD_DIR := ./build
BIN_DIR := ./bin
TST_DIR := ./test
LIB_DIR := ./lib
INC_DIR := ./inc
TMP_DIR := ./tmp

CXX  := g++
NVCC := nvcc
CURL := curl
7Z   := 7z

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

$(LIB_DIR):
	mkdir -p $(LIB_DIR)

$(INC_DIR):
	mkdir -p $(INC_DIR)

$(TMP_DIR):
	mkdir -p $(TMP_DIR)

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

# Get GLEW and GLFW
GLEW_URL := https://github.com/nigels-com/glew/releases/download/glew-2.2.0/glew-2.2.0-win32.zip
GLEW_DIR := $(TMP_DIR)/glew
GLEW_LIB := $(GLEW_DIR)/glew-2.2.0/lib/Release/x64/glew32.lib
GLEW_INC := $(GLEW_DIR)/glew-2.2.0/include/GL/*

GLFW_URL := https://github.com/glfw/glfw/releases/download/3.4/glfw-3.4.bin.WIN64.zip
GLFW_DIR := $(TMP_DIR)/glfw
GLFW_LIB := $(GLFW_DIR)/glfw-3.4.bin.WIN64/lib-mingw-w64/glfw3.dll
GLFW_INC := $(GLFW_DIR)/glfw-3.4.bin.WIN64/include/GLFW/*

download: $(TMP_DIR) $(LIB_DIR) $(INC_DIR)
	mkdir -p $(INC_DIR)/GL
	$(CURL) -L $(GLEW_URL) -o$(GLEW_DIR).zip
	$(7Z) x $(GLEW_DIR).zip -o$(GLEW_DIR) -y -aoa
	mv -f $(GLEW_LIB) $(LIB_DIR)
	mv -f $(GLEW_INC) $(INC_DIR)/GL

	mkdir -p $(INC_DIR)/GLFW
	$(CURL) -L $(GLFW_URL) -o$(GLFW_DIR).zip
	$(7Z) x $(GLFW_DIR).zip -o$(GLFW_DIR) -y -aoa
	mv -f $(GLFW_LIB) $(LIB_DIR)
	mv -f $(GLFW_INC) $(INC_DIR)/GLFW

# Clean
.PHONY: clean-all
clean-all:
	rm -rf $(BIN_DIR) $(BUILD_DIR) *.bin

.PHONY: clean-setup
clean-setup:
	rm -rf $(TMP_DIR) $(INC_DIR) $(LIB_DIR)