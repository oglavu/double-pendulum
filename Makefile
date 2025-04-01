
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

LIBS := cudart
LIBS_FLAGS := $(addprefix -l,$(LIBS))

CC   := gcc
CXX  := g++
NVCC := nvcc
CURL := curl
7Z   := 7z

LDFLAGS   := $(LIBS_FLAGS)
NVCCFLAGS := --x cu -Wno-deprecated-gpu-targets
CXXFLAGS  := -x c++

PREPROCESSOR := python3 ./replace_greek.py

# Find all .cu and .cuh files
SHARED_SRCS := $(wildcard $(SRC_DIR)/*.cu) $(wildcard $(SRC_DIR)/*.cpp) $(wildcard $(SRC_DIR)/*.c)
SHARED_HEDS := $(wildcard $(SRC_DIR)/*.cuh) $(wildcard $(SRC_DIR)/*.hpp) $(wildcard $(SRC_DIR)/*.h)

STORAGE_SRCS = $(wildcard $(SSTORAGE_DIR)/*.cu) $(wildcard $(SSTORAGE_DIR)/*.cpp) $(wildcard $(SSTORAGE_DIR)/*.c)
STORAGE_HEDS = $(wildcard $(SSTORAGE_DIR)/*.cuh) $(wildcard $(SSTORAGE_DIR)/*.hpp) $(wildcard $(SSTORAGE_DIR)/*.h)
STORAGE_OBJS = $(STORAGE_SRCS:$(SSTORAGE_DIR)/%=$(BSTORAGE_DIR)/%.o) $(SHARED_SRCS:$(SRC_DIR)/%=$(BUILD_DIR)/%.o)
STORAGE_PP_H = $(STORAGE_HEDS:$(SSTORAGE_DIR)/%=$(BSTORAGE_DIR)/%) $(SHARED_HEDS:$(SRC_DIR)/%=$(BUILD_DIR)/%)

# Output binary
STORAGE_EXE := $(BIN_DIR)/storage.exe
VISUAL_EXE  := $(BIN_DIR)/visual.exe

# Default target
all: $(STORAGE_EXE) #$(VISUAL_EXE)
#python3 ./replace_greek.py src/storage/mmf.hpp build/storage/mmf.hpp.pp


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

# Preprocess src/* and src/storage/* files
$(BUILD_DIR)/%.pp: $(SRC_DIR)/%
	$(PREPROCESSOR) $< $@
$(BSTORAGE_DIR)/%.pp: $(SSTORAGE_DIR)/%
	$(PREPROCESSOR) $< $@

# Prepare header .cuh, .hpp and .h from /build and /build/storage
$(BUILD_DIR)/%: $(BUILD_DIR)/%.pp
	rename "s/\.pp$$//" $<
$(BSTORAGE_DIR)/%: $(BSTORAGE_DIR)/%.pp
	rename "s/\.pp$$//" $<

# Compile preprocessed .c, .cpp and .cu files from /build and /build/storage
$(BUILD_DIR)/%.cu.o: $(BUILD_DIR)/%.cu.pp
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
$(BUILD_DIR)/%.cpp.o: $(BUILD_DIR)/%.cpp.pp
	$(CXX) $(CXXFLAGS) -c $< -o $@
$(BUILD_DIR)/%.c.o: $(BUILD_DIR)/%.c.pp
	$(CC) $(GCCFLAGS) -c $< -o $@

$(BSTORAGE_DIR)/%.cu.o: $(BSTORAGE_DIR)/%.cu.pp
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
$(BSTORAGE_DIR)/%.cpp.o: $(BSTORAGE_DIR)/%.cpp.pp
	$(CXX) $(CXXFLAGS) -c $< -o $@
$(BSTORAGE_DIR)/%.c.o: $(BSTORAGE_DIR)/%.c.pp
	$(CC) $(GCCFLAGS) -c $< -o $@

# Link into final executable
$(STORAGE_EXE): $(BIN_DIR) $(BUILD_DIR) $(BSTORAGE_DIR) $(STORAGE_PP_H) $(STORAGE_OBJS) 
	$(NVCC) $(STORAGE_OBJS) -o $@ $(LDFLAGS)
	rm -rf $(BIN_DIR)/*.exp $(BIN_DIR)/*.lib


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

.PHONY: print-vars
print-vars:
	@echo Shared srcs: $(SHARED_SRCS)
	@echo Shared heds: $(SHARED_HEDS)
	@echo Storage srcs: $(STORAGE_SRCS)
	@echo Storage heds: $(STORAGE_HEDS)
	@echo Storage objs: $(STORAGE_OBJS)
	@echo Storage pp h: $(STORAGE_PP_H)