
SRC_DIR := ./src
BUILD_DIR := ./build
BIN_DIR := ./bin
TEST_DIR := ./test

CXX := g++

NVCCFLAG := -Wno-deprecated-gpu-targets

replacer := $(SRC_DIR)/replace_greek.py
CCBIN := "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.37.32822\bin\Hostx64\x64"

all:
	mkdir -p $(BUILD_DIR) $(BIN_DIR)
	python $(replacer) $(SRC_DIR)/kernel.cuh $(BUILD_DIR)/kernel.cuh
	python $(replacer) $(SRC_DIR)/kernel.cu $(BUILD_DIR)/kernel.cu
	python $(replacer) $(SRC_DIR)/main.cu $(BUILD_DIR)/main.cu
	nvcc -ccbin $(CCBIN) $(NVCCFLAG) -c $(BUILD_DIR)/kernel.cu  -o $(BUILD_DIR)/kernel.obj
	nvcc -ccbin $(CCBIN) $(NVCCFLAG) -c $(BUILD_DIR)/main.cu -o $(BUILD_DIR)/main.obj
	nvcc -ccbin $(CCBIN) $(NVCCFLAG) $(BUILD_DIR)/kernel.obj $(BUILD_DIR)/main.obj -o $(BIN_DIR)/main.exe 

gen:
	mkdir -p $(BUILD_DIR) $(BIN_DIR)
	python $(replacer) $(TEST_DIR)/rand_generator.cpp $(BUILD_DIR)/rand_generator.cpp
	$(CXX) $(BUILD_DIR)/rand_generator.cpp -o $(BIN_DIR)/rand_generator.exe

int:
	mkdir -p $(BUILD_DIR) $(BIN_DIR)
	python $(replacer) $(TEST_DIR)/bin_interpret.cpp $(BUILD_DIR)/bin_interpret.cpp
	$(CXX) $(BUILD_DIR)/bin_interpret.cpp -o $(BIN_DIR)/bin_interpret.exe

clean:
	rm -rf $(BIN_DIR) $(BUILD_DIR)