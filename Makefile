
SRC_DIR := ./src
BUILD_DIR := ./build
BIN_DIR := ./bin
TEST_DIR := ./test

CXX := g++

all:
	python $(SRC_DIR)/replace_greek.py $(SRC_DIR)/main.cu $(BUILD_DIR)/main.cu
	nvcc $(BUILD_DIR)/main.cu -o $(BIN_DIR)/main.exe -ccbin "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.37.32822\bin\Hostx64\x64"

gen:
	python $(SRC_DIR)/replace_greek.py $(TEST_DIR)/rand_generator.cpp $(BUILD_DIR)/rand_generator.cpp
	g++ $(BUILD_DIR)/rand_generator.cpp -o $(BIN_DIR)/rand_generator.exe

int:
	g++ $(TEST_DIR)/bin_interpret.cpp -o $(BIN_DIR)/bin_interpret.exe