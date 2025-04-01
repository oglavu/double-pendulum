#include <iostream>
#include <string>
#include <fstream>

#if __linux__
#include <cstdint>
#endif

struct double4 {
    double x, y, z, w;
};

int main(int argc, char *argv[]) {
    // Ensure the user provided the number of random numbers as an argument
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_file> <num_numbers> [<options>]\n";
        return 1;
    }

    // Selection option
    // -s <step> <selected>
    // step:     splits the array in slices of _step_ count (!= 0)
    // selected: selects the index in the slice (zero-indexed)
    uint32_t selected  = 0, step = 1;
    bool selection_option = (argc == 6 && !std::string("-s").compare(argv[3]));
    if (selection_option) {
        step     = std::atoi(argv[4]);
        selected = std::atoi(argv[5]);
    }

    // Number of double4s to be printed
    uint32_t n = std::atoi(argv[2]);

    if (step == 0) {
        std::cerr << "Error: Invalid step. Must be greater than 0.\n";
        return 2;
    }

    // Read the file in binary mode
    std::ifstream file(argv[1], std::ios::binary);
    if (!file) {
        std::cerr << "Error: Could not open file for writing.\n";
        return 3;
    }

    double4* array = new double4[n];
    file.read((char*)array, n * sizeof(double4));

    if (n*sizeof(double4) > file.gcount()) {
        std::cerr << "Error: Exceeded file size.\n";
        return 4;
    }

    // Print numbers
    for (uint32_t i = selected; i < n; i += step) {
        std::cout << array[i].x <<" "<< array[i].y <<" "<< array[i].z <<" "<< array[i].w <<std::endl;
    }

    // Clean
    file.close();
    delete[] array;
    std::cout << "\nRead " << n << " double4s from '" << argv[1] <<"'.\n";
    if (selection_option)
        std::cout << "Selected row is " << selected << " and every " << step << " is printed out.\n";
    return 0;
}
