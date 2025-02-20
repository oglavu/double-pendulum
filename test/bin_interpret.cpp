#include <iostream>
#include <fstream>

int main(int argc, char *argv[]) {
    // Ensure the user provided the number of random numbers as an argument
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_file> <num_numbers>\n";
        return 1;
    }

    // Convert input argument to an integer
    size_t n = std::atoi(argv[2]);

    // Open the file in binary mode
    std::ifstream file(argv[1], std::ios::binary);
    if (!file) {
        std::cerr << "Error: Could not open file for writing.\n";
        return 1;
    }

    double* array = new double[n];
    file.read((char*)array, n * sizeof(double));

    // Generate and write numbers
    for (size_t i = 0; i < n; i++) {
        printf("%f%c", array[i], ((i+1)%4 ? ' ' : '\n'));
    }

    // Close the file
    file.close();
    delete[] array;
    std::cout << "\nRead " << n << " doubles from '" << argv[1] <<"'.\n";
    return 0;
}
