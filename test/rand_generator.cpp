#include <iostream>
#include <fstream>
#include <random>
#include <cstdlib>

int main(int argc, char *argv[]) {
    // Ensure the user provided the number of random numbers as an argument
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <output_file> <num_random_numbers>\n";
        return 1;
    }

    // Convert input argument to an integer
    size_t num_random = std::atoi(argv[2]);
    if (num_random <= 0) {
        std::cerr << "Error: Invalid number of random numbers.\n";
        return 1;
    }

    // Open the file in binary mode
    std::ofstream file(argv[1], std::ios::binary);
    if (!file) {
        std::cerr << "Error: Could not open file for writing.\n";
        return 1;
    }

    // Random number generator
    std::random_device rd;  // Seed
    std::mt19937 gen(rd()); // Mersenne Twister RNG
    std::uniform_real_distribution<double> dist(0.0f, 1.0f); // [0,1] range

    struct vec4 {
        double θ1, θ2, ω1, ω2;
    };

    struct vec4 v;
    for (size_t i = 0; i < num_random; i++) {
        v.θ1 = dist(gen) * 6.28;
        v.θ2 = dist(gen) * 3.14;
        v.ω1 = dist(gen) * 10;
        v.ω2 = dist(gen) * 10;
        file.write(reinterpret_cast<const char*>(&v), sizeof(v));
    }

    // Close the file
    file.close();
    std::cout << "Generated " << num_random << " random numbers in '" << argv[1] <<"'.\n";
    return 0;
}
