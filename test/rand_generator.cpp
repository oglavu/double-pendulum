#include <iostream>
#include <fstream>
#include <random>

#define π (3.14f)

#if (REAL_TYPE == 1)
    typedef double real_t;
#elif (REAL_TYPE == 0)
    typedef float real_t;
#else
    #error "Unsupported real number type. Try 0 (flaot) or 1 (double)."
#endif

#define GEN_GENERATORS(rd, range, gen, dist) \
    std::mt19937 gen(rd()); \
    std::uniform_real_distribution<real_t> dist(range.first, range.second);

typedef std::pair<float, float> pairf_t;

struct real4_t {
    real_t x, y, z, w;
};

struct args_t {

    pairf_t θ1_range{0, 2*π},
        θ2_range{0, 2*π},
        ω1_range{-10, 10},
        ω2_range{-10, 10};

};

int read_args(int argc, char* argv[], args_t& myArgs) {
    /*
        Reads command line arguements and parses them
        
        Error -1: no option's dash
        Error -2: invalid option
        Error -3: invalid range
    */

    float t1, t2;
    for (int i=0; i<argc; i += 3) {
        if (argv[i][0] != '-') {
            return -1;
        } 
        if (std::string("-teta1").compare(argv[i]) == 0) {
            t1 = myArgs.θ1_range.first  = atof(argv[i+1]);
            t2 = myArgs.θ1_range.second = atof(argv[i+2]);
        } else if (std::string("-teta2").compare(argv[i]) == 0) {
            t1 = myArgs.θ2_range.first  = atof(argv[i+1]);
            t2 = myArgs.θ2_range.second = atof(argv[i+2]);
        } else if (std::string("-omega1").compare(argv[i]) == 0) {
            t1 = myArgs.ω1_range.first  = atof(argv[i+1]);
            t2 = myArgs.ω1_range.second = atof(argv[i+2]);
        } else if (std::string("-omega2").compare(argv[i]) == 0) {
            t1 = myArgs.ω2_range.first  = atof(argv[i+1]);
            t2 = myArgs.ω2_range.second = atof(argv[i+2]);
        } else {
            return -2;
        }

        if (t1 > t2) return -3;

    }
    return 0;
}


int main(int argc, char *argv[]) {
    // Ensure the user provided the number of random numbers as an argument
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <output_file> <num_random_numbers> [<options>]\n";
        return 1;
    }

    // Convert input argument to an integer
    uint32_t n = std::atoi(argv[2]);

    args_t myArgs;
    if (read_args(argc-3, &argv[3], myArgs) < 0) {
        std::cerr << "Error: Bad cmd line args. Usage -teta1 {val1} {val2}, where val1 <= val2. \n";
        return 2;
    }

    // Open the file in binary mode
    std::ofstream file(argv[1], std::ios::binary);
    if (!file) {
        std::cerr << "Error: Could not open file for writing.\n";
        return 1;
    }

    // Random number generator
    std::random_device rd[4];  // Seed for Mersenne Twister RNG
    GEN_GENERATORS(rd[0], myArgs.θ1_range, θ1_gen, θ1_dist);
    GEN_GENERATORS(rd[1], myArgs.θ2_range, θ2_gen, θ2_dist);
    GEN_GENERATORS(rd[2], myArgs.ω1_range, ω1_gen, ω1_dist);
    GEN_GENERATORS(rd[3], myArgs.ω2_range, ω2_gen, ω2_dist);

    real4_t* array = new real4_t[n];

    for (uint32_t i = 0; i < n; i++) { //dist(gen)
        array[i].x = θ1_dist(θ1_gen);
        array[i].y = θ2_dist(θ2_gen);
        array[i].z = ω1_dist(ω1_gen);
        array[i].w = ω2_dist(ω2_gen);
    }

    file.write((char*)array, n * sizeof(real4_t));

    // Close the file
    file.close();
    delete[] array;
    std::string type_name = (REAL_TYPE ? "double" : "float");
    std::cout << "Generated " << n << " random " << type_name << "4s in '" << argv[1] <<"'.\n";
    return 0;
}
