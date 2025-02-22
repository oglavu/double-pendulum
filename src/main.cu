#include <stdio.h>
#include <windows.h>
#include "kernel.cuh"

void read_args(int argc, char* argv[]) {
    double h_l1 = 1, h_l2 = 1, 
        h_m1 = 1, h_m2 = 1,
        h_g  = 9.81,
        h_h  = 0.025;
    for (int i=5; i<argc; i += 2) {
        if (argv[i][0] != '-') {
            printf("Bad cmd line args"); exit(-1);
        } 
        if (strcmp(argv[i], "-l1") == 0) {
            h_l1 = atof(argv[i+1]);
        } else if (strcmp(argv[i], "-l2") == 0) {
            h_l2 = atof(argv[i+1]);
        } else if (strcmp(argv[i], "-m1") == 0) {
            h_m1 = atof(argv[i+1]);
        } else if (strcmp(argv[i], "-m2") == 0) {
            h_m2 = atof(argv[i+1]);
        } else if (strcmp(argv[i], "-g") == 0) {
            h_g = atof(argv[i+1]);
        } else if (strcmp(argv[i], "-m2") == 0) {
            h_h = atof(argv[i+1]);
        }

    }

    cudaMemcpyToSymbol(kernel::l1, &h_l1, sizeof(double));
    cudaMemcpyToSymbol(kernel::l2, &h_l2, sizeof(double));
    cudaMemcpyToSymbol(kernel::m1, &h_m1, sizeof(double));
    cudaMemcpyToSymbol(kernel::m2, &h_m2, sizeof(double));
    cudaMemcpyToSymbol(kernel::g,  &h_g,  sizeof(double));
    cudaMemcpyToSymbol(kernel::h,  &h_h,  sizeof(double));

}

struct mmap_t {
    HANDLE file;
    HANDLE hMap;
    size_t sz;
    void*  h_array;
};

int map_file_rd(const char* filename, struct mmap_t* map) {

    map->file = CreateFileA(
        filename, GENERIC_READ | GENERIC_WRITE, 0, 0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0
    );
    if (map->file == INVALID_HANDLE_VALUE) {
        printf("Error opening read file: %d\n", GetLastError());
        return -1;
    }

    SetFilePointer(map->file, map->sz, 0, FILE_BEGIN);
    SetEndOfFile(map->file);

    // create mapping
    map->hMap = CreateFileMapping(
        map->file, 0, PAGE_READWRITE, 0, map->sz, 0
    );
    if (!map->hMap) {
        printf("Error creating file mapping: %d\n", GetLastError());
        CloseHandle(map->file);
        return -2;
    }

    map->h_array = MapViewOfFile(map->hMap, FILE_MAP_WRITE, 0, 0, map->sz);
    if (!map->h_array) {
        printf("Error mapping view of file: %d\n", GetLastError());
        CloseHandle(map->hMap);
        CloseHandle(map->file);
        return -1;
    }

    return 0;
}

int map_file_wr(const char* filename, struct mmap_t* map) {
    map->file = CreateFileA(
        filename, GENERIC_READ | GENERIC_WRITE, 0, 0, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, 0
    );
    if (map->file == INVALID_HANDLE_VALUE) {
        printf("Error opening write file: %d\n", GetLastError());
        return -1;
    }

    SetFilePointer(map->file, map->sz, 0, FILE_BEGIN);
    SetEndOfFile(map->file);

    // create mapping
    map->hMap = CreateFileMapping(
        map->file, 0, PAGE_READWRITE, 0, map->sz, 0
    );
    if (!map->hMap) {
        printf("Error creating file mapping: %d\n", GetLastError());
        CloseHandle(map->file);
        return -2;
    }

    map->h_array = MapViewOfFile(map->hMap, FILE_MAP_WRITE, 0, 0, map->sz);
    if (!map->h_array) {
        printf("Error mapping view of file: %d\n", GetLastError());
        CloseHandle(map->hMap);
        CloseHandle(map->file);
        return -1;
    }

    return 0;
}

void unmap_file(struct mmap_t* map) {
    FlushViewOfFile(map->h_array, map->sz);
    UnmapViewOfFile(map->h_array);
    CloseHandle(map->hMap);
    CloseHandle(map->file);
}

int main(int argc, char* argv[]) {
    // usage: main.exe <input_file> <output_file> <num_instances> <num_iterations> [<options>]

    // read args
    if (argc < 5) {
        printf("Too few args.\n Usage: main.exe <input_file> <output_file> [<options>]");
        return -1;
    }

    char* srcFilename = (char*)malloc(100*sizeof(char)), 
        * dstFilename = (char*)malloc(100*sizeof(char));
    strcpy(srcFilename, argv[1]);
    strcpy(dstFilename, argv[2]);

    const uint32_t N = atol(argv[3]);
    const uint32_t M = atol(argv[4]);

    read_args(argc, argv);

    // open files
    double4 *d_initArray,
            *d_dataArray;

    size_t init_size = N * sizeof(double4);
    size_t data_size = N * M * sizeof(double4);

    struct mmap_t initMap, dataMap;
    initMap.sz = init_size;
    dataMap.sz = data_size;

    if (map_file_rd(srcFilename, &initMap) < 0) {
        return -2;
    } 
    if (map_file_wr(dstFilename, &dataMap) < 0) {
        unmap_file(&initMap);
        return -2;
    }

    for (int i=0; i<1024; i++) {
        printf("%lf ", ((double4*)initMap.h_array)[i].x);
    }
    
    // cuda op
    cudaMemcpyToSymbol(kernel::N,  &N,  sizeof(uint32_t));
    cudaMemcpyToSymbol(kernel::M,  &M,  sizeof(uint32_t));

    cudaMalloc(&d_initArray, init_size);
    cudaMalloc(&d_dataArray, data_size);

    cudaMemcpy(d_initArray, initMap.h_array, init_size, cudaMemcpyHostToDevice);

    kernel::RK4<<<1, N>>>(d_initArray, d_dataArray);

    cudaMemcpy(dataMap.h_array, d_dataArray, data_size, cudaMemcpyDeviceToHost);

    // cleanup
    cudaFree(d_initArray);
    cudaFree(d_dataArray);

    unmap_file(&initMap);
    unmap_file(&dataMap);

    delete[] srcFilename, dstFilename;

    return 0;
}