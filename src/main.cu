#include <stdio.h>
#include <windows.h>
#include "kernel.cuh"


struct args_t {
    kernel::constants_t consts = {
        1, 1,  // l1, l2
        1, 1,  // m1, m2
        0.025, // h
        9.81,  //g
    };
    char srcFilename[100];
    char dstFilename[100];

};

void read_args(int argc, char* argv[], args_t& myArgs) {

    // read args
    if (argc < 5) {
        printf("Too few args.\n Usage: main.exe <input_file> <output_file> [<options>]");
        exit(-1);
    }

    strcpy(myArgs.srcFilename, argv[1]);
    strcpy(myArgs.dstFilename, argv[2]);

    myArgs.consts.N = atol(argv[3]);
    myArgs.consts.M = atol(argv[4]);
    
    for (int i=5; i<argc; i += 2) {
        if (argv[i][0] != '-') {
            printf("Bad cmd line args"); exit(-1);
        } 
        if (strcmp(argv[i], "-l1") == 0) {
            myArgs.consts.l1 = atof(argv[i+1]);
        } else if (strcmp(argv[i], "-l2") == 0) {
            myArgs.consts.l2 = atof(argv[i+1]);
        } else if (strcmp(argv[i], "-m1") == 0) {
            myArgs.consts.m1 = atof(argv[i+1]);
        } else if (strcmp(argv[i], "-m2") == 0) {
            myArgs.consts.m2 = atof(argv[i+1]);
        } else if (strcmp(argv[i], "-g") == 0) {
            myArgs.consts.g = atof(argv[i+1]);
        } else if (strcmp(argv[i], "-m2") == 0) {
            myArgs.consts.h = atof(argv[i+1]);
        }

    }

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
        printf("Error creating read file mapping: %d\n", GetLastError());
        CloseHandle(map->file);
        return -2;
    }

    map->h_array = MapViewOfFile(map->hMap, FILE_MAP_WRITE, 0, 0, map->sz);
    if (!map->h_array) {
        printf("Error mapping view of read file: %d\n", GetLastError());
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

    SYSTEM_INFO info;
    GetSystemInfo(&info);
    const uint32_t PAGE_SIZE = info.dwAllocationGranularity;

    const uint64_t padded_sz = (map->sz + PAGE_SIZE - 1)/PAGE_SIZE * PAGE_SIZE;
    const uint32_t szL = (uint32_t)padded_sz;
    const uint32_t szH = (uint32_t)(padded_sz >> 32);

    printf("%llx %x %x", padded_sz, szL, szH);

    SetFilePointer(map->file, szL, (long*)&szH, FILE_BEGIN);
    SetEndOfFile(map->file);

    // create mapping
    map->hMap = CreateFileMapping(
        map->file, 0, PAGE_READWRITE, szH, szL, 0
    );
    if (!map->hMap) {
        printf("Error creating write file mapping: %d\n", GetLastError());
        CloseHandle(map->file);
        return -2;
    }

    map->h_array = MapViewOfFile(map->hMap, FILE_MAP_WRITE, 0, szH, szL);
    if (!map->h_array) {
        printf("Error mapping view of write file: %d\n", GetLastError());
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

    args_t myArgs;

    read_args(argc, argv, myArgs);
    kernel::set_constants(myArgs.consts);

    // open files
    double4 *d_initArray,
            *d_dataArray;

    size_t init_size = myArgs.consts.N * sizeof(double4);
    size_t data_size = myArgs.consts.N * myArgs.consts.M * sizeof(double4);

    struct mmap_t initMap, dataMap;
    initMap.sz = init_size;
    dataMap.sz = data_size;

    if (map_file_rd(myArgs.srcFilename, &initMap) < 0) {
        return -2;
    } 
    if (map_file_wr(myArgs.dstFilename, &dataMap) < 0) {
        unmap_file(&initMap);
        return -2;
    }

    // cuda op
    cudaMalloc(&d_initArray, init_size);
    cudaMalloc(&d_dataArray, data_size);

    cudaMemcpy(d_initArray, initMap.h_array, init_size, cudaMemcpyHostToDevice);

    kernel::RK4<<<1, myArgs.consts.N>>>(d_initArray, d_dataArray);

    cudaMemcpy(dataMap.h_array, d_dataArray, data_size, cudaMemcpyDeviceToHost);

    // cleanup
    cudaFree(d_initArray);
    cudaFree(d_dataArray);

    unmap_file(&initMap);
    unmap_file(&dataMap);

    return 0;
}