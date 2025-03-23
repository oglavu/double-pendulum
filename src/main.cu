#include <stdio.h>
#include <iostream>
#include <windows.h>
#include "kernel.cuh"


#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

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

struct basket_t {
    struct fourlet {
        double4 data[4];
    };
    struct fourlet arr[512];
};

void read_args(int argc, char* argv[], args_t& myArgs) {

    // read args
    if (argc < 4) {
        printf("Too few args.\n Usage: main.exe <input_file> <num_inst> <num_iter> [<options>]");
        exit(-1);
    }

    strcpy(myArgs.srcFilename, argv[1]);
    strcpy(myArgs.dstFilename, "./output.bin");

    myArgs.consts.N = atol(argv[2]);
    myArgs.consts.M = atol(argv[3]);
    
    for (int i=4; i<argc; i += 2) {
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
        } else if (strcmp(argv[i], "-o" ) == 0) {
            strcpy(myArgs.dstFilename, argv[i+1]);
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

int map_file_wr(
    const char* filename, 
    struct mmap_t* maps, 
    const uint16_t n, 
    const size_t seg_size
) {
    HANDLE file = CreateFileA(
        filename, GENERIC_READ | GENERIC_WRITE, 0, 0, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, 0
    );
    if (file == INVALID_HANDLE_VALUE) {
        printf("Error opening write file: %d\n", GetLastError());
        return -1;
    }

    const uint64_t sz  = n * seg_size;
    const uint32_t szL = (uint32_t)sz;
    const uint32_t szH = (uint32_t)(sz >> 32);

    SetFilePointer(file, szL, (long*)&szH, FILE_BEGIN);
    SetEndOfFile(file);

    // create mapping
    HANDLE hMap = CreateFileMapping(
        file, 0, PAGE_READWRITE, szH, szL, 0
    );
    if (!hMap) {
        printf("Error creating write file mapping: %d\n", GetLastError());
        CloseHandle(file);
        return -2;
    }

    uint64_t offset = 0;
    for (int i=0; offset < sz; ++i) {
        maps[i].file = file;
        maps[i].hMap = hMap;
        maps[i].sz = seg_size;
        maps[i].h_array = MapViewOfFile(hMap, FILE_MAP_WRITE, (offset >> 32), (uint32_t)offset, seg_size);
        
        if (!maps[i].h_array) {
            printf("Error mapping view of write file: %d\n", GetLastError());
            for (; i>0; --i) {
                UnmapViewOfFile(maps[i].h_array); 
            }
            CloseHandle(hMap);
            CloseHandle(file);
            return -1;
        }
        offset += seg_size; 
    }

    return 0;
}

void unmap_file(struct mmap_t* maps, int n) {
    for (int i=0; i<n; i++) {
        FlushViewOfFile(maps[i].h_array, maps[i].sz);
        UnmapViewOfFile(maps[i].h_array);
    }
    CloseHandle(maps[0].hMap);
    CloseHandle(maps[0].file);
}

int main(int argc, char* argv[]) {
    // usage: main.exe <input_file> <num_instances> <num_iterations> [<options>]

    args_t myArgs;
    read_args(argc, argv, myArgs);

    SYSTEM_INFO info;
    GetSystemInfo(&info);
    const uint32_t PAGE_SIZE = info.dwAllocationGranularity;
    const uint64_t SEG_SIZE  = PAGE_SIZE << 8;

    size_t free_size, _;
    gpuErrChk( cudaMemGetInfo(&free_size, &_) );

    // open files
    double4 *d_initArray,
            *d_dataArray;

    size_t init_size = myArgs.consts.N * sizeof(double4);
    size_t data_size = myArgs.consts.N * myArgs.consts.M * sizeof(double4);
    uint16_t seg_count = (data_size + SEG_SIZE - 1) / SEG_SIZE;

    struct mmap_t initMap, *dataMaps = new mmap_t[seg_count];
    initMap.sz = init_size;

    if (map_file_rd(myArgs.srcFilename, &initMap) < 0) {
        return -2;
    } 
    if (map_file_wr(myArgs.dstFilename, dataMaps, seg_count, SEG_SIZE) < 0) {
        unmap_file(&initMap, 1);
        return -2;
    }

    // cuda op

    gpuErrChk( cudaMalloc(&d_initArray, init_size) );

    size_t turn_size = 1 << 30; // free_size & (-SEG_SIZE);
    // while( cudaMalloc(&d_dataArray, turn_size) != cudaSuccess ) {
    //     // allocate whole number of SEGs
    //     turn_size = (turn_size >> 1) & (-SEG_SIZE);
    //     if (turn_size < SEG_SIZE) {
    //         printf("Not possible to allocate even one SEG\n");
    //     }
    // }
    gpuErrChk( cudaMalloc(&d_dataArray, turn_size) );
    const uint32_t sg_per_turn = turn_size / SEG_SIZE;
    const uint32_t bs_per_sg = SEG_SIZE / sizeof(basket_t);

    myArgs.consts.M = sg_per_turn * bs_per_sg;
    
    //printf("sg_per_turn: %lu\n bs_per_sg: %lu\n M: %lu\n SEG_SIZE: 0x%llx\n", sg_per_turn, bs_per_sg, myArgs.consts.M, SEG_SIZE);
    kernel::set_constants(myArgs.consts);
    for (int j=0, i=0; i<seg_count; ++i) {
        
        if (i % sg_per_turn == 0) {
            double4* tail_basket = (double4*)& ((basket_t*)dataMaps[i-1].h_array)[bs_per_sg-2];
            double4* init_basket = (j == 0 ? (double4*)initMap.h_array : tail_basket);

            gpuErrChk( cudaMemcpy(d_initArray, init_basket, init_size, cudaMemcpyHostToDevice) );

            kernel::RK4<<<1, myArgs.consts.N>>>(d_initArray, d_dataArray);

            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                std::cout << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
            }
            cudaDeviceSynchronize();
            ++j;
        }

        // void* src = d_dataArray;
        void* src = (void*)((uint64_t)d_dataArray + (i%sg_per_turn)*SEG_SIZE);
        void* dst = dataMaps[i].h_array;
        
        //printf("i: %d, src: 0x%llx, dst: 0x%llx\n", i, (uint64_t)src, (uint64_t)dst);
        //std::memset(dst, 1, SEG_SIZE);
        //gpuErrChk( cudaMemset(src, 1, SEG_SIZE) );
        gpuErrChk( cudaMemcpy(dst, src, (size_t)SEG_SIZE, cudaMemcpyDeviceToHost));
    }

    // cleanup
    cudaFree(d_initArray);
    cudaFree(d_dataArray);

    unmap_file(&initMap, 1);
    unmap_file(dataMaps, seg_count);

    delete[] dataMaps;

    return 0;
}