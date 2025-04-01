#include <stdio.h>
#include <iostream>
#include <future>
#include "mmf.hpp"
#include "../physics_kernel.cuh"

#define TURN_SIZE (1UL<<31) // GPU: 2 GB
#define SEG_SIZE  (1UL<<31) // RAM: 2 GB

#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      std::cerr << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
      if (abort) exit(code);
   }
}

struct args_t {
    physics_kernel::constants_t consts = {
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

int main(int argc, char* argv[]) {
    // usage: main.exe <input_file> <num_instances> <num_iterations> [<options>]

    args_t myArgs;
    read_args(argc, argv, myArgs);

    // calc consts
    const size_t basket_count = myArgs.consts.N;
    const size_t basket_size  = myArgs.consts.N * sizeof(double4);
    const size_t data_size    = myArgs.consts.M * basket_size;
    const size_t seg_count    = (data_size + SEG_SIZE - 1) / SEG_SIZE;
    const size_t seg_per_turn = TURN_SIZE / SEG_SIZE;
    const size_t bs_per_seg   = SEG_SIZE / basket_size;
    const size_t turn_count   = (data_size + TURN_SIZE - 1) / TURN_SIZE;

    myArgs.consts.M = seg_per_turn * bs_per_seg;
    physics_kernel::set_constants(myArgs.consts);

    struct mmf::mmap_t initMap, *dataMaps = new mmf::mmap_t[seg_count];
    initMap.sz = basket_size;

    if (mmf::map_file_rd(myArgs.srcFilename, &initMap) < 0) {
        return -2;
    } 
    if (mmf::map_file_wr(myArgs.dstFilename, dataMaps, seg_count, SEG_SIZE) < 0) {
        unmap_file(&initMap, 1);
        return -2;
    }

    // cuda op
    std::future<void> memcpy_ftr;

    double4 *d_initArray,
            *d_dataArray;
    gpuErrChk( cudaMalloc(&d_initArray, basket_size) );
    gpuErrChk( cudaMalloc(&d_dataArray, TURN_SIZE) );
#if(SEG_SIZE==TURN_SIZE)
    double4* h_dataArray;
    gpuErrChk( cudaMallocHost(&h_dataArray, SEG_SIZE) );
#else
    cudaStream_t D2H_streams[seg_per_turn];
    for (uint64_t i=0; i<seg_per_turn; ++i)
        cudaStreamCreate(&D2H_streams[i]);
#endif
    gpuErrChk( cudaMemcpy(d_initArray, initMap.h_array, basket_size, cudaMemcpyHostToDevice) );

    for (uint64_t i=0; i<turn_count; ++i) {
        
        physics_kernel::RK4<<<1, myArgs.consts.N>>>(d_initArray, d_dataArray);
        gpuErrChk( cudaPeekAtLastError() );

        cudaDeviceSynchronize();

#if(SEG_SIZE==TURN_SIZE)
        if (i > 0) memcpy_ftr.wait();

        gpuErrChk( cudaMemcpy(h_dataArray, d_dataArray, TURN_SIZE, cudaMemcpyDeviceToHost) );
        memcpy_ftr = std::async(std::launch::async, [=]() {
            std::memcpy(dataMaps[i].h_array, h_dataArray, TURN_SIZE);
        });
#else
        for (uint64_t j=0; j<seg_per_turn; ++j) {
            void* dst = dataMaps[i*seg_per_turn + j].h_array;
            void* src = (void*)((size_t)d_dataArray + j*seg_per_turn);
            gpuErrChk( cudaMemcpyAsync(dst, src, SEG_SIZE, cudaMemcpyDeviceToHost, D2H_streams[j]) );
        }
        cudaDeviceSynchronize();
#endif
    }

#if(SEG_SIZE==TURN_SIZE)
    memcpy_ftr.wait();
#endif

    // cleanup
    cudaFree(d_initArray);
    cudaFree(d_dataArray);
#if(SEG_SIZE==TURN_SIZE)
    cudaFreeHost(h_dataArray);
#else
    for (uint64_t i=0; i<seg_per_turn; ++i)
        cudaStreamDestroy(D2H_streams[i]);
#endif

    mmf::unmap_file(&initMap, 1);
    mmf::unmap_file(dataMaps, seg_count);

    delete[] dataMaps;

    return 0;
}