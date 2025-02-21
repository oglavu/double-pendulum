#include "stdio.h"
#include <windows.h>

#define N 512
#define M 100000

__constant__ double h;
__constant__ double g;

__constant__ double l1;
__constant__ double l2;
__constant__ double m1;
__constant__ double m2;

__device__ double fθ1(double t, double θ1, double θ2, double ω1, double ω2) {
    return ω1;
}

__device__ double fθ2(double t, double θ1, double θ2, double ω1, double ω2) {
    return ω2;
}

__device__ double fω1(double t, double θ1, double θ2, double ω1, double ω2) {
    double Δθ = θ1 - θ2;

    double dividend = (
        -g*(2*m1+m2)*sin(θ1) 
        -m2*g*sin(θ1-2*θ2) 
        -2*sin(Δθ)*m2*(l2*ω2*ω2 + l1*cos(Δθ)*ω1*ω1)
    );

    double divisor = (
        l1*(2*m1 + m2 - m2*cos(2*Δθ))
    );

    return dividend / divisor;
}

__device__ double fω2(double t, double θ1, double θ2, double ω1, double ω2) {
    double Δθ = θ1 - θ2;

    double dividend = (
        2*sin(Δθ)*(
            (m1+m2)*l1*ω1*ω1 + 
            g*(m1+m2)*cos(θ1) + 
            m2*l2*cos(Δθ)*ω2*ω2
        )
    );

    double divisor = (
        l2*(2*m1 + m2 - m2*cos(2*Δθ))
    );

    return dividend / divisor;
}

__global__ void RK4(double4 *initArray, double4 *dataArray) {
    // Get thread index in 2D
    double   t     = 0;
    int      ix    = threadIdx.x;

    double θ1 = initArray[0].x,
        θ2 = initArray[0].y,
        ω1 = initArray[0].z,
        ω2 = initArray[0].w;

    for (int i=0; i<M; ++i) {

        double k1θ1 = fθ1(t, θ1, θ2, ω1, ω2);
        double k1θ2 = fθ2(t, θ1, θ2, ω1, ω2);
        double k1ω1 = fω1(t, θ1, θ2, ω1, ω2);
        double k1ω2 = fω2(t, θ1, θ2, ω1, ω2);
    
        double k2θ1 = fθ1(t + h/2, θ1 + h*k1θ1/2, θ2 + h*k1θ2/2, ω1 + h*k1ω1/2, ω2 + h*k1ω2/2);
        double k2θ2 = fθ2(t + h/2, θ1 + h*k1θ1/2, θ2 + h*k1θ2/2, ω1 + h*k1ω1/2, ω2 + h*k1ω2/2);
        double k2ω1 = fω1(t + h/2, θ1 + h*k1θ1/2, θ2 + h*k1θ2/2, ω1 + h*k1ω1/2, ω2 + h*k1ω2/2);
        double k2ω2 = fω2(t + h/2, θ1 + h*k1θ1/2, θ2 + h*k1θ2/2, ω1 + h*k1ω1/2, ω2 + h*k1ω2/2);
        
        double k3θ1 = fθ1(t + h/2, θ1 + h*k2θ1/2, θ2 + h*k2θ2/2, ω1 + h*k2ω1/2, ω2 + h*k2ω2/2);
        double k3θ2 = fθ2(t + h/2, θ1 + h*k2θ1/2, θ2 + h*k2θ2/2, ω1 + h*k2ω1/2, ω2 + h*k2ω2/2);
        double k3ω1 = fω1(t + h/2, θ1 + h*k2θ1/2, θ2 + h*k2θ2/2, ω1 + h*k2ω1/2, ω2 + h*k2ω2/2);
        double k3ω2 = fω2(t + h/2, θ1 + h*k2θ1/2, θ2 + h*k2θ2/2, ω1 + h*k2ω1/2, ω2 + h*k2ω2/2);
        
        double k4θ1 = fθ1(t + h, θ1 + h*k3θ1, θ2 + h*k3θ2, ω1 + h*k3ω1, ω2 + h*k3ω2);
        double k4θ2 = fθ2(t + h, θ1 + h*k3θ1, θ2 + h*k3θ2, ω1 + h*k3ω1, ω2 + h*k3ω2);
        double k4ω1 = fω1(t + h, θ1 + h*k3θ1, θ2 + h*k3θ2, ω1 + h*k3ω1, ω2 + h*k3ω2);
        double k4ω2 = fω2(t + h, θ1 + h*k3θ1, θ2 + h*k3θ2, ω1 + h*k3ω1, ω2 + h*k3ω2);
    
        dataArray[ix*M + i].x =  θ1 = θ1 + h/6 * (k1θ1 + 2*k2θ1 + 2*k3θ1 + k4θ1);
        dataArray[ix*M + i].y =  θ2 = θ2 + h/6 * (k1θ2 + 2*k2θ2 + 2*k3θ2 + k4θ2);
        dataArray[ix*M + i].z =  ω1 = ω1 + h/6 * (k1ω1 + 2*k2ω1 + 2*k3ω1 + k4ω1);
        dataArray[ix*M + i].w =  ω2 = ω2 + h/6 * (k1ω2 + 2*k2ω2 + 2*k3ω2 + k4ω2);
    
    }
    
}

void read_args(int argc, char* argv[]) {
    double h_l1 = 1, h_l2 = 1, 
        h_m1 = 1, h_m2 = 1,
        h_g  = 9.81,
        h_h  = 0.025;
    for (int i=3; i<argc; i += 2) {
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

    cudaMemcpyToSymbol(l1, &h_l1, sizeof(double));
    cudaMemcpyToSymbol(l2, &h_l2, sizeof(double));
    cudaMemcpyToSymbol(m1, &h_m1, sizeof(double));
    cudaMemcpyToSymbol(m2, &h_m2, sizeof(double));
    cudaMemcpyToSymbol(g,  &h_g,  sizeof(double));
    cudaMemcpyToSymbol(h,  &h_h,  sizeof(double));

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
    // usage: main.exe <input_file> <output_file> [<options>]

    // read args
    if (argc < 3) {
        printf("Too few args.\n Usage: main.exe <input_file> <output_file> [<options>]");
        return -1;
    }

    char* srcFilename = (char*)malloc(100*sizeof(char)), 
        * dstFilename = (char*)malloc(100*sizeof(char));
    strcpy(srcFilename, argv[1]);
    strcpy(dstFilename, argv[2]);

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
    
    // cuda op
    cudaMalloc(&d_initArray, init_size);
    cudaMalloc(&d_dataArray, data_size);

    cudaMemcpy(d_initArray, initMap.h_array, init_size, cudaMemcpyHostToDevice);

    RK4<<<1, N>>>(d_initArray, d_dataArray);

    cudaMemcpy(dataMap.h_array, d_dataArray, data_size, cudaMemcpyDeviceToHost);

    // cleanup
    cudaFree(d_initArray);
    cudaFree(d_dataArray);

    unmap_file(&initMap);
    unmap_file(&dataMap);

    return 0;
}