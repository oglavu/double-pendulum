#include "stdio.h"
#include <windows.h>

#define N 256
#define M 10000

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
    for (int i=1; i<argc; i += 2) {
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


int main(int argc, char* argv[]) {

    read_args(argc, argv);

    double4 *d_initArray, *h_initArray,
            *d_dataArray, *h_dataArray;

    size_t init_size = N * sizeof(double4);
    size_t data_size = N * M * sizeof(double4);
    
    h_initArray = (double4*)malloc(init_size);

    HANDLE h_dataFile = CreateFileA(
        "test.txt", GENERIC_READ | GENERIC_WRITE, 0, 0, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, 0
    );

    if (h_dataFile == INVALID_HANDLE_VALUE) {
        printf("Error opening file: %d\n", GetLastError());
        return -1;
    }

    SetFilePointer(h_dataFile, data_size, 0, FILE_BEGIN);
    SetEndOfFile(h_dataFile);

    // Create file mapping
    HANDLE hMap = CreateFileMapping(h_dataFile, NULL, PAGE_READWRITE, 0, data_size, NULL);
    if (!hMap) {
        printf("Error creating file mapping: %d\n", GetLastError());
        CloseHandle(h_dataFile);
        return -1;
    }

    // Map file to memory
    h_dataArray = (double4 *)MapViewOfFile(hMap, FILE_MAP_WRITE, 0, 0, data_size);
    if (!h_dataArray) {
        printf("Error mapping view of file: %d\n", GetLastError());
        CloseHandle(hMap);
        CloseHandle(h_dataFile);
        return -1;
    }


    cudaMalloc(&d_initArray, init_size);
    cudaMalloc(&d_dataArray, data_size);

    for (int i=0; i<N; i++) {
        h_initArray[i].x = 0.785;
        h_initArray[i].y = 1.570;
        h_initArray[i].z = 3.1;
        h_initArray[i].w = 1.1;
    }

    cudaMemcpy(d_initArray, h_initArray, init_size, cudaMemcpyHostToDevice);

    RK4<<<1, N>>>(d_initArray, d_dataArray);

    cudaMemcpy(h_dataArray, d_dataArray, data_size, cudaMemcpyDeviceToHost);

    // for(int i=0; i<N; i++) {
    //     printf("(%f %f %f %f) \n", 
    //         h_initArray[i].x, h_initArray[i].y, h_initArray[i].z, h_initArray[i].w);
    //     for(int j=0; j<M; j++) {
    //         double4 e = h_dataArray[i * M + j];
    //         printf("(%f %f %f %f) \n", e.x, e.y, e.z, e.w);
    //     }
    //     printf("\n");
    // }

    cudaFree(d_initArray);
    cudaFree(d_dataArray);
    free(h_initArray);
    free(h_dataArray);

    FlushViewOfFile(h_dataArray, data_size);

    // Cleanup
    UnmapViewOfFile(h_dataArray);
    CloseHandle(hMap);
    CloseHandle(h_dataFile);

    return 0;
}