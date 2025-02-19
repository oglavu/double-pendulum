#include "stdio.h"

#define N 256
#define M 1000
__device__ double h = 0.025;
__device__ double g = 9.81;

__constant__ double   l1 = 1;
__constant__ double   l2 = 1;
__constant__ double   m1 = 1;
__constant__ double   m2 = 1;

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
    
        dataArray[i].x =  θ1 = θ1 + h/6 * (k1θ1 + 2*k2θ1 + 2*k3θ1 + k4θ1);
        dataArray[i].y =  θ2 = θ2 + h/6 * (k1θ2 + 2*k2θ2 + 2*k3θ2 + k4θ2);
        dataArray[i].z =  ω1 = ω1 + h/6 * (k1ω1 + 2*k2ω1 + 2*k3ω1 + k4ω1);
        dataArray[i].w =  ω2 = ω2 + h/6 * (k1ω2 + 2*k2ω2 + 2*k3ω2 + k4ω2);
    
    }
    
}

int main() {
    double4 *d_initArray, *h_initArray,
            *d_dataArray, *h_dataArray;

    size_t init_size = N * sizeof(double4);
    size_t data_size = N * M * sizeof(double4);
    
    h_initArray = (double4*)malloc(init_size);
    h_dataArray = (double4*)malloc(data_size);
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

    for(int i=0; i<N; i++) {
        printf("(%f %f %f %f) \n", 
            h_initArray[i].x, h_initArray[i].y, h_initArray[i].z, h_initArray[i].w);
        for(int j=0; j<M; j++) {
            double4 e = h_dataArray[i * M + j];
            printf("(%f %f %f %f) \n", e.x, e.y, e.z, e.w);
        }
        printf("\n");
    }

    cudaFree(d_initArray);
    cudaFree(d_dataArray);
    free(h_initArray);
    free(h_dataArray);

    return 0;
}