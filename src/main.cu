#include "stdio.h"

#define N 256
#define M 1001
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

__global__ void RK4(double4 *matrix, double t) {
    // Get thread index in 2D
    int      ix    = threadIdx.x * M;
    double4 *start = &matrix[ix];

    double θ1 = start[0].x,
        θ2 = start[0].y,
        ω1 = start[0].z,
        ω2 = start[0].w;

    for (int i=0; i<M-1; ++i) {

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
    
        start[i+1].x =  θ1 = θ1 + h/6 * (k1θ1 + 2*k2θ1 + 2*k3θ1 + k4θ1);
        start[i+1].y =  θ2 = θ2 + h/6 * (k1θ2 + 2*k2θ2 + 2*k3θ2 + k4θ2);
        start[i+1].z =  ω1 = ω1 + h/6 * (k1ω1 + 2*k2ω1 + 2*k3ω1 + k4ω1);
        start[i+1].w =  ω2 = ω2 + h/6 * (k1ω2 + 2*k2ω2 + 2*k3ω2 + k4ω2);

    }
    
}



int main() {
    double4 *d_matrix, *h_matrix;

    size_t matrix_size = N * M * sizeof(double4);
    
    h_matrix = (double4*)malloc(matrix_size);
    cudaMalloc(&d_matrix, matrix_size);

    for (int i=0; i<N; i++) {
        h_matrix[i*M].x = 0.785;
        h_matrix[i*M].y = 1.570;
        h_matrix[i*M].z = 3.1;
        h_matrix[i*M].w = 1.1;
    }

    cudaMemcpy(d_matrix, h_matrix, matrix_size, cudaMemcpyHostToDevice);

    RK4<<<1, N>>>(d_matrix, 0);

    cudaMemcpy(h_matrix, d_matrix, matrix_size, cudaMemcpyDeviceToHost);

    for(int i=0; i<N; i++) {
        for(int j=0; j<M; j++) {
            double4 e = h_matrix[i * M + j];
            printf("(%f %f %f %f) \n", e.x, e.y, e.z, e.w);
        }
        printf("\n");
    }

    cudaFree(d_matrix);
    free(h_matrix);

    return 0;
}