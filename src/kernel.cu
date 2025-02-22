
#include "kernel.cuh"

using namespace kernel;

void kernel::set_constants(const struct kernel::constants_t& c) {
    cudaMemcpyToSymbol(l1, &c.l1, sizeof(double));
    cudaMemcpyToSymbol(l2, &c.l2, sizeof(double));
    cudaMemcpyToSymbol(m1, &c.m1, sizeof(double));
    cudaMemcpyToSymbol(m2, &c.m2, sizeof(double));
    cudaMemcpyToSymbol(g,  &c.g,  sizeof(double));
    cudaMemcpyToSymbol(h,  &c.h,  sizeof(double));
    cudaMemcpyToSymbol(N,  &c.N,  sizeof(uint32_t));
    cudaMemcpyToSymbol(M,  &c.M,  sizeof(uint32_t));
}


__device__ double kernel::fθ1(double t, double θ1, double θ2, double ω1, double ω2) {
    return ω1;
}

__device__ double kernel::fθ2(double t, double θ1, double θ2, double ω1, double ω2) {
    return ω2;
}

__device__ double kernel::fω1(double t, double θ1, double θ2, double ω1, double ω2) {
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

__device__ double kernel::fω2(double t, double θ1, double θ2, double ω1, double ω2) {
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

__global__ void kernel::RK4(double4 *initArray, double4 *dataArray) {
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

    dataArray[0].x = dataArray[0].y = l1;
    dataArray[0].z = dataArray[0].w = m1;
    
}
