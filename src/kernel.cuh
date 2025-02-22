
#ifndef KERNEL_H
#define KERNEL_H

namespace kernel{
    __constant__ uint32_t N, M;

    __constant__ double h;
    __constant__ double g;
    
    __constant__ double l1;
    __constant__ double l2;
    __constant__ double m1;
    __constant__ double m2;
    
    
    __device__ double fθ1(double t, double θ1, double θ2, double ω1, double ω2);
    
    __device__ double fθ2(double t, double θ1, double θ2, double ω1, double ω2);
    
    __device__ double fω1(double t, double θ1, double θ2, double ω1, double ω2);
    
    __device__ double fω2(double t, double θ1, double θ2, double ω1, double ω2);
    
    __global__ void RK4(double4 *initArray, double4 *dataArray);
        
}

#endif // KERNEL_H
