
#ifndef KERNEL_H
#define KERNEL_H

__constant__ uint32_t N;
__constant__ uint32_t M;

__constant__ double h;
__constant__ double g;

__constant__ double l1;
__constant__ double l2;
__constant__ double m1;
__constant__ double m2;

namespace kernel{

    struct constants_t {
        double l1, l2, m1, m2, h, g;
        uint32_t N, M;
    };

    void set_constants(const struct kernel::constants_t& c);
    
    __device__ double fθ1(double t, double θ1, double θ2, double ω1, double ω2);
    
    __device__ double fθ2(double t, double θ1, double θ2, double ω1, double ω2);
    
    __device__ double fω1(double t, double θ1, double θ2, double ω1, double ω2);
    
    __device__ double fω2(double t, double θ1, double θ2, double ω1, double ω2);
    
    __global__ void RK4(double4 *initArray, double4 *dataArray);
        
}

#endif // KERNEL_H
