
#ifndef PHYSICS_KERNEL_H
#define PHYSICS_KERNEL_H

#include "types.hpp"

__constant__ uint32_t N;
__constant__ uint32_t M;

__constant__ real_t h;
__constant__ real_t g;

__constant__ real_t l1;
__constant__ real_t l2;
__constant__ real_t m1;
__constant__ real_t m2;

namespace physics_kernel {

    void set_constants(const struct constants_t& c);
    
    __device__ real_t fθ1(real_t t, real_t θ1, real_t θ2, real_t ω1, real_t ω2);
    
    __device__ real_t fθ2(real_t t, real_t θ1, real_t θ2, real_t ω1, real_t ω2);
    
    __device__ real_t fω1(real_t t, real_t θ1, real_t θ2, real_t ω1, real_t ω2);
    
    __device__ real_t fω2(real_t t, real_t θ1, real_t θ2, real_t ω1, real_t ω2);
    
    __global__ void RK4(real4_t *initArray, vertex_t *dataArray);

    void kernel_call(uint32_t gridSize, uint32_t blockSize, real4_t* initArray, vertex_t* dataArray);
        
}

#endif // PHYSICS_KERNEL_H
