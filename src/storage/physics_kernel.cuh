
#ifndef PHYSICS_KERNEL_H
#define PHYSICS_KERNEL_H

#if __linux__
    #include <stdint.h>
#endif

#if (REAL_TYPE == 1)
    typedef double real_t;
    typedef double4 real4_t;
#elif (REAL_TYPE == 0)
    typedef float real_t;
    typedef float4 real4_t;
#else
    #error "Unsupported real number type. Try 0 (flaot) or 1 (double)."
#endif

__constant__ uint32_t N;
__constant__ uint32_t M;

__constant__ real_t h;
__constant__ real_t g;

__constant__ real_t l1;
__constant__ real_t l2;
__constant__ real_t m1;
__constant__ real_t m2;

namespace physics_kernel {

    struct constants_t {
        real_t l1, l2, m1, m2, h, g;
        uint32_t N, M;
    };

    void set_constants(const struct physics_kernel::constants_t& c);
    
    __device__ real_t fθ1(real_t t, real_t θ1, real_t θ2, real_t ω1, real_t ω2);
    
    __device__ real_t fθ2(real_t t, real_t θ1, real_t θ2, real_t ω1, real_t ω2);
    
    __device__ real_t fω1(real_t t, real_t θ1, real_t θ2, real_t ω1, real_t ω2);
    
    __device__ real_t fω2(real_t t, real_t θ1, real_t θ2, real_t ω1, real_t ω2);
    
    __global__ void RK4(real4_t *initArray, real4_t *dataArray);

    void kernel_call(u_int32_t gridSize, u_int32_t blockSize, real4_t* initArray, real4_t* dataArray);
        
}

#endif // PHYSICS_KERNEL_H
