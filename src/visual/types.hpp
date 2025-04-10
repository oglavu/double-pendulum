#ifndef TYPES_H
#define TYPES_H

#include "cuda_runtime.h"

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

struct constants_t {
    real_t l1, l2, m1, m2, h, g;
    uint32_t N, M;
};

struct vertex_t {
    int ix;
    real_t x, y;
};

#endif // TYPES_H