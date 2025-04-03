
#ifndef MATH_KERNEL_H
#define MATH_KERNEL_H

#if (REAL_TYPE == 1)
    typedef double real_t;
    typedef double4 real4_t;
#elif (REAL_TYPE == 0)
    typedef float real_t;
    typedef float4 real4_t;
#else
    #error "Unsupported real number type. Try 0 (flaot) or 1 (double)."
#endif

struct vertex_t {
    int ix;
    real_t x, y;
    
    vertex_t(int ix, real_t x, real_t y): 
        ix(ix), x(x), y(y) { }
};

namespace math_kernel {

    __global__ void calc_vertices(real4_t* d_inArray, struct vertex_t* d_outArray);

};


#endif // MATH_KERNEL_H