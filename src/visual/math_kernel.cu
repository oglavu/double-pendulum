#include "math_kernel.cuh"
#include "../physics_kernel.cuh"

__global__ void math_kernel::calc_vertices(real4_t* d_inArray, struct vertex_t* d_outArray) {

    const real_t x_A = 0.5, y_A = 0.25;

    for (u_int64_t ix_m = 0; ix_m < M; ix_m++) {

        for (u_int64_t ix_n = 0; ix_n < N; ix_n++) {
    
            u_int64_t ix = 3*(ix_m*N + ix_n);

            real_t θ1 = d_inArray[ix_m*N + ix_n].x,
                θ2 = d_inArray[ix_m*N + ix_n].y;

            real_t x_B = x_A + l1*sin(θ1);
            real_t y_B = y_A + l1*cos(θ1);

            real_t x_C = x_B + l2*sin(θ2);
            real_t y_C = y_B + l2*cos(θ2);

            d_outArray[ix]   = {ix_n, x_A, y_A};
            d_outArray[ix+1] = {ix_n, x_B, y_B};
            d_outArray[ix+2] = {ix_n, x_C, y_C};

        }

    }

}