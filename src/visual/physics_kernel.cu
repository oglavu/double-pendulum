
#include "physics_kernel.cuh"

using namespace physics_kernel;

void physics_kernel::set_constants(const struct physics_kernel::constants_t& c) {
    cudaMemcpyToSymbol(l1, &c.l1, sizeof(real_t));
    cudaMemcpyToSymbol(l2, &c.l2, sizeof(real_t));
    cudaMemcpyToSymbol(m1, &c.m1, sizeof(real_t));
    cudaMemcpyToSymbol(m2, &c.m2, sizeof(real_t));
    cudaMemcpyToSymbol(g,  &c.g,  sizeof(real_t));
    cudaMemcpyToSymbol(h,  &c.h,  sizeof(real_t));
    cudaMemcpyToSymbol(N,  &c.N,  sizeof(uint32_t));
    cudaMemcpyToSymbol(M,  &c.M,  sizeof(uint32_t));
}


__device__ real_t physics_kernel::fθ1(real_t t, real_t θ1, real_t θ2, real_t ω1, real_t ω2) {
    return ω1;
}

__device__ real_t physics_kernel::fθ2(real_t t, real_t θ1, real_t θ2, real_t ω1, real_t ω2) {
    return ω2;
}

__device__ real_t physics_kernel::fω1(real_t t, real_t θ1, real_t θ2, real_t ω1, real_t ω2) {
    real_t Δθ = θ1 - θ2;

    real_t dividend = (
        -g*(2*m1+m2)*sin(θ1) 
        -m2*g*sin(θ1-2*θ2) 
        -2*sin(Δθ)*m2*(l2*ω2*ω2 + l1*cos(Δθ)*ω1*ω1)
    );

    real_t divisor = (
        l1*(2*m1 + m2 - m2*cos(2*Δθ))
    );

    return dividend / divisor;
}

__device__ real_t physics_kernel::fω2(real_t t, real_t θ1, real_t θ2, real_t ω1, real_t ω2) {
    real_t Δθ = θ1 - θ2;

    real_t dividend = (
        2*sin(Δθ)*(
            (m1+m2)*l1*ω1*ω1 + 
            g*(m1+m2)*cos(θ1) + 
            m2*l2*cos(Δθ)*ω2*ω2
        )
    );

    real_t divisor = (
        l2*(2*m1 + m2 - m2*cos(2*Δθ))
    );

    return dividend / divisor;
}

__global__ void physics_kernel::RK4(real4_t *initArray, vertex_t *dataArray) {
    // Get thread index in 2D
    real_t t = 0;
    const int record_ix = threadIdx.x;
    const real_t x_A = 0.0, y_A = 0.0;

    real_t θ1 = initArray[record_ix].x,
        θ2 = initArray[record_ix].y,
        ω1 = initArray[record_ix].z,
        ω2 = initArray[record_ix].w;

    for (int basket_ix=0; basket_ix < M; ++basket_ix) {

        real_t k1θ1 = fθ1(t, θ1, θ2, ω1, ω2);
        real_t k1θ2 = fθ2(t, θ1, θ2, ω1, ω2);
        real_t k1ω1 = fω1(t, θ1, θ2, ω1, ω2);
        real_t k1ω2 = fω2(t, θ1, θ2, ω1, ω2);
    
        real_t k2θ1 = fθ1(t + h/2, θ1 + h*k1θ1/2, θ2 + h*k1θ2/2, ω1 + h*k1ω1/2, ω2 + h*k1ω2/2);
        real_t k2θ2 = fθ2(t + h/2, θ1 + h*k1θ1/2, θ2 + h*k1θ2/2, ω1 + h*k1ω1/2, ω2 + h*k1ω2/2);
        real_t k2ω1 = fω1(t + h/2, θ1 + h*k1θ1/2, θ2 + h*k1θ2/2, ω1 + h*k1ω1/2, ω2 + h*k1ω2/2);
        real_t k2ω2 = fω2(t + h/2, θ1 + h*k1θ1/2, θ2 + h*k1θ2/2, ω1 + h*k1ω1/2, ω2 + h*k1ω2/2);
        
        real_t k3θ1 = fθ1(t + h/2, θ1 + h*k2θ1/2, θ2 + h*k2θ2/2, ω1 + h*k2ω1/2, ω2 + h*k2ω2/2);
        real_t k3θ2 = fθ2(t + h/2, θ1 + h*k2θ1/2, θ2 + h*k2θ2/2, ω1 + h*k2ω1/2, ω2 + h*k2ω2/2);
        real_t k3ω1 = fω1(t + h/2, θ1 + h*k2θ1/2, θ2 + h*k2θ2/2, ω1 + h*k2ω1/2, ω2 + h*k2ω2/2);
        real_t k3ω2 = fω2(t + h/2, θ1 + h*k2θ1/2, θ2 + h*k2θ2/2, ω1 + h*k2ω1/2, ω2 + h*k2ω2/2);
        
        real_t k4θ1 = fθ1(t + h, θ1 + h*k3θ1, θ2 + h*k3θ2, ω1 + h*k3ω1, ω2 + h*k3ω2);
        real_t k4θ2 = fθ2(t + h, θ1 + h*k3θ1, θ2 + h*k3θ2, ω1 + h*k3ω1, ω2 + h*k3ω2);
        real_t k4ω1 = fω1(t + h, θ1 + h*k3θ1, θ2 + h*k3θ2, ω1 + h*k3ω1, ω2 + h*k3ω2);
        real_t k4ω2 = fω2(t + h, θ1 + h*k3θ1, θ2 + h*k3θ2, ω1 + h*k3ω1, ω2 + h*k3ω2);
    
        θ1 = θ1 + h/6 * (k1θ1 + 2*k2θ1 + 2*k3θ1 + k4θ1);
        θ2 = θ2 + h/6 * (k1θ2 + 2*k2θ2 + 2*k3θ2 + k4θ2);
        ω1 = ω1 + h/6 * (k1ω1 + 2*k2ω1 + 2*k3ω1 + k4ω1);
        ω2 = ω2 + h/6 * (k1ω2 + 2*k2ω2 + 2*k3ω2 + k4ω2);

        int vertex_ix = 3*(basket_ix*N + record_ix);

        real_t x_B = x_A + sin(θ1)/2.0;
        real_t y_B = y_A + cos(θ1)/2.0;
        real_t x_C = x_B + sin(θ2)/2.0;
        real_t y_C = y_B + cos(θ2)/2.0;

        dataArray[vertex_ix+0] = {record_ix, x_A, -y_A};
        dataArray[vertex_ix+1] = {record_ix, x_B, -y_B};
        dataArray[vertex_ix+2] = {record_ix, x_C, -y_C};
    
    }

    initArray[record_ix].x = θ1;
    initArray[record_ix].y = θ2;
    initArray[record_ix].z = ω1;
    initArray[record_ix].w = ω2;
    
}

void physics_kernel::kernel_call(uint32_t gridSize, uint32_t blockSize, real4_t* initArray, physics_kernel::vertex_t* dataArray) {
    physics_kernel::RK4<<<gridSize, blockSize>>>(initArray, dataArray);
}
