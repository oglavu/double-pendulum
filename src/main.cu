#include "stdio.h"

#define N 128
#define M 100001
__device__ double h = 0.025;
__device__ double g = 9.81;

__constant__ double   l1 = 1;
__constant__ double   l2 = 1;
__constant__ double   m1 = 1;
__constant__ double   m2 = 1;

__device__ double fteta1(double t, double teta1, double teta2, double omega1, double omega2) {
    return omega1;
}

__device__ double fteta2(double t, double teta1, double teta2, double omega1, double omega2) {
    return omega2;
}

__device__ double fomega1(double t, double teta1, double teta2, double omega1, double omega2) {
    double deltateta = teta1 - teta2;

    double dividend = (
        -g*(2*m1+m2)*sin(teta1) 
        -m2*g*sin(teta1-2*teta2) 
        -2*sin(deltateta)*m2*(l2*omega2*omega2 + l1*cos(deltateta)*omega1*omega1)
    );

    double divisor = (
        l1*(2*m1 + m2 - m2*cos(2*deltateta))
    );

    return dividend / divisor;
}

__device__ double fomega2(double t, double teta1, double teta2, double omega1, double omega2) {
    double deltateta = teta1 - teta2;

    double dividend = (
        2*sin(deltateta)*(
            (m1+m2)*l1*omega1*omega1 + 
            g*(m1+m2)*cos(teta1) + 
            m2*l2*cos(deltateta)*omega2*omega2
        )
    );

    double divisor = (
        l2*(2*m1 + m2 - m2*cos(2*deltateta))
    );

    return dividend / divisor;
}

__global__ void RK4(double4 *matrix, double t) {
    // Get thread index in 2D
    int      ix    = threadIdx.x * M;
    double4 *start = &matrix[ix];

    double teta1 = start[0].x,
        teta2 = start[0].y,
        omega1 = start[0].z,
        omega2 = start[0].w;

    for (int i=0; i<M-1; ++i) {

        double k1teta1 = fteta1(t, teta1, teta2, omega1, omega2);
        double k1teta2 = fteta2(t, teta1, teta2, omega1, omega2);
        double k1omega1 = fomega1(t, teta1, teta2, omega1, omega2);
        double k1omega2 = fomega2(t, teta1, teta2, omega1, omega2);
    
        double k2teta1 = fteta1(t + h/2, teta1 + h*k1teta1/2, teta2 + h*k1teta2/2, omega1 + h*k1omega1/2, omega2 + h*k1omega2/2);
        double k2teta2 = fteta2(t + h/2, teta1 + h*k1teta1/2, teta2 + h*k1teta2/2, omega1 + h*k1omega1/2, omega2 + h*k1omega2/2);
        double k2omega1 = fomega1(t + h/2, teta1 + h*k1teta1/2, teta2 + h*k1teta2/2, omega1 + h*k1omega1/2, omega2 + h*k1omega2/2);
        double k2omega2 = fomega2(t + h/2, teta1 + h*k1teta1/2, teta2 + h*k1teta2/2, omega1 + h*k1omega1/2, omega2 + h*k1omega2/2);
        
        double k3teta1 = fteta1(t + h/2, teta1 + h*k2teta1/2, teta2 + h*k2teta2/2, omega1 + h*k2omega1/2, omega2 + h*k2omega2/2);
        double k3teta2 = fteta2(t + h/2, teta1 + h*k2teta1/2, teta2 + h*k2teta2/2, omega1 + h*k2omega1/2, omega2 + h*k2omega2/2);
        double k3omega1 = fomega1(t + h/2, teta1 + h*k2teta1/2, teta2 + h*k2teta2/2, omega1 + h*k2omega1/2, omega2 + h*k2omega2/2);
        double k3omega2 = fomega2(t + h/2, teta1 + h*k2teta1/2, teta2 + h*k2teta2/2, omega1 + h*k2omega1/2, omega2 + h*k2omega2/2);
        
        double k4teta1 = fteta1(t + h, teta1 + h*k3teta1, teta2 + h*k3teta2, omega1 + h*k3omega1, omega2 + h*k3omega2);
        double k4teta2 = fteta2(t + h, teta1 + h*k3teta1, teta2 + h*k3teta2, omega1 + h*k3omega1, omega2 + h*k3omega2);
        double k4omega1 = fomega1(t + h, teta1 + h*k3teta1, teta2 + h*k3teta2, omega1 + h*k3omega1, omega2 + h*k3omega2);
        double k4omega2 = fomega2(t + h, teta1 + h*k3teta1, teta2 + h*k3teta2, omega1 + h*k3omega1, omega2 + h*k3omega2);
    
        start[i+1].x =  teta1 = teta1 + h/6 * (k1teta1 + 2*k2teta1 + 2*k3teta1 + k4teta1);
        start[i+1].y =  teta2 = teta2 + h/6 * (k1teta2 + 2*k2teta2 + 2*k3teta2 + k4teta2);
        start[i+1].z =  omega1 = omega1 + h/6 * (k1omega1 + 2*k2omega1 + 2*k3omega1 + k4omega1);
        start[i+1].w =  omega2 = omega2 + h/6 * (k1omega2 + 2*k2omega2 + 2*k3omega2 + k4omega2);

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