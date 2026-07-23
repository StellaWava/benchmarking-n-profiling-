/*
Radix FFT - using cooley radix-2 algorithm
this kernel is  more for stress testing...due to 
highly uncoalesced strides and intense shared memory bank conflict

exploits standard cuComplex structure 
execute: nvcc -O3 -lineinfo spmv.cu -o spmv_profile
profiling: nsys profile --stats=true ./gemm_profile
*/

#include <iostream>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cuda/cmath>

#define FFT_SIZE 256 // Must be a power of 2 for Radix-2

// 1. Device Logic: Radix-2 Cooley-Tukey FFT Kernel
__global__ void fft_radix2_kernel(cuFloatComplex *X) {
    
    // Allocate physical Shared Memory inside the SM to hold the complex signal
    __shared__ cuFloatComplex shared_X[FFT_SIZE];

    int tx = threadIdx.x; // Thread mapping within the warp

    // Phase 1: Coalesced load from VRAM into the SM's local SRAM
    if (tx < FFT_SIZE) {
        shared_X[tx] = X[tx];
    }
    // Block synchronization to guarantee data is completely staged
    __syncthreads();

    // Phase 2: Cooley-Tukey Radix-2 Butterfly Iterations
    // "size" represents the distance between paired elements in the butterfly stages
    for (int size = 2; size <= FFT_SIZE; size <<= 1) {
        int half_size = size >> 1;
        
        // Calculate the stride grouping for this specific thread
        int section = tx / half_size;
        int section_index = tx % half_size;
        int i = section * size + section_index;

        if (i < FFT_SIZE) {
            // Compute the Twiddle Factor phase angle W_N^k
            float angle = -2.0f * 3.141592653589793f * section_index / size;
            cuFloatComplex twiddle = make_cuFloatComplex(cosf(angle), sinf(angle));

            // Fetch paired elements from Shared Memory
            cuFloatComplex u = shared_X[i];
            cuFloatComplex t = cuCmul(shared_X[i + half_size], twiddle);

            // Execute Butterfly additions/subtractions in place
            shared_X[i]             = cuCadd(u, t);
            shared_X[i + half_size] = cuCsub(u, t);
        }

        // Synchronize threads before the next stage begins to prevent RAW/WAW hazards
        __syncthreads();
    }

    // Phase 3: Write the transformed frequencies back out to Global Memory VRAM
    if (tx < FFT_SIZE) {
        X[tx] = shared_X[tx];
    }
}

int main() {
    int N = FFT_SIZE;
    size_t size = N * sizeof(cuFloatComplex);

    // Host allocations (CPU Memory)
    cuFloatComplex *h_X = (cuFloatComplex*)malloc(size);

    // Initialize input data with a simple 1D sine wave signal
    for (int i = 0; i < N; i++) {
        h_X[i] = make_cuFloatComplex(sinf(2.0f * 3.141592f * i / 16.0f), 0.0f);
    }

    // 2. Host Allocation & Memory Transport to GPU VRAM
    cuFloatComplex *d_X;
    cudaMalloc((void**)&d_X, size);
    cudaMemcpy(d_X, h_X, size, cudaMemcpyHostToDevice);

    // 3. Grid Configuration: 1 Thread Block containing N threads maps onto 1 SM
    int threadsPerBlock = N;
    int numBlocks = 1;

    // Triple Chevron Gateway Launch
    fft_radix2_kernel<<<numBlocks, threadsPerBlock>>>(d_X);

    // Retrieve results and free up pools
    cudaMemcpy(h_X, d_X, size, cudaMemcpyDeviceToHost);

    std::cout << "FFT Result Output [First 2 bins]:\n";
    std::cout << "Bin 0: Real=" << cuCrealf(h_X[0]) << ", Imag=" << cuCimagf(h_X[0]) << "\n";
    std::cout << "Bin 1: Real=" << cuCrealf(h_X[1]) << ", Imag=" << cuCimagf(h_X[1]) << "\n";

    cudaFree(d_X);
    free(h_X);
    return 0;
}
