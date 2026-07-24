/*
(C = \alpha (A*B) + beta C). 
GEMM is 2D hence fully exploits the threadblock. 
- using shared memory , test shared memory bank conflict
- - unified memory, 
*/


#include <iostream>
#include <cuda/cmath>
#include <iostream>
#include <chrono>
#include <vector>
#include <chrono>
#include <fstream>
#include <cstdlib>
#include <cmath>

// Define compile-time physical tile dimensions inside the SM SRAM
#define TILE_DIM 32

// 1. Device Logic: Tiled Matrix Multiplication Kernel
__global__ void gemm_tiled_kernel(const float *A, const float *B, float *C, int N) {
    
    // Allocate physical Shared Memory arrays inside this specific SM
    // Compilers must know this size at compile time to map SM resources
    __shared__ float tile_A[TILE_DIM][TILE_DIM];
    __shared__ float tile_B[TILE_DIM][TILE_DIM];

    // Thread mapping to a 2D coordinate grid space
    int tx = threadIdx.x; 
    int ty = threadIdx.y;

    // Identify row and column indices for the global output matrix C
    int row = blockIdx.y * TILE_DIM + ty;
    int col = blockIdx.x * TILE_DIM + tx;

    float value = 0.0f;

    // Loop across all tiles needed to compute this cell (Data Movement Phase)
    for (int phase = 0; phase < cuda::ceil_div(N, TILE_DIM); ++phase) {
        
        // --- 1. Cooperative Load into Shared Memory Tiles ---
        // Global Memory Coalescing Check: Threads read contiguous rows/cols
        if (row < N && (phase * TILE_DIM + tx) < N) {
            tile_A[ty][tx] = A[row * N + (phase * TILE_DIM + tx)];
        } else {
            tile_A[ty][tx] = 0.0f; // Thread masking/padding for boundary mismatch
        }

        if (col < N && (phase * TILE_DIM + ty) < N) {
            tile_B[ty][tx] = B[(phase * TILE_DIM + ty) * N + col];
        } else {
            tile_B[ty][tx] = 0.0f;
        }

        // Synchronize threads inside the Block to ensure the entire tile is loaded
        __syncthreads();

        // --- 2. Compute Phase on On-Chip SRAM ---
        // Threads execute index math over the local immutable tiles
        for (int k = 0; k < TILE_DIM; ++k) {
            value += tile_A[ty][k] * tile_B[k][tx];
        }

        // Synchronize again to prevent data hazards (WAR) before the next phase overwrites the tiles
        __syncthreads();
    }

    // Write final accumulated value out to Global Memory VRAM
    if (row < N && col < N) {
        C[row * N + col] = value;
    }
}

int main() {
    // Square matrix configuration size N x N
    int N = 512; 
    size_t size = N * N * sizeof(float);

    // Host allocations (CPU Memory)
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Fill matrices with dummy values
    for (int i = 0; i < N*N; i++) { h_A[i] = 1.0f; h_B[i] = 2.0f; }

    // 2. Host Allocation & Memory Transport to GPU VRAM
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 3. 2D Grid Configuration and Launch Geometry Shapes (Rule 3)
    dim3 threadsPerBlock(TILE_DIM, TILE_DIM); // 32x32 = 1024 threads
    dim3 numBlocks(cuda::ceil_div(N, TILE_DIM), cuda::ceil_div(N, TILE_DIM));

    // kernal warm up
    gemm_tiled_kernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    //Active Timing Block
    const auto start = std::chrono::steady_clock::now();
    //launch kernel 
    gemm_tiled_kernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);

    //sync gpu
    cudaDeviceSynchronize();
    //measure time
    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = end - start;
    const double seconds = elapsed.count();

    // Retrieve results
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    //get results
    std::cout << "GEMM matmul C: [" << d_C <<"]\n";

    // 7. Calculate Metrics
    constexpr double flops_per_element = 2.0;
    const double total_flops = static_cast<double>(num_rows) * flops_per_element;
    constexpr double bytes_per_element = 3.0 * sizeof(double);
    const double total_bytes = static_cast<double>(num_rows) * bytes_per_element;

    const double gflops = total_flops / seconds / 1.0e9;
    const double bandwidth_gb_s = total_bytes / seconds / 1.0e9;
    const double arithmetic_intensity = flops_per_element / bytes_per_element;

    double checksum = 0.0;
    for (std::size_t i = 0; i < num_rows; ++i) {
        checksum += h_y[i];
    }
    

    //free up memory
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return 0;
}