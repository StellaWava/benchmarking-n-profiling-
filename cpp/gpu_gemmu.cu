/*
(C = \alpha (A*B) + beta C).
an H100 has 132 SMs, and a Blackwell B200
GEMM is 2D hence fully exploits the threadblock.
- using shared memory , test shared memory bank conflict
- unified memory
equation:  = sum of pdt of shared tile K + C[{row}][{col}]
no concurrency and pinned memory here -
*/

#include <iostream>
#include <chrono>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// Define compile-time physical tile dimensions inside the SM SRAM = 1024 threads
#define TILE_DIM 32

__global__ void gemm_kernel(
    const float *A,
    const float *B,
    float *C,
    int M,
    int K,
    int N,
    float alpha,
    float beta)
{
    //allocate the device shared memory array - strategy 2
    __shared__ float tile_A[TILE_DIM][TILE_DIM];
    __shared__ float tile_B[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    float sum = 0.0f;

    //loop through shared dimension K in tile size
    //Boundary mapping
    for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; ++t) {
        //load matrix A to shared memory
        /*
        Emphasize size of submatrices for K..i.e. run a row check and a col check for both rows and cols
        row check: sub row entering is less than larger matrix M, also tiled within overall dimension of K
        i.e. (t *TILE_DIM + threadIdx.x) < K
        */
        if (row < M && (t * TILE_DIM + threadIdx.x) < K) {
            tile_A[threadIdx.y][threadIdx.x] =
            A[row * K + (t * TILE_DIM + threadIdx.x)];
        } 
        else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f; 
        }

        //load matrix B into shared memory as well
        /*
        col check: sub col entering is less than larger matrix N, also tiled within overall dimension of K
        i.e. (t *TILE_DIM + threadIdx.y) < K
        */
        if (col < N && (t * TILE_DIM + threadIdx.y) < K) {
            tile_B[threadIdx.y][threadIdx.x] =
            B[(t * TILE_DIM + threadIdx.y) * N + col];
        } 
        else {
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        //ensure all tiles are loaded
        __syncthreads();

        for (int k = 0; k < TILE_DIM; ++k){
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }
        __syncthreads();

    }

    //write final alpha/beta and results back to global memory
    if (row < M && col < N) {
        int idx = row * N + col;
        C[idx] = (alpha * sum) + (beta * C[idx]);
    }
}

//error check
void checkcuda(cudaError_t result)
{
    if (result != cudaSuccess) {
        std::cerr << "CUDA Runtime Error: "
                  << cudaGetErrorString(result)
                  << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main()
{
    const int M = 4096;
    const int N = 4096;
    const int K = 4096;

    const float alpha = 1.5f;
    const float beta = 0.5f;

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    //Unified Memory allocation
    float *A;
    float *B;
    float *C;

    checkcuda(cudaMallocManaged(&A, size_A));
    checkcuda(cudaMallocManaged(&B, size_B));
    checkcuda(cudaMallocManaged(&C, size_C));

    //initialize matrices directly through unified memory
    for (int i = 0; i < M * K; ++i) A[i] = 1.0f;
    for (int i = 0; i < K * N; ++i) B[i] = 0.5f;
    for (int i = 0; i < M * N; ++i) C[i] = 2.0f;

    //grid mapping on device
    dim3 blockDim(TILE_DIM, TILE_DIM);
    dim3 gridDim((N + TILE_DIM - 1) / TILE_DIM,(M + TILE_DIM - 1) / TILE_DIM);

    //launch kernel
    gemm_kernel<<<gridDim, blockDim>>>(A, B, C, M, N, K, alpha, beta);

    checkcuda(cudaGetLastError());
    checkcuda(cudaDeviceSynchronize());

    //Active Timing Block
    const auto start = std::chrono::steady_clock::now();

    //launch kernel
    gemm_kernel<<<gridDim, blockDim>>>(A, B, C, M, N, K, alpha, beta);

    checkcuda(cudaGetLastError());

    //sync gpu
    checkcuda(cudaDeviceSynchronize());

    //measure time
    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = end - start;
    const double seconds = elapsed.count();

    //No device-to-host copy is required.
    //After synchronization, C can be read directly by the CPU.

    //verify results
    float expected = 1.5f * (K * (1.0f * 0.5f)) + beta * 2.0f;
    std::cout << "Baseline GEMM done. Results:"<< C[0]<< " (Expected: "<< expected<< ")"<< std::endl;

    // 7. Calculate Metrics
    const std::size_t num_rows = static_cast<std::size_t>(M) * static_cast<std::size_t>(N);
    constexpr double flops_per_element = 2.0;
    const double total_flops = flops_per_element * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
    constexpr double bytes_per_element = 3.0 * sizeof(double);
    const double total_bytes = static_cast<double>(num_rows) * bytes_per_element;

    const double gflops = total_flops / seconds / 1.0e9;
    const double bandwidth_gb_s = total_bytes / seconds / 1.0e9;
    const double arithmetic_intensity = flops_per_element / bytes_per_element;

    double checksum = 0.0;

    for (std::size_t i = 0; i < num_rows; ++i) {
        checksum += C[i];
    }

    std::cout << "CUDA GPU GEMM | Bandwidth: " << bandwidth_gb_s << " GB/s | Performance: " << gflops << " GFLOP/s\n";

    // --- CSV MANAGEMENT ---
    std::ifstream check_empty("roofline.csv");

    bool add_header =
        !check_empty.is_open() ||
        check_empty.peek() ==
            std::ifstream::traits_type::eof();

    check_empty.close();

    std::ofstream csv("roofline.csv", std::ios::app);

    if (add_header) {
        csv << "machine,dwarf name,AI,bandwidth (GB/s),"
               "performance (GFLOP/s),threads,checksum\n";
    }

    const char *machine_name = "CUDA GPU";

    csv << machine_name<< "," << "GEMM Tileu" << "," << arithmetic_intensity << "," << bandwidth_gb_s << "," << gflops << "," << "CUDA" << "," << checksum << "\n";

    //free unified memory
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}