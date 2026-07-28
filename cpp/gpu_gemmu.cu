/*
(C = \alpha (A*B) + beta C).
an H100 has 132 SMs, and a Blackwell B200
GEMM is 2D hence fully exploits the threadblock.
- using shared memory , test shared memory bank conflict
- unified memory
equation:  = sum of pdt of shared tile K + C[{row}][{col}]
no concurrency and pinned memory here -
*/
//roofline model
#include <iostream>
#include <chrono>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <numeric>
#include <cuda_runtime.h>

// Define compile-time physical tile dimensions inside the SM SRAM = 1024 threads
#define TILE_DIM 32

struct HardwareLimits {
    double peak_gflops;
    double peak_bandwidth_gb_s;
    double ridge_point;
};

HardwareLimits get_gpu_limits() {
    HardwareLimits limits{0.0, 0.0, 0.0};
    int device_id = 0;
    if (cudaGetDevice(&device_id) == cudaSuccess) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device_id);
        // Peak theoretical single-precision (FP32) calculation
        limits.peak_gflops = prop.multiProcessorCount * 128.0 * (prop.clockRate / 1.0e6);
        limits.peak_bandwidth_gb_s = (prop.memoryClockRate * 1000.0 * (prop.memoryBusWidth / 8.0) * 2.0) / 1.0e9;
    } else {
        limits.peak_gflops = 10000.0; 
        limits.peak_bandwidth_gb_s = 900.0;
    }
    limits.ridge_point = limits.peak_gflops / limits.peak_bandwidth_gb_s;
    return limits;
}

std::string get_machine_name() {
#if defined(_WIN32) || defined(_WIN64)
    const char* env = std::getenv("COMPUTERNAME");
#else
    const char* env = std::getenv("HOSTNAME");
    if (!env) env = std::getenv("NAME");
#endif
    return env ? std::string(env) : "GPU_Machine";
}

__global__ void gemm_kernel(
    const float *A, const float *B, float *C, 
    int M, int K, int N, float alpha, float beta) {
    
    // allocate the device shared memory array - strategy 2
    __shared__ float tile_A[TILE_DIM][TILE_DIM];
    __shared__ float tile_B[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    float sum = 0.0f;

    // loop through shared dimension K in tile size
    // Boundary mapping
    for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; ++t) {
        // load matrix A to shared memory
        if (row < M && (t * TILE_DIM + threadIdx.x) < K) {
            tile_A[threadIdx.y][threadIdx.x] = A[row * K + (t * TILE_DIM + threadIdx.x)];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // load matrix B into shared memory as well
        if (col < N && (t * TILE_DIM + threadIdx.y) < K) {
            tile_B[threadIdx.y][threadIdx.x] = B[(t * TILE_DIM + threadIdx.y) * N + col];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // ensure all tiles are loaded
        __syncthreads();

        for (int k = 0; k < TILE_DIM; ++k){
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }
        
        __syncthreads();
    }

    // write final alpha/beta and results back to global memory
    if (row < M && col < N) {
        int idx = row * N + col;
        C[idx] = (alpha * sum) + (beta * C[idx]);
    }
}

// error check
void checkcuda(cudaError_t result) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void run_gemm_um_sweep(int dim, const std::string& machine_name, std::ofstream& csv, const HardwareLimits& hw) {
    // Keep sizes square matching your sweep loop configuration
    const int M = dim;
    const int N = dim;
    const int K = dim;
    const float alpha = 1.5f;
    const float beta = 0.5f;

    size_t size_A = static_cast<size_t>(M) * K * sizeof(float);
    size_t size_B = static_cast<size_t>(K) * N * sizeof(float);
    size_t size_C = static_cast<size_t>(M) * N * sizeof(float);

    // Unified Memory allocation
    float *A; float *B; float *C;
    checkcuda(cudaMallocManaged(&A, size_A));
    checkcuda(cudaMallocManaged(&B, size_B));
    checkcuda(cudaMallocManaged(&C, size_C));

    // initialize matrices directly through unified memory
    for (int i = 0; i < M * K; ++i) A[i] = 1.0f;
    for (int i = 0; i < K * N; ++i) B[i] = 0.5f;
    for (int i = 0; i < M * N; ++i) C[i] = 2.0f;

    // grid mapping on device
    dim3 blockDim(TILE_DIM, TILE_DIM);
    dim3 gridDim((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);

    // --- Time total pipeline initialization from here ---
    const auto total_start = std::chrono::steady_clock::now();

    // launch kernel (Warm-up pass where driver implicitly handles page migration)
    gemm_kernel<<<gridDim, blockDim>>>(A, B, C, M, K, N, alpha, beta);
    checkcuda(cudaGetLastError());
    checkcuda(cudaDeviceSynchronize());

    // Active Timing Block (Measures pure kernel execution once memory has settled)
    const auto start = std::chrono::steady_clock::now();
    gemm_kernel<<<gridDim, blockDim>>>(A, B, C, M, K, N, alpha, beta);
    checkcuda(cudaGetLastError());
    checkcuda(cudaDeviceSynchronize());
    const auto end = std::chrono::steady_clock::now();
    
    const auto total_end = std::chrono::steady_clock::now();

    const std::chrono::duration<double> elapsed = end - start;
    const double seconds = elapsed.count();
    
    const double total_sec = std::chrono::duration<double>(total_end - total_start).count();

    // Calculate Metrics matching your exact layout rules
    const std::size_t num_rows = static_cast<std::size_t>(M) * static_cast<std::size_t>(N);
    constexpr double flops_per_element = 2.0;
    const double total_flops = flops_per_element * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
    
    // Correct algorithmic footprint calculation for dense matrix reuse
    const double total_bytes = static_cast<double>(size_A + size_B + (2.0 * size_C));

    const double gflops = total_flops / seconds / 1.0e9;
    const double bandwidth_gb_s = total_bytes / seconds / 1.0e9;
    const double arithmetic_intensity = total_flops / total_bytes;

    double checksum = 0.0;
    for (std::size_t i = 0; i < num_rows; ++i) {
        checksum += C[i];
    }

    std::cout << "Matrix Size: " << M << "x" << N << " | Bandwidth: " << bandwidth_gb_s << " GB/s | Performance: " << gflops << " GFLOP/s\n";

    // Write parameters side-by-side with your automated hardware ceilings
    csv << machine_name << ","
        << "gemm Dense Tiled UM cuda" << ","
        << M << "," 
        << arithmetic_intensity << ","
        << bandwidth_gb_s << ","
        << gflops << ","
        << seconds << ","      // kernel_time (s)
        << total_sec << ","    // total_time_with_io (s)
        << hw.peak_gflops << ","
        << hw.peak_bandwidth_gb_s << ","
        << hw.ridge_point << ","
        << checksum << "\n";

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
}

int main() {
    std::string machine_name = get_machine_name();
    HardwareLimits hw = get_gpu_limits();

    // Sweeping dimensions configurations (Square N x N matrices)
    std::vector<int> sweep_sizes = {256, 512, 1024, 2048, 4096};

    std::ifstream check_empty("roofline.csv");
    bool add_header = !check_empty.is_open() || check_empty.peek() == std::ifstream::traits_type::eof();
    check_empty.close();

    std::ofstream csv("roofline.csv", std::ios::app);
    if (add_header) {
        csv << "machine,dwarf name,problem_size,AI,measured_bandwidth(GB/s),"
               "measured_performance(GFLOP/s),kernel_time(s),total_time_with_io(s),"
               "peak_compute(GFLOP/s),peak_bandwidth(GB/s),ridge_point,checksum\n";
    }

    std::cout << "Running Unified Memory GEMM Sweeps on: " << machine_name << "\n";
    for (int N : sweep_sizes) {
        run_gemm_um_sweep(N, machine_name, csv, hw);
    }

    return 0;
}
