/*
(C = \alpha (A*B) + beta C).
an H100 has 132 SMs, and a Blackwell B200
GEMM is 2D hence fully exploits the threadblock.
- using shared memory , test shared memory bank conflict
- - unified memory,
equation:  = sum of pdt of shared tile K + C[{row}][{col}]
no concurrency and pinned memory here -
*/


//////
#include <iostream>
#include <chrono>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <numeric>
#include <cuda_runtime.h>

// Tile dimensions matching optimal thread block geometry boundaries
#define TILE_DIM 16

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

// Optimized Tiled GEMM Kernel utilizing explicit Shared Memory Blocks
__global__ void gemm_tiled_kernel(const float *A, const float *B, float *C, int N) {
    // 1. Static Allocation of On-Chip Shared Memory Tiles
    __shared__ float tile_A[TILE_DIM][TILE_DIM];
    __shared__ float tile_B[TILE_DIM][TILE_DIM];

    // 2. Global Coordinate Boundary Mapping
    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    
    float sum = 0.0f;

    // Loop across matrix sub-tiles in phases
    int num_phases = (N + TILE_DIM - 1) / TILE_DIM;
    for (int phase = 0; phase < num_phases; ++phase) {
        
        // 3. Load Tile A into Shared Memory with safe out-of-bound padding
        if (row < N && (phase * TILE_DIM + threadIdx.x) < N) {
            tile_A[threadIdx.y][threadIdx.x] = A[row * N + (phase * TILE_DIM + threadIdx.x)];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // 4. Load Tile B into Shared Memory with safe out-of-bound padding
        if (col < N && (phase * TILE_DIM + threadIdx.y) < N) {
            tile_B[threadIdx.y][threadIdx.x] = B[(phase * TILE_DIM + threadIdx.y) * N + col];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // 5. BARRIER: Wait for entire block to finish populating shared storage arrays
        __syncthreads();

        // 6. Compute dot product for the current tile phase out of fast registers
        for (int k = 0; k < TILE_DIM; ++k) {
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }

        // 7. BARRIER: Ensure calculations wrap up before step iterations overwrite tiles
        __syncthreads();
    }

    // 8. Commit final register accumulation back to discrete global VRAM memory maps
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

void run_gemm_discrete_sweep(int N, const std::string& machine_name, std::ofstream& csv, const HardwareLimits& hw) {
    size_t matrix_size = static_cast<size_t>(N) * N;
    size_t bytes_per_matrix = matrix_size * sizeof(float);

    // Discrete Host Memory Staging
    std::vector<float> h_A(matrix_size, 1.0f);
    std::vector<float> h_B(matrix_size, 2.0f);
    std::vector<float> h_C(matrix_size, 0.0f);

    // Discrete Device Memory Allocations
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, bytes_per_matrix);
    cudaMalloc((void**)&d_B, bytes_per_matrix);
    cudaMalloc((void**)&d_C, bytes_per_matrix);

    // Time explicit PCIe data movement
    const auto transfer_start = std::chrono::steady_clock::now();
    cudaMemcpy(d_A, h_A.data(), bytes_per_matrix, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), bytes_per_matrix, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // Map Block Dimensions matching our shared arrays
    dim3 threadsPerBlock(TILE_DIM, TILE_DIM);
    dim3 numBlocks((N + TILE_DIM - 1) / TILE_DIM, 
                   (N + TILE_DIM - 1) / TILE_DIM);

    // Warm-up pass
    gemm_tiled_kernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    //Kernel Timing Isolation Block
    const auto kernel_start = std::chrono::steady_clock::now();
    gemm_tiled_kernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize(); 
    const auto kernel_end = std::chrono::steady_clock::now();

    // Time Return pipeline path
    cudaMemcpy(h_C.data(), d_C, bytes_per_matrix, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize(); 
    const auto total_end = std::chrono::steady_clock::now();

    double kernel_sec = std::chrono::duration<double>(kernel_end - kernel_start).count();
    double total_sec = std::chrono::duration<double>(total_end - transfer_start).count();

    // Workload Roofline Performance Metrics
    const double total_flops = 2.0 * static_cast<double>(N) * N * N;
    const double total_bytes = 3.0 * static_cast<double>(bytes_per_matrix);

    const double kernel_gflops = total_flops / kernel_sec / 1.0e9;
    const double kernel_bandwidth = total_bytes / kernel_sec / 1.0e9;
    const double arithmetic_intensity = total_flops / total_bytes;

    double checksum = 0.0;
    for (size_t i = 0; i < matrix_size; ++i) { checksum += h_C[i]; }

    std::cout << "Matrix Size: " << N << "x" << N << " | Tiled Kernel BW: " << kernel_bandwidth << " GB/s"
              << " | Performance: " << kernel_gflops << " GFLOP/s\n";

    csv << machine_name << ","
        << "gemm Tiled Shared Discrete cuda" << ","
        << N << "," 
        << arithmetic_intensity << ","
        << kernel_bandwidth << ","
        << kernel_gflops << ","
        << kernel_sec << ","
        << total_sec << "," 
        << hw.peak_gflops << ","
        << hw.peak_bandwidth_gb_s << ","
        << hw.ridge_point << ","
        << checksum << "\n";

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

int main() {
    std::string machine_name = get_machine_name();
    HardwareLimits hw = get_gpu_limits();

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

    std::cout << "Running Tiled Shared Discrete GEMM CUDA Sweeps on: " << machine_name << "\n";
    for (int N : sweep_sizes) {
        run_gemm_discrete_sweep(N, machine_name, csv, hw);
    }

    return 0;
}