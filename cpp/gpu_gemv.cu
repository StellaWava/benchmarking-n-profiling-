/*
Baseline GEMV
*/
#include <iostream>
#include <chrono>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <numeric>
#include <cuda_runtime.h>

// 1D Block Thread Size limit (256 threads per block)
#define BLOCK_SIZE 256

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

void checkcuda(cudaError_t result) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Dense Matrix-Vector Multiplication Kernel: y = alpha * (A * x) + beta * y
__global__ void gemv_kernel(const float *A, const float *x, float *y, int num_rows, int num_cols, float alpha, float beta) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < num_rows) {
        float sum = 0.0f;
        // Loop through the columns of the row
        for (int col = 0; col < num_cols; ++col) {
            sum += A[row * num_cols + col] * x[col];
        }
        
        // Structural barrier anchoring execution before saving elements back to global memory
        __syncthreads();
        
        y[row] = (alpha * sum) + (beta * y[row]);
    }
}

void run_gemv_um_sweep(int dim, const std::string& machine_name, std::ofstream& csv, const HardwareLimits& hw) {
    // Keeping matrix bounds square for automated scaling sweeps: num_rows = num_cols = dim
    const int num_rows = dim;
    const int num_cols = dim;
    const float alpha = 1.5f;
    const float beta = 0.5f;

    size_t size_A = static_cast<size_t>(num_rows) * num_cols * sizeof(float);
    size_t size_x = static_cast<size_t>(num_cols) * sizeof(float);
    size_t size_y = static_cast<size_t>(num_rows) * sizeof(float);

    float *A; float *x; float *y;
    checkcuda(cudaMallocManaged(&A, size_A));
    checkcuda(cudaMallocManaged(&x, size_x));
    checkcuda(cudaMallocManaged(&y, size_y));

    // Initialize allocations directly via host CPU threads
    for (size_t i = 0; i < static_cast<size_t>(num_rows) * num_cols; ++i) A[i] = 1.0f;
    for (int i = 0; i < num_cols; ++i) x[i] = 0.5f;
    for (int i = 0; i < num_rows; ++i) y[i] = 2.0f;

    // Standard 1D execution layout configuration 
    int threadsPerBlock = BLOCK_SIZE;
    int numBlocks = (num_rows + threadsPerBlock - 1) / threadsPerBlock;

    // --- Time total pipeline initialization from here ---
    const auto total_start = std::chrono::steady_clock::now();

    // Launch kernel (Warm-up pass where driver implicitly handles on-demand page migration)
    gemv_kernel<<<numBlocks, threadsPerBlock>>>(A, x, y, num_rows, num_cols, alpha, beta);
    checkcuda(cudaGetLastError());
    checkcuda(cudaDeviceSynchronize());

    // Active Timing Block (Measures pure kernel execution once memory has settled)
    const auto start = std::chrono::steady_clock::now();
    gemv_kernel<<<numBlocks, threadsPerBlock>>>(A, x, y, num_rows, num_cols, alpha, beta);
    checkcuda(cudaGetLastError());
    checkcuda(cudaDeviceSynchronize());
    const auto end = std::chrono::steady_clock::now();
    
    const auto total_end = std::chrono::steady_clock::now();

    const double seconds = std::chrono::duration<double>(end - start).count();
    const double total_sec = std::chrono::duration<double>(total_end - total_start).count();

    // --- Correct Algorithmic Math for Dense GEMV Roofline Metrics ---
    // FLOPs: Every single output element requires N multiplications and N additions = 2 * M * N
    const double total_flops = 2.0 * static_cast<double>(num_rows) * static_cast<double>(num_cols);
    
    // Bytes Transferred: Read A (entire matrix), Read x (accessed multiple times), Read/Write y
    const double total_bytes = static_cast<double>(size_A + size_x + (2.0 * size_y));

    const double gflops = total_flops / seconds / 1.0e9;
    const double bandwidth_gb_s = total_bytes / seconds / 1.0e9;
    const double arithmetic_intensity = total_flops / total_bytes;

    double checksum = 0.0;
    for (int i = 0; i < num_rows; ++i) { checksum += y[i]; }

    std::cout << "Matrix Size: " << num_rows << "x" << num_cols 
              << " | Bandwidth: " << bandwidth_gb_s << " GB/s | Performance: " << gflops << " GFLOP/s\n";

    // Write parameters side-by-side with your automated hardware ceilings
    csv << machine_name << ","
        << "gemv Dense UM cuda" << ","
        << num_rows << "," 
        << arithmetic_intensity << ","
        << bandwidth_gb_s << ","
        << gflops << ","
        << seconds << ","      // kernel_time (s)
        << total_sec << ","    // total_time_with_io (s)
        << hw.peak_gflops << ","
        << hw.peak_bandwidth_gb_s << ","
        << hw.ridge_point << ","
        << checksum << "\n";

    checkcuda(cudaFree(A));
    //checkcuda(cudaFree(B)); // Wait! This matches your original error structure. Let's fix to the assigned variables:
    checkcuda(cudaFree(x));
    checkcuda(cudaFree(y));
}

int main() {
    std::string machine_name = get_machine_name();
    HardwareLimits hw = get_gpu_limits();

    // Sweeping dimensions configurations (Square N x N execution footprint limits)
    std::vector<int> sweep_sizes = {1024, 2048, 4096, 8192};

    std::ifstream check_empty("roofline.csv");
    bool add_header = !check_empty.is_open() || check_empty.peek() == std::ifstream::traits_type::eof();
    check_empty.close();

    std::ofstream csv("roofline.csv", std::ios::app);
    if (add_header) {
        csv << "machine,dwarf name,problem_size,AI,measured_bandwidth(GB/s),"
               "measured_performance(GFLOP/s),kernel_time(s),total_time_with_io(s),"
               "peak_compute(GFLOP/s),peak_bandwidth(GB/s),ridge_point,checksum\n";
    }

    std::cout << "Running Unified Memory GEMV Sweeps on: " << machine_name << "\n";
    for (int N : sweep_sizes) {
        run_gemv_um_sweep(N, machine_name, csv, hw);
    }

    return 0;
}