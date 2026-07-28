/*
Baseline code for STREAM_TRIAD
Building the roofline model - 

*/

#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <cstdlib>
#include <numeric>
#include <cuda_runtime.h>

struct HardwareLimits {
    double peak_gflops;
    double peak_bandwidth_gb_s;
    double ridge_point;
};

// Automatically query active hardware specifications 
HardwareLimits get_gpu_limits() {
    HardwareLimits limits{0.0, 0.0, 0.0};
    int device_id = 0;
    
    if (cudaGetDevice(&device_id) == cudaSuccess) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device_id);
        
        // VRAM Bandwidth = Memory Clock (kHz) * 1000 * Bus Width (bits/8) * 2 (DDR multiplier) / 1e9
        limits.peak_bandwidth_gb_s = (prop.memoryClockRate * 1000.0 * (prop.memoryBusWidth / 8.0) * 2.0) / 1.0e9;
        
        // Multi-processor configuration lookup for theoretical peak compute
        // Standard baseline configuration fallback if device query doesn't expose raw FLOPS directly
        limits.peak_gflops = prop.multiProcessorCount * 128.0 * (prop.clockRate / 1.0e6); 
    } else {
        // Fallback architecture metrics (e.g., standard baseline workstation defaults)
        limits.peak_gflops = 5000.0;
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

__global__ void stream_triad_kernel(const double* A, const double* B, double* C, double scalar, size_t N) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i] * scalar;
    }
}

void run_sweep(size_t N, const std::string& machine_name, std::ofstream& csv, const HardwareLimits& hw) {
    constexpr double scalar = 2.0;

    std::vector<double> h_A(N, 5.0);
    std::vector<double> h_B(N, 10.0);
    std::vector<double> h_C(N, 0.0);

    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * sizeof(double));
    cudaMalloc(&d_B, N * sizeof(double));
    cudaMalloc(&d_C, N * sizeof(double));

    // --- Time Data Transfer Host to Device (H2D) ---
    const auto transfer_start = std::chrono::steady_clock::now();
    cudaMemcpy(d_A, h_A.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    
    int threads_per_block = 256;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    // Warm-up pass
    stream_triad_kernel<<<blocks_per_grid, threads_per_block>>>(d_A, d_B, d_C, scalar, N);
    cudaDeviceSynchronize();

    // --- Time Pure Kernel Execution ---
    const auto kernel_start = std::chrono::steady_clock::now();
    stream_triad_kernel<<<blocks_per_grid, threads_per_block>>>(d_A, d_B, d_C, scalar, N);
    cudaDeviceSynchronize();
    const auto kernel_end = std::chrono::steady_clock::now();

    // --- Time Data Transfer Device to Host (D2H) ---
    cudaMemcpy(h_C.data(), d_C, N * sizeof(double), cudaMemcpyDeviceToHost);
    const auto total_end = std::chrono::steady_clock::now();

    // Calculate isolated phase timings
    const double kernel_sec = std::chrono::duration<double>(kernel_end - kernel_start).count();
    const double total_sec = std::chrono::duration<double>(total_end - transfer_start).count();

    // Compute Metrics
    constexpr double flops_per_element = 2.0;
    constexpr double bytes_per_element = 3.0 * sizeof(double);
    const double total_flops = static_cast<double>(N) * flops_per_element;
    const double total_bytes = static_cast<double>(N) * bytes_per_element;

    const double kernel_gflops = total_flops / kernel_sec / 1.0e9;
    const double kernel_bandwidth = total_bytes / kernel_sec / 1.0e9;
    const double arithmetic_intensity = flops_per_element / bytes_per_element;
    
    // Aggregate absolute end validation checksum
    double checksum = 0.0;
    for (std::size_t i = 0; i < N; ++i) { checksum += h_C[i]; }

    std::cout << "Size: " << N << " | Kernel BW: " << kernel_bandwidth << " GB/s | Kernel Performance: " << kernel_gflops << " GFLOP/s\n";

    // Write parameters, isolated measurements, and theoretical ceilings side-by-side
    csv << machine_name << ","
        << "stream triad direct cuda" << ","
        << N << ","
        << arithmetic_intensity << ","
        << kernel_bandwidth << ","
        << kernel_gflops << ","
        << kernel_sec << ","
        << total_sec << "," // Includes explicit data transfer time
        << hw.peak_gflops << ","
        << hw.peak_bandwidth_gb_s << ","
        << hw.ridge_point << ","
        << checksum << "\n";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    std::string machine_name = get_machine_name();
    HardwareLimits hw = get_gpu_limits();

    // Automated array sweep targets
    std::vector<size_t> sweep_sizes = {1000000, 5000000, 10000000, 50000000};

    std::ifstream check_empty("roofline.csv");
    bool add_header = !check_empty.is_open() || check_empty.peek() == std::ifstream::traits_type::eof();
    check_empty.close();

    std::ofstream csv("roofline.csv", std::ios::app);
    if (add_header) {
        csv << "machine,dwarf name,problem_size,AI,measured_bandwidth(GB/s),"
               "measured_performance(GFLOP/s),kernel_time(s),total_time_with_io(s),"
               "peak_compute(GFLOP/s),peak_bandwidth(GB/s),ridge_point,checksum\n";
    }

    std::cout << "Running CUDA Direct profiling sweeps on: " << machine_name << "\n";
    for (size_t N : sweep_sizes) {
        run_sweep(N, machine_name, csv, hw);
    }

    return 0;
}