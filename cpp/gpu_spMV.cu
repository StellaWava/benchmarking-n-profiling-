// File for SpMv Profiling - SpMV is mapped using CSR approach
//CSR approach is row index(I), row value(V) NNS, and column index (C) NNS
//y = A.x where y and x are dense inputs but A is sparce.

#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <numeric>
#include <cuda_runtime.h>

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
        // Compute peak theoretical single-precision (FP32) performance instead of FP64
        // Base FP32 mapping estimation if architecture is unknown
        limits.peak_gflops = prop.multiProcessorCount * 128.0 * (prop.clockRate / 1.0e6);
        limits.peak_bandwidth_gb_s = (prop.memoryClockRate * 1000.0 * (prop.memoryBusWidth / 8.0) * 2.0) / 1.0e9;
    } else {
        limits.peak_gflops = 10000.0; // Sample Workstation baseline (FP32)
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

__global__ void spmv_kernel(const int *I, const float *V, const int *C, const float *x, float *y, int num_rows){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows){
        float dot_product = 0.0f;
        int row_start = I[row];
        int row_end = I[row + 1];
        for (int element = row_start; element < row_end; ++element){
            int col = C[element];
            dot_product += V[element] * x[col];
        }
        y[row] = dot_product;
    }
}

// Generate a synthetic pseudo-random CSR matrix for stable benchmarking
void generate_random_csr(int num_rows, int num_cols, float density, 
                         std::vector<int>& h_I, std::vector<float>& h_V, std::vector<int>& h_C) {
    h_I.resize(num_rows + 1, 0);
    int current_nnz = 0;
    
    for (int i = 0; i < num_rows; ++i) {
        h_I[i] = current_nnz;
        for (int j = 0; j < num_cols; ++j) {
            if ((static_cast<float>(rand()) / RAND_MAX) < density) {
                h_V.push_back(1.5f); // Constant non-zero to verify reproducibility
                h_C.push_back(j);
                current_nnz++;
            }
        }
    }
    h_I[num_rows] = current_nnz;
}

void run_spmv_sweep(int N, float density, const std::string& machine_name, std::ofstream& csv, const HardwareLimits& hw) {
    int num_rows = N;
    int num_cols = N;

    std::vector<int> h_I;
    std::vector<float> h_V;
    std::vector<int> h_C;
    generate_random_csr(num_rows, num_cols, density, h_I, h_V, h_C);
    
    int nnz = h_I[num_rows];
    std::vector<float> h_x(num_cols, 1.0f);
    std::vector<float> h_y(num_rows, 0.0f);

    size_t size_I = (num_rows + 1) * sizeof(int);
    size_t size_V = nnz * sizeof(float);
    size_t size_C = nnz * sizeof(int);
    size_t size_x = num_cols * sizeof(float);
    size_t size_y = num_rows * sizeof(float);

    int *d_I; float *d_V; int *d_C; float *d_x; float *d_y;
    cudaMalloc((void**)&d_I, size_I);
    cudaMalloc((void**)&d_V, size_V);
    cudaMalloc((void**)&d_C, size_C);
    cudaMalloc((void**)&d_x, size_x);
    cudaMalloc((void**)&d_y, size_y);

    // --- Time H2D Data Transfer ---
    const auto transfer_start = std::chrono::steady_clock::now();
    cudaMemcpy(d_I, h_I.data(), size_I, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V.data(), size_V, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C.data(), size_C, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x.data(), size_x, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int numBlocks = (num_rows + threadsPerBlock - 1) / threadsPerBlock;

    // Warm up
    spmv_kernel<<<numBlocks, threadsPerBlock>>>(d_I, d_V, d_C, d_x, d_y, num_rows);
    cudaDeviceSynchronize();

    // --- Time Pure GPU Execution ---
    const auto kernel_start = std::chrono::steady_clock::now();
    spmv_kernel<<<numBlocks, threadsPerBlock>>>(d_I, d_V, d_C, d_x, d_y, num_rows);
    cudaDeviceSynchronize();
    const auto kernel_end = std::chrono::steady_clock::now();

    // --- Time D2H Data Transfer ---
    cudaMemcpy(h_y.data(), d_y, size_y, cudaMemcpyDeviceToHost);
    const auto total_end = std::chrono::steady_clock::now();

    double kernel_sec = std::chrono::duration<double>(kernel_end - kernel_start).count();
    double total_sec = std::chrono::duration<double>(total_end - transfer_start).count();

    // --- Correct Math for CSR SpMV Metrics ---
    // FLOPs: Every non-zero requires 1 Multiply and 1 Accumulate = 2 * NNZ
    double total_flops = 2.0 * static_cast<double>(nnz);
    
    // Bytes Accessed: Read I, V, C, and vector x (indirect memory access penalty overlooked for base baseline) + Write y
    double total_bytes = static_cast<double>((num_rows + 1) * sizeof(int) + 
                                              nnz * sizeof(float) + 
                                              nnz * sizeof(int) + 
                                              nnz * sizeof(float) + // Vector x cache access approximation
                                              num_rows * sizeof(float));

    double kernel_gflops = total_flops / kernel_sec / 1.0e9;
    double kernel_bandwidth = total_bytes / kernel_sec / 1.0e9;
    double arithmetic_intensity = total_flops / total_bytes;

    double checksum = 0.0;
    for (int i = 0; i < num_rows; ++i) { checksum += h_y[i]; }

    std::cout << "Rows: " << num_rows << " | NNZ: " << nnz 
              << " | Kernel BW: " << kernel_bandwidth << " GB/s"
              << " | Performance: " << kernel_gflops << " GFLOP/s\n";

    csv << machine_name << ","
        << "spMV CSR cuda" << ","
        << num_rows << "," // Track primary square matrix bound dimension
        << arithmetic_intensity << ","
        << kernel_bandwidth << ","
        << kernel_gflops << ","
        << kernel_sec << ","
        << total_sec << ","
        << hw.peak_gflops << ","
        << hw.peak_bandwidth_gb_s << ","
        << hw.ridge_point << ","
        << checksum << "\n";

    cudaFree(d_I); cudaFree(d_V); cudaFree(d_C); cudaFree(d_x); cudaFree(d_y);
}

int main() {
    std::string machine_name = get_machine_name();
    HardwareLimits hw = get_gpu_limits();
    srand(1337); // Seed generator for stable tracking across comparisons

    // Scaling matrix configurations (num_rows = num_cols)
    std::vector<int> sweep_sizes = {1000, 5000, 10000, 20000};
    float matrix_density = 0.05f; // Keep it sparse at 5% density

    std::ifstream check_empty("roofline.csv");
    bool add_header = !check_empty.is_open() || check_empty.peek() == std::ifstream::traits_type::eof();
    check_empty.close();

    std::ofstream csv("roofline.csv", std::ios::app);
    if (add_header) {
        csv << "machine,dwarf name,problem_size,AI,measured_bandwidth(GB/s),"
               "measured_performance(GFLOP/s),kernel_time(s),total_time_with_io(s),"
               "peak_compute(GFLOP/s),peak_bandwidth(GB/s),ridge_point,checksum\n";
    }

    std::cout << "Running SpMV CSR CUDA Sweeps on: " << machine_name << "\n";
    for (int N : sweep_sizes) {
        run_spmv_sweep(N, matrix_density, machine_name, csv, hw);
    }

    return 0;
}