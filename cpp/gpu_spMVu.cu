//spMV baseline 
//roofline model 
#include <iostream>
#include <chrono>
#include <vector>
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

void run_spmv_um_sweep(int num_rows, const std::string& machine_name, std::ofstream& csv, const HardwareLimits& hw) {
    int nnz = num_rows * 4; // Constant 4 non-zeros per row average
    
    size_t size_I = (num_rows + 1) * sizeof(int);
    size_t size_V = nnz * sizeof(float);
    size_t size_C = nnz * sizeof(int);
    size_t size_vector = num_rows * sizeof(float);

    int *I; float *V; int *C; float *x; float *y;
    cudaMallocManaged((void**)&I, size_I);
    cudaMallocManaged((void**)&V, size_V);
    cudaMallocManaged((void**)&C, size_C);
    cudaMallocManaged((void**)&x, size_vector);
    cudaMallocManaged((void**)&y, size_vector);

    // Initialize memory directly via host CPU threads
    for (int i = 0; i <= num_rows; ++i) I[i] = i * 4;
    for (int i = 0; i < nnz; ++i) {
        V[i] = 10.0f;
        C[i] = (i % num_rows);
    }
    for (int i = 0; i < num_rows; ++i) {
        x[i] = 1.0f;
        y[i] = 0.0f;
    }

    int device = 0;
    
    // --- Time H2D Pipeline Migration via Unified Memory Prefetch ---
    const auto transfer_start = std::chrono::steady_clock::now();
    cudaMemPrefetchAsync(I, size_I, device, NULL);
    cudaMemPrefetchAsync(V, size_V, device, NULL);
    cudaMemPrefetchAsync(C, size_C, device, NULL);
    cudaMemPrefetchAsync(x, size_vector, device, NULL);
    cudaDeviceSynchronize(); // Guarantee items migrated before benchmarking

    int threadsPerBlock = 256;
    int numBlocks = (num_rows + threadsPerBlock - 1) / threadsPerBlock;

    // --- WARM UP PASS ---
    spmv_kernel<<<numBlocks, threadsPerBlock>>>(I, V, C, x, y, num_rows);
    cudaDeviceSynchronize();

    // --- ACTIVE ISOLATED KERNEL TIMING BLOCK ---
    const auto kernel_start = std::chrono::steady_clock::now();
    spmv_kernel<<<numBlocks, threadsPerBlock>>>(I, V, C, x, y, num_rows);
    cudaDeviceSynchronize();
    const auto kernel_end = std::chrono::steady_clock::now();

    // --- Time D2H Return Path to Host ---
    cudaMemPrefetchAsync(y, size_vector, cudaCpuDeviceId, NULL);
    cudaDeviceSynchronize(); 
    const auto total_end = std::chrono::steady_clock::now();

    double kernel_sec = std::chrono::duration<double>(kernel_end - kernel_start).count();
    double total_sec = std::chrono::duration<double>(total_end - transfer_start).count();

    // Calculate Real Application Workload Metrics
    const double total_flops = 2.0 * static_cast<double>(nnz);
    const double total_bytes = static_cast<double>(size_I + size_V + size_C + size_vector + size_vector);

    const double kernel_gflops = total_flops / kernel_sec / 1.0e9;
    const double kernel_bandwidth = total_bytes / kernel_sec / 1.0e9;
    const double arithmetic_intensity = total_flops / total_bytes;

    double checksum = 0.0;
    for (int i = 0; i < num_rows; ++i) { checksum += y[i]; }

    std::cout << "Rows: " << num_rows << " | UM Kernel BW: " << kernel_bandwidth << " GB/s"
              << " | Performance: " << kernel_gflops << " GFLOP/s\n";

    // Standardize CSV column formatting matching previous scripts
    csv << machine_name << ","
        << "spMV CSR UM cuda" << ","
        << num_rows << ","
        << arithmetic_intensity << ","
        << kernel_bandwidth << ","
        << kernel_gflops << ","
        << kernel_sec << ","
        << total_sec << "," // Captures prefetch migration overhead side-by-side
        << hw.peak_gflops << ","
        << hw.peak_bandwidth_gb_s << ","
        << hw.ridge_point << ","
        << checksum << "\n";

    cudaFree(I); cudaFree(V); cudaFree(C); cudaFree(x); cudaFree(y);
}

int main() {
    std::string machine_name = get_machine_name();
    HardwareLimits hw = get_gpu_limits();

    // Problem sizes sweep configuration
    std::vector<int> sweep_sizes = {10000, 50000, 100000, 200000};

    std::ifstream check_empty("roofline.csv");
    bool add_header = !check_empty.is_open() || check_empty.peek() == std::ifstream::traits_type::eof();
    check_empty.close();

    std::ofstream csv("roofline.csv", std::ios::app);
    if (add_header) {
        csv << "machine,dwarf name,problem_size,AI,measured_bandwidth(GB/s),"
               "measured_performance(GFLOP/s),kernel_time(s),total_time_with_io(s),"
               "peak_compute(GFLOP/s),peak_bandwidth(GB/s),ridge_point,checksum\n";
    }

    std::cout << "Running SpMV Unified Memory CUDA Sweeps on: " << machine_name << "\n";
    for (int N : sweep_sizes) {
        run_spmv_um_sweep(N, machine_name, csv, hw);
    }

    return 0;
}