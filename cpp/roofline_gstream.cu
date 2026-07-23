#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <cstdlib>

//get system info if avaliable 
std::string get_machine_name() {
#if defined(_WIN32) || defined(_WIN64)
    const char* env = std::getenv("COMPUTERNAME");
#else
    const char* env = std::getenv("HOSTNAME");
    if (!env) env = std::getenv("NAME");
#endif
    return env ? std::string(env) : "GPU_Machine";
}


// The CUDA "Kernel" - This runs entirely on the GPU cores
__global__ void stream_triad_kernel(const double* A, const double* B, double* C, double scalar, size_t N) {
    // Calculate the unique global element index for this specific GPU thread
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Ensure we don't read out of bounds if N is not a perfect multiple of block size
    if (i < N) {
        C[i] = A[i] + B[i] * scalar;
    }
}


//orchastration from host(cpu) to device (gpu)
int main() {
    //define array size /entry variables 
    constexpr std::size_t N = 50000000;
    constexpr double scalar = 2.0;
    
    //execute machine function
    std::string machine_name = get_machine_name();

    // 1. Allocate Host (CPU) memory vectors
    std::vector<double> h_A(N, 5.0);
    std::vector<double> h_B(N, 10.0);
    std::vector<double> h_C(N, 0.0);

    // 2. Allocate Device (GPU VRAM) pointers
    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * sizeof(double));
    cudaMalloc(&d_B, N * sizeof(double));
    cudaMalloc(&d_C, N * sizeof(double));

    // 3. Copy data from Host (RAM) to Device (VRAM)
    cudaMemcpy(d_A, h_A.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    // 4. Configure GPU execution grid - threads perblock and number of blocks
    int threads_per_block = 256; 
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;
    //int blocks_per_grid = cuda::ceil_dev(N, threads_per_block);

    // Warm-up pass to boot up GPU clocks
    stream_triad_kernel<<<blocks_per_grid, threads_per_block>>>(d_A, d_B, d_C, scalar, N);
    cudaDeviceSynchronize(); // Wait for warm-up to finish

    // 5. Active Timing Block
    const auto start = std::chrono::steady_clock::now();

    // Launch the CUDA kernel on the GPU
    stream_triad_kernel<<<blocks_per_grid, threads_per_block>>>(d_A, d_B, d_C, scalar, N);
    
    // Explicitly wait for the GPU to finish computation before stopping the clock
    cudaDeviceSynchronize(); 

    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = end - start;
    const double seconds = elapsed.count();

    // 6. Copy final results back to Host (RAM) for validation
    cudaMemcpy(h_C.data(), d_C, N * sizeof(double), cudaMemcpyDeviceToHost);

    // 7. Calculate Metrics
    constexpr double flops_per_element = 2.0;
    const double total_flops = static_cast<double>(N) * flops_per_element;
    constexpr double bytes_per_element = 3.0 * sizeof(double);
    const double total_bytes = static_cast<double>(N) * bytes_per_element;

    const double gflops = total_flops / seconds / 1.0e9;
    const double bandwidth_gb_s = total_bytes / seconds / 1.0e9;
    const double arithmetic_intensity = flops_per_element / bytes_per_element;

    double checksum = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        checksum += h_C[i];
    }

    std::cout << "CUDA GPU | Bandwidth: " << bandwidth_gb_s << " GB/s"
              << " | Performance: " << gflops << " GFLOP/s\n";

    // 8. CSV Management
    std::ifstream check_empty("roofline.csv");
    bool add_header = !check_empty.is_open() || check_empty.peek() == std::ifstream::traits_type::eof();
    check_empty.close();

    std::ofstream csv("roofline.csv", std::ios::app);
    if (add_header) {
        csv << "machine,dwarf name,AI,bandwidth (GB/s),performance (GFLOP/s),threads,checksum\n";
    }

    csv << machine_name << ","
        << "stream triad direct cuda" << ","
        << arithmetic_intensity << ","
        << bandwidth_gb_s << ","
        << gflops << ","
        << "CUDA" << ","
        << checksum << "\n";

    // 9. Clean up GPU memory allocations
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}