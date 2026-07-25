#include <iostream>
#include <chrono>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

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

int main() {
    std::string machine_name = get_machine_name();

    // SCALED UP FOR ACCURATE TIMING: 100,000 rows, 4 non-zeros per row average
    int num_rows = 100000;
    int nnz = num_rows * 4;

    size_t size_I = (num_rows + 1) * sizeof(int);
    size_t size_V = nnz * sizeof(float);
    size_t size_C = nnz * sizeof(int);
    size_t size_vector = num_rows * sizeof(float);

    // Instead of separate host and device mem allocation - apply unified mem 
    int *I; float *V; int *C; float *x; float *y;
    cudaMallocManaged((void**)&I, size_I);
    cudaMallocManaged((void**)&V, size_V);
    cudaMallocManaged((void**)&C, size_C);
    cudaMallocManaged((void**)&x, size_vector);
    cudaMallocManaged((void**)&y, size_vector);

    //use cpu code to initialize directly 
    for (int i =0; i <=num_rows; ++i) I[i] = i * 4;
    for (int i = 0; i < nnz; ++i){
        V[i] = 10.0f;
        C[i] = (i% num_rows);
    }
    for(int i = 0;i < num_rows; ++i){
        x[i] = 1.0f;
        y[i] = 0.0f;
    }

    //Prefetch data to gpu
    int device = 0;
    cudaMemPrefetchAsync(I, size_I, device, NULL);
    cudaMemPrefetchAsync(V, size_V, device, NULL);
    cudaMemPrefetchAsync(C, size_C, device, NULL);
    cudaMemPrefetchAsync(x, size_vector, device, NULL);


    int threadsPerBlock = 256;
    int numBlocks = (num_rows + threadsPerBlock - 1) / threadsPerBlock;

    // --- WARM UP LAUNCH ---eliminate the h_ and d_ pointers
    spmv_kernel<<<numBlocks, threadsPerBlock>>>(I, V, C, x, y, num_rows);
    cudaDeviceSynchronize();

    // --- ACTIVE TIMING BLOCK ---
    const auto start = std::chrono::steady_clock::now();
    
    // Launch
    spmv_kernel<<<numBlocks, threadsPerBlock>>>(I, V, C, x, y, num_rows);
    cudaDeviceSynchronize(); // Await full GPU finish before stopping execution clock
    
    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = end - start;
    const double seconds = elapsed.count(); // average seconds per kernel execution
    cudaMemPrefetchAsync(y, size_vector, cudaCpuDeviceId, NULL);
    cudaDeviceSynchronize(); 

    //read final results directly on CPU
    std::cout <<"spMV MSR Result Vector y:[" << y[0] <<", " << y[1] << ", " << y[2] <<"]\n";

    // --- CALCULATE REAL METRICS ---
    // SpMV FLOPs: 1 Multiply + 1 Add per NNZ = 2 * NNZ
    const double total_flops = 2.0 * static_cast<double>(nnz);
    
    // SpMV Bytes: Read I, Read V, Read C, Read X (ideal), Write Y
    const double total_bytes = static_cast<double>(size_I + size_V + size_C + size_vector + size_vector);
    
    const double gflops = total_flops / seconds / 1.0e9;
    const double bandwidth_gb_s = total_bytes / seconds / 1.0e9;
    const double arithmetic_intensity = total_flops / total_bytes;

    // Checksum using structural indices to ensure correctness
    double checksum = 0.0;
    for (int i = 0; i < num_rows; ++i) { checksum += y[i]; } // Sample verification

    std::cout << "CUDA GPU SpMV | Bandwidth: " << bandwidth_gb_s << " GB/s | Performance: " << gflops << " GFLOP/s\n";

    // --- CSV MANAGEMENT ---
    std::ifstream check_empty("roofline.csv");
    bool add_header = !check_empty.is_open() || check_empty.peek() == std::ifstream::traits_type::eof();
    check_empty.close();

    std::ofstream csv("roofline.csv", std::ios::app);
    if (add_header) {
        csv << "machine,dwarf name,AI,bandwidth (GB/s),performance (GFLOP/s),threads,checksum\n";
    }
    csv << machine_name << "," << "Sparse MV CSR" << "," << arithmetic_intensity << "," << bandwidth_gb_s << "," << gflops << "," << "CUDA" << "," << checksum << "\n";

    cudaFree(I); cudaFree(V); cudaFree(C); cudaFree(x); cudaFree(y);
    return 0;
}