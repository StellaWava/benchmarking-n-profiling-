// File for SpMv Profiling - SpMV is mapped using CSR approach
//CSR approach is row index(I), row value(V) NNS, and column index (C) NNS
//y = A.x where y and x are dense inputs but A is sparce.

/*
*/

#include <iostream>
#include <chrono>
#include <vector>
#include <chrono>
#include <fstream>
#include <cstdlib>
#include <cmath> 

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

//initialize
__global__ void spmv_kernel(const int *I, const float *V, const int *C, const float *x, float *y, int num_rows){
    //thread alignment
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    //boundary check
    if (row < num_rows){
        float dot_product = 0.0f;

        //find where I starts and ends
        int row_start = I[row];
        int row_end = I[row + 1];

        //loop through the NNZ elements in the row
        for (int element = row_start; element < row_end; ++element){
            int col = C[element];
            dot_product += V[element] * x[col];
        }

        //store accumulated dot product into the y 
        y[row] = dot_product;

    }
    
}

//cpu to host mapping 
int main() {

    // Example: A tiny 3x3 sparse matrix with 4 non-zero elements (NNZ = 4)
    int num_rows = 3;
    int nnz = 4;

    //execute machine function
    std::string machine_name = get_machine_name();

    size_t size_I = (num_rows + 1) * sizeof(int);
    size_t size_V = nnz * sizeof(float);
    size_t size_C = nnz * sizeof(int);
    size_t size_vector = num_rows * sizeof(float);

    // Host allocations (CPU RAM)
    int h_I[] = {0, 2, 3, 4};             // Row pointers
    float h_V[] = {10.0f, 20.0f, 30.0f, 40.0f}; // Non-zero values
    int h_C[] = {0, 2, 1, 2};             // Column indices
    float h_x[] = {1.0f, 1.0f, 1.0f};     // Input vector x
    float h_y[3] = {0.0f};                // Output vector y

    // 2. Host Allocation & Memory Transport (Device VRAM Space)
    int *d_I; float *d_V; int *d_C; float *d_x; float *d_y;
    cudaMalloc((void**)&d_I, size_I);
    cudaMalloc((void**)&d_V, size_V);
    cudaMalloc((void**)&d_C, size_C);
    cudaMalloc((void**)&d_x, size_vector);
    cudaMalloc((void**)&d_y, size_vector);


    // Pump data across the PCIe Bus (Architecture 0) | Copy data from Host (RAM) to Device (VRAM)
    cudaMemcpy(d_I, h_I, size_I, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, size_V, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, size_vector, cudaMemcpyHostToDevice);

    // 3. Grid Configuration & Launch Geometry
    int threadsPerBlock = 256;
    int numBlocks = (num_rows + threadsPerBlock -1) / threadsPerBlock;
    //int numBlocks = cuda::ceil_div(num_rows, threadsPerBlock);

    // Triple Chevron Gateway Launch cuda kernel 
    spmv_kernel<<<numBlocks, threadsPerBlock>>>(d_I, d_V, d_C, d_x, d_y, num_rows);
    cudaDeviceSynchronize(); //warm up 

    // 5. Active Timing Block
    const auto start = std::chrono::steady_clock::now();
    //launch kernel 
    spmv_kernel<<<numBlocks, threadsPerBlock>>>(d_I, d_V, d_C, d_x, d_y, num_rows);
    
    // wait for the GPU to finish computation before stopping the clock
    cudaDeviceSynchronize(); 

    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = end - start;
    const double seconds = elapsed.count();

    // copy final results to back host 
    cudaMemcpy(h_y, d_y, size_vector, cudaMemcpyDeviceToHost);
    //get results
    std::cout << "SpMV Result Vector y: [" << h_y[0] << ", " << h_y[1] << ", " << h_y[2] << "]\n";


    
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
        << "spMV cuda" << ","
        << arithmetic_intensity << ","
        << bandwidth_gb_s << ","
        << gflops << ","
        << "CUDA" << ","
        << checksum << "\n";

    // 9. Clean up GPU memory allocations
    cudaFree(d_I); 
    cudaFree(d_V); 
    cudaFree(d_C); 
    cudaFree(d_x); 
    cudaFree(d_y);

    return 0;

}



/*
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

    // Host allocations
    std::vector<int> h_I(num_rows + 1);
    std::vector<float> h_V(nnz, 10.0f);
    std::vector<int> h_C(nnz, 0);
    std::vector<float> h_x(num_rows, 1.0f);
    std::vector<float> h_y(num_rows, 0.0f);

    // Initialize mock CSR structure (Each row gets 4 elements)
    for(int i = 0; i <= num_rows; ++i) h_I[i] = i * 4;
    for(int i = 0; i < nnz; ++i) h_C[i] = (i % num_rows); 

    // Device Allocation
    int *d_I; float *d_V; int *d_C; float *d_x; float *d_y;
    cudaMalloc((void**)&d_I, size_I);
    cudaMalloc((void**)&d_V, size_V);
    cudaMalloc((void**)&d_C, size_C);
    cudaMalloc((void**)&d_x, size_vector);
    cudaMalloc((void**)&d_y, size_vector);

    // Host to Device Copy
    cudaMemcpy(d_I, h_I.data(), size_I, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V.data(), size_V, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C.data(), size_C, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x.data(), size_vector, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int numBlocks = (num_rows + threadsPerBlock - 1) / threadsPerBlock;

    // --- WARM UP LAUNCH ---
    spmv_kernel<<<numBlocks, threadsPerBlock>>>(d_I, d_V, d_C, d_x, d_y, num_rows);
    cudaDeviceSynchronize();

    // --- ACTIVE TIMING BLOCK ---
    const auto start = std::chrono::steady_clock::now();
    
    // Loop 100 times to get a stable average and overcome system clock limits
    int iterations = 100;
    for(int iter = 0; iter < iterations; ++iter) {
        spmv_kernel<<<numBlocks, threadsPerBlock>>>(d_I, d_V, d_C, d_x, d_y, num_rows);
    }
    cudaDeviceSynchronize(); // Await full GPU finish before stopping execution clock
    
    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = end - start;
    const double seconds = elapsed.count() / iterations; // average seconds per kernel execution

    // Copy back final results
    cudaMemcpy(h_y.data(), d_y, size_vector, cudaMemcpyDeviceToHost);

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
    for (int i = 0; i < 5; ++i) { checksum += h_y[i]; } // Sample verification

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

    cudaFree(d_I); cudaFree(d_V); cudaFree(d_C); cudaFree(d_x); cudaFree(d_y);
    return 0;
}
*/