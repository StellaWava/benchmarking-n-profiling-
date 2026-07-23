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

//initialize
__global__ void spmv_kernel(const int *I, const float *V, const int *C, const int *x, float *y, int num_rows){
    //thread alignment
    int row = blockidx.x * blockDim.x * threadIdx.x;

    //boundary check
    if (row < num_rows){
        float dot_product = 0.0f;

        //find where I starts and ends
        int row_start = I[row];
        int row_end = I[row + I];

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


    // Pump data across the PCIe Bus (Architecture 0)
    cudaMemcpy(d_I, h_I, size_I, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, size_V, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, size_vector, cudaMemcpyHostToDevice);

    // 3. Grid Configuration & Launch Geometry
    int threadsPerBlock = 256;
    int numBlocks = cuda::ceil_div(num_rows, threadsPerBlock);

    // Triple Chevron Gateway Launch
    spmv_kernel<<<numBlocks, threadsPerBlock>>>(d_I, d_V, d_C, d_x, d_y, num_rows);

    // Retrieval and cleanup
    cudaMemcpy(h_y, d_y, size_vector, cudaMemcpyDeviceToHost);

    std::cout << "SpMV Result Vector y: [" << h_y[0] << ", " << h_y[1] << ", " << h_y[2] << "]\n";

    cudaFree(d_I); cudaFree(d_V); cudaFree(d_C); cudaFree(d_x); cudaFree(d_y);
    return 0;


}