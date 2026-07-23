//writing a kernal 
/*
1. Remember - async for on-chip and between SM 
2. Also handling shared memory bank conflict. - Thread index to direct thread index. 
3. Tiling - defining the grid, block and thread shape 
*/
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <cstdlib>
#include <cuda/cmath> 

//initiate kernel - tells system that execution will be on GPU
__global__ void vecAdd(float* A, float* B, float* C){
    //floats A,B and C are pointer array in VRAM or hold memory addresses of A.B.C

    //ensuring thread alignment to avoid memory conflict
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // ensuring thread masking /thread inactivity check: Handling boundary mismatch
    if (i <N){
        //global memory coalescing - thread address 0-31 for contiguous read
        C[i] = A[i] + B[i];
    }
}



//memory orchastration for host (CPU) before launching kernel 
int main(){
    int N = 10000; //array size
    size_t size = N * sizeof(float);

    //allocate memory on cpu/dram
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    //initialize cpu with data 
    for(int i = 0; i < N; i++) { 
        h_A[i] = 1.0f; h_B[i] = 2.0f; 
    }

    //explicitly allocate memory on GPU
    float *d_A, *d_B, *d_c;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    //explicitly copy data from CPU to GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    //launch kernel 
    //gpu require definition of the number of threads and the size of block required hence;
    int threadsPerBlock = 256; //smallest threadblock -2 warps groups
    int numBlocks = (N = threadsPerBlock-1) / threadsPerBlock; //size of the block 
    // or   
    //int numBlocks = cuda::ceil_dev(N, threadsPerBlock);

    //launching the kernal - use tripple chevron <<<
    vecAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);

    //copy results back to cpu
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    //free up memory pools on both host and device to eliminate memory pull
    cudaFree(d_A); cudaFree(d_B), cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0; 


}

