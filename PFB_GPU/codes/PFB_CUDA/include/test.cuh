# pragma once
#include <iostream>
#include <vector>
#include <cuda_runtime_api.h>
#include <memory.h>
#include <cstdlib>
#include <ctime>
#include <stdio.h>
#include <cuda/cmath>
#include <chrono>
#include "PFB.cuh"

// Wrapping everything in the namespace now
namespace test {
__global__ void GPUvecAdd(float* A, float* B, float* C, int vectorSize)
{
    int workIndex = threadIdx.x + blockIdx.x * blockDim.x;
    // Peform element-wise addition of A and B, store the result in C
    // Each thread computes one element of the output vector C
    if (workIndex < vectorSize) {
    C[workIndex] = A[workIndex] + B[workIndex];
    } else {
        printf("workIndex %d is out of bounds for vector size %d\n", workIndex, vectorSize);
    }
}

void CPUvecAdd(float* A, float* B, float* C, int vectorSize)
{
    for (int i = 0; i < vectorSize; ++i) {
        C[i] = A[i] + B[i];
    }
}

bool compareVectors(float* A, float* B, int vectorSize, float eps=0.000001) {
    for (int i = 0; i < vectorSize; ++i) {
        if (std::abs(A[i] - B[i]) > eps) {
            std::cout << "Mismatch at index " << i << ": " << A[i] << " != " << B[i] << std::endl;
            return false;
        }
    }
    return true;
}

void initVectors(float* A, float* B, int vectorSize) {
    for (int i = 0; i < vectorSize; ++i) {
        A[i] = static_cast<float>(i);
        B[i] = static_cast<float>(2 * i);
    }
}

void unifiedMemCompare(int vectorSize){
    std::cout << "Comparing unified memory approach:" << std::endl;
    auto alloc_start = std::chrono::high_resolution_clock::now();
    // Initialise pointers to all necessary vectors, nullptrs for now
    float* A = nullptr;
    float* B = nullptr;
    float* C_gpu = nullptr;
    // Allocate unified memory for the vectors only for GPU computation, CPU will use standard vectors
    CUDA_CHECK(cudaMallocManaged(&A, vectorSize * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&B, vectorSize * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&C_gpu, vectorSize * sizeof(float)));
    std::vector<float> C_cpu_vector(vectorSize);
    float* C_cpu = C_cpu_vector.data();
    // Initialize A and B on the host
    initVectors(A, B, vectorSize);
    // Launch the GPU kernel to perform vector addition
    int threads = 256;
    int blocks = cuda::ceil_div(vectorSize, threads);
    auto alloc_end = std::chrono::high_resolution_clock::now();
    std::cout << "Memory allocation and initialization took " << std::chrono::duration_cast<std::chrono::milliseconds>(alloc_end - alloc_start).count() << " ms" << std::endl;
    auto gpu_start = std::chrono::high_resolution_clock::now();
    GPUvecAdd<<<blocks, threads>>>(A, B, C_gpu, vectorSize);
    // Wait for the GPU to finish before accessing the results
    CUDA_CHECK(cudaDeviceSynchronize());
    auto gpu_end = std::chrono::high_resolution_clock::now();
    std::cout << "GPU computation took " << std::chrono::duration_cast<std::chrono::milliseconds>(gpu_end - gpu_start).count() << " ms" << std::endl;

    auto cpu_start = std::chrono::high_resolution_clock::now();
    // Perform the same vector addition on the CPU for comparison
    CPUvecAdd(A, B, C_cpu, vectorSize);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::cout << "CPU computation took " << std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_start).count() << " ms" << std::endl;
    // Compare the results from GPU and CPU computations
    if (compareVectors(C_gpu, C_cpu, vectorSize)) {
        std::cout << "Results match!" << std::endl;
    } else {
        std::cout << "Results do not match!" << std::endl;
    }
    // Free the unified memory
    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(B));
    CUDA_CHECK(cudaFree(C_gpu));
}

void explicitMemCompare(int vectorSize){
    std::cout << "Comparing explicit memory management approach:" << std::endl;
    auto alloc_start = std::chrono::high_resolution_clock::now();
    // Initialise pointers to all necessary vectors on the host, nullptrs for now
    float* A = nullptr;
    float* B = nullptr;
    float* C_gpu = nullptr;
    float* C_cpu = nullptr;

    // Initialise pointer to all necessary vectors on the device, nullptrs for now
    float* devA = nullptr;
    float* devB = nullptr;
    float* devC_gpu = nullptr;

    // Allocate memory for the vectors on the host
    CUDA_CHECK(cudaMallocHost(&A, vectorSize * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&B, vectorSize * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&C_gpu, vectorSize * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&C_cpu, vectorSize * sizeof(float)));  // no need for pinned memory for CPU result

    // Initialize A and B on the host
    initVectors(A, B, vectorSize);

    // Allocate memory for the vectors on the device
    CUDA_CHECK(cudaMalloc(&devA, vectorSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&devB, vectorSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&devC_gpu, vectorSize * sizeof(float)));

    // Copy A and B from host to device
    CUDA_CHECK(cudaMemcpy(devA, A, vectorSize * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(devB, B, vectorSize * sizeof(float), cudaMemcpyHostToDevice));

    // Launch the GPU kernel to perform vector addition
    int threads = 256;
    int blocks = cuda::ceil_div(vectorSize, threads);
    auto alloc_end = std::chrono::high_resolution_clock::now();
    std::cout << "Memory allocation and data transfer took " << std::chrono::duration_cast<std::chrono::milliseconds>(alloc_end - alloc_start).count() << " ms" << std::endl;
    auto gpu_start = std::chrono::high_resolution_clock::now();
    GPUvecAdd<<<blocks, threads>>>(devA, devB, devC_gpu, vectorSize);
    // Wait for the GPU to finish before accessing the results
    CUDA_CHECK(cudaDeviceSynchronize());
    auto gpu_end = std::chrono::high_resolution_clock::now();
    std::cout << "GPU computation took " << std::chrono::duration_cast<std::chrono::milliseconds>(gpu_end - gpu_start).count() << " ms " << std::endl;

    // Copy the result from device to host
    CUDA_CHECK(cudaMemcpy(C_gpu, devC_gpu, vectorSize * sizeof(float), cudaMemcpyDeviceToHost));

    auto cpu_start = std::chrono::high_resolution_clock::now();
    // Perform the same vector addition on the CPU for comparison
    CPUvecAdd(A, B, C_cpu, vectorSize);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::cout << "CPU computation took " << std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_start).count() << " ms" << std::endl;

    // Compare the results from GPU and CPU computations
    if (compareVectors(C_gpu, C_cpu, vectorSize)) {
        std::cout << "Results match!" << std::endl;
    } else {
        std::cout << "Results do not match!" << std::endl;
    }

    // Free the allocated memory
    CUDA_CHECK(cudaFreeHost(A));
    CUDA_CHECK(cudaFreeHost(B));
    CUDA_CHECK(cudaFreeHost(C_gpu));
    CUDA_CHECK(cudaFreeHost(C_cpu));
    CUDA_CHECK(cudaFree(devA));
    CUDA_CHECK(cudaFree(devB));
    CUDA_CHECK(cudaFree(devC_gpu));
}
} // namespace test