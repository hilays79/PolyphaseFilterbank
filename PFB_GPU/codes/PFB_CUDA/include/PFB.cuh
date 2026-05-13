// This header file contains the declaration of the PFB class and its member functions.
// Since I want variable NBIT implementation, all the functions are templates, defined right here.

#pragma once
#include <iostream>
#include <vector>
#include <cuda/std/complex>
#include <cuda_runtime_api.h>
#include <cuda/cmath>
#include "dsp.hpp"
#include "kernels.cuh" 
#include <chrono>
#include <type_traits> 
#include <cufft.h>     

#define CUDA_CHECK(expr_to_check) do { \
    cudaError_t result = (expr_to_check); \
    if (result != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << ": " \
                  << cudaGetErrorString(result) << std::endl; \
        exit(1); \
    } \
} while (0)

#define CUFFT_CHECK(expr_to_check) do { \
    cufftResult result = (expr_to_check); \
    if (result != CUFFT_SUCCESS) { \
        std::cerr << "cuFFT error in " << __FILE__ << ":" << __LINE__ << ": " \
                  << result << std::endl; \
        exit(1); \
    } \
} while (0)

template <typename T>
class PFB {
    private:
        int n_taps; 
        int n_chan; 
        int n_windows; 
        int n_integrations; 
        bool atomic; 
        int n_batches; 
        
        int filter_length; 
        
        // --- Batch Tracking Variables ---
        int N_input_time_blocks;     
        int N_out_blocks_total;      
        int N_out_blocks_batch;      
        int N_in_blocks_batch;       
        int i_batch_max; 
        
        int input_length; 
        int output_length; 
        
        int valid_output_length; 
        
        int n_time_blocks; 

        T* d_coeffs;
        cuda::std::complex<T>* d_inputData; 
        cuda::std::complex<T>* d_outputDataComplex; 
        T* d_outputDataReal;
        void* d_work_area;

        double setup_time;
        double exec_time;
        double fir_time;
        double fft_time;

        cufftHandle fft_plan;  
        cufftType fft_type;  

    public:
        PFB(int M, int N, int W, int n_integrations_in, int n_batches, bool atomic);  
        ~PFB();  

        void FIR(int i_batch); 
        void FFT(int i_batch); 
        void PSD(int i_batch); 
        
        void execute_PFB(std::complex<T>* h_inputData); 
        void getOutput(T* h_outputData);
};

// --- IMPLEMENTATION OF CONSTRUCTOR AND DESTRUCTOR START ---
template <typename T>
PFB<T>::PFB(int M, int N, int W, int n_integrations_in, int n_batches, bool atomic) : 
    n_taps(M), n_chan(N), n_windows(W), n_batches(n_batches), atomic(atomic), setup_time(0.0), exec_time(0.0), fir_time(0.0), fft_time(0.0)
{ 
    auto s_start = std::chrono::high_resolution_clock::now(); 
    
    // Per requirements, hardcode n_integrations to 1 for calculation simplicity
    n_integrations = 1;

    filter_length = n_taps * n_chan; 
    
    // --- Whiteboard Algorithm Padding & Batching ---
    N_input_time_blocks = n_taps * n_windows; 
    N_out_blocks_total = N_input_time_blocks - n_taps + 1; 

    // ERROR CHECK 1: Ensure n_batches <= N - M + 1
    if (n_batches > N_out_blocks_total) {
        std::cerr << "ERROR: n_batches (" << n_batches << ") must be <= N - M + 1 (" << N_out_blocks_total << ")." << std::endl;
        exit(1);
    }
    
    // N_out,blocks,batch = ceil((N - M + 1) / n_batches) using cuda::ceil_div
    N_out_blocks_batch = cuda::ceil_div(N_out_blocks_total, n_batches);
    
    // N_in,blocks,batch = N_out,blocks,batch + M - 1
    N_in_blocks_batch = N_out_blocks_batch + n_taps - 1;
    
    // Calculate i_batch_max = ceil[ (N - N_in,blocks,batch) / N_out,blocks,batch ]
    int numerator = N_input_time_blocks - N_in_blocks_batch;
    i_batch_max = cuda::ceil_div(numerator, N_out_blocks_batch);
    
    // ERROR CHECK 2: Ensure i_batch_max <= n_batches - 1
    if (i_batch_max > n_batches - 1) {
        std::cerr << "ERROR: Calculated i_batch_max (" << i_batch_max << ") must be <= n_batches - 1 (" << n_batches - 1 << ")." << std::endl;
        exit(1);
    }
    
    int n_actual_batches = i_batch_max + 1;

    // As proven by the math, (i_max + 2 - M)*P collapses to exactly:
    int N_padded_out_blocks = n_actual_batches * N_out_blocks_batch;
    int N_padded_input_blocks = N_padded_out_blocks + n_taps - 1;

    input_length = N_padded_input_blocks * n_chan; 
    output_length = N_padded_out_blocks * n_chan; 
    n_time_blocks = N_padded_out_blocks;
    
    // Exact unpadded length for getOutput() retrieval (since integrations = 1)
    valid_output_length = N_out_blocks_total * n_chan;

    // Calculate how many zeros we actually added
    int padded_input_blocks_added = N_padded_input_blocks - N_input_time_blocks;

    // Allocate memory
    CUDA_CHECK(cudaMalloc(&d_coeffs, filter_length * sizeof(T)));  
    CUDA_CHECK(cudaMalloc(&d_inputData, input_length * sizeof(cuda::std::complex<T>)));  
    CUDA_CHECK(cudaMalloc(&d_outputDataComplex, output_length * sizeof(cuda::std::complex<T>)));  
    CUDA_CHECK(cudaMalloc(&d_outputDataReal, output_length * sizeof(T)));  
    
    // FFT Plan creation
    if (std::is_same<T, float>::value) {
        fft_type = CUFFT_C2C; 
    } else if (std::is_same<T, double>::value) {
        fft_type = CUFFT_Z2Z; 
    } else {
        std::cerr << "Unsupported data type for FFT plan. Only float and double are supported." << std::endl;
        exit(1);
    }
    
    int n[] = {n_chan};

    // ---------- MANUAL PLANNING ------------
    // For cufftMakePlanMany(), plan creation and memory allocation are separate steps, so we create the plan first and then allocate memory.
    // Disabling auto-allocation
    CUFFT_CHECK(cufftCreate(&fft_plan));
    CUFFT_CHECK(cufftSetAutoAllocation(fft_plan, 0));
    size_t work_area_size = 0;  // Variable to hold the required work area size for the FFT plan
    // Create an emply plan first
    

    // Make the plan with the specified parameters, but with zero work area size to query the required size
    CUFFT_CHECK(cufftMakePlanMany(fft_plan, 1, n, 
                                    NULL, 1, n_chan, 
                                    NULL, 1, n_chan, 
                                    fft_type, N_out_blocks_batch, &work_area_size));


    CUFFT_CHECK(cufftGetSize(fft_plan, &work_area_size));

    // Allocate the required work area on the device
    std::cout << "Required work area size for FFT plan: " << work_area_size << " bytes" << std::endl;
    d_work_area = NULL;
    if (work_area_size > 0) {
        CUDA_CHECK(cudaMalloc(&d_work_area, work_area_size));
        // Assign custom memory buffer to the plan
        CUFFT_CHECK(cufftSetWorkArea(fft_plan, d_work_area));
    }

    // ------------ AUTOMATIC PLANNING ----------
    // // cufftPlanMany() does everything in one step, including creating the plan, assigning that plan to the passed pointer, allocating memory, and all.
    // cufft_result = CUFFT_CHECK(cufftPlanMany(&fft_plan, 1, n, 
    //                         NULL, 1, n_chan, 
    //                         NULL, 1, n_chan, 
    //                         fft_type, N_out_blocks_batch)); 

    auto s_end = std::chrono::high_resolution_clock::now(); 
    setup_time += std::chrono::duration<double>(s_end - s_start).count();
    
    // Requested Print Statements
    std::cout << "GPU PFB initialized." << std::endl;
    std::cout << "Zero-padded input blocks added: " << padded_input_blocks_added << std::endl;
    std::cout << "Actual batches executed: " << n_actual_batches << std::endl;
}

template <typename T>
PFB<T>::~PFB() {
    auto s_start = std::chrono::high_resolution_clock::now(); 

    CUDA_CHECK(cudaFree(d_coeffs));
    CUDA_CHECK(cudaFree(d_inputData));
    CUDA_CHECK(cudaFree(d_outputDataComplex));
    CUDA_CHECK(cudaFree(d_outputDataReal));
    
    cufftDestroy(fft_plan);
    // Free the manually allocated work area
    if (d_work_area) {
        CUDA_CHECK(cudaFree(d_work_area));
    }
    auto s_end = std::chrono::high_resolution_clock::now(); 
    setup_time += std::chrono::duration<double>(s_end - s_start).count();
}

// --- IMPLEMENTATION OF MEMBER FUNCTIONS START ---

template <typename T>
void PFB<T>::execute_PFB(std::complex<T>* h_inputData) {
    auto s_start = std::chrono::high_resolution_clock::now(); 
    
    // 1) Zero-pad the entire device input buffer first, technically only the last padded_input_blocks_added blocks need to be zero, but this is simpler.
    // POSSIBLE OPTIMISATION: Zero-pad just the end.
    CUDA_CHECK(cudaMemset(d_inputData, 0, input_length * sizeof(cuda::std::complex<T>)));
    
    // 2) Copy the valid input blocks into the start of the padded array
    CUDA_CHECK(cudaMemcpy(d_inputData, h_inputData, N_input_time_blocks * n_chan * sizeof(std::complex<T>), cudaMemcpyHostToDevice));

    // 3) Generate filter coefficients 
    int half_filter_length = cuda::ceil_div(filter_length, 2); 
    std::vector<T> win_coeffs = windowing::generate_win_coeffs<T>(n_taps, n_chan);  
    cudaHostRegister(win_coeffs.data(), win_coeffs.size() * sizeof(T), cudaHostRegisterDefault);
    CUDA_CHECK(cudaMemcpy(d_coeffs, win_coeffs.data(), half_filter_length * sizeof(T), cudaMemcpyHostToDevice));
    cudaHostUnregister(win_coeffs.data());

    auto s_end = std::chrono::high_resolution_clock::now();
    setup_time += std::chrono::duration<double>(s_end - s_start).count();

    // 4) Execute PFB in batches up to i_batch_max
    for (int i_batch = 0; i_batch <= i_batch_max; ++i_batch) {
        FIR(i_batch); 
        FFT(i_batch); 
        PSD(i_batch); 
    }

    std::cout << "GPU_SETUP_TIME: " << setup_time << "\n";
    std::cout << "GPU_EXEC_TIME: " << exec_time << "\n";
    std::cout << "GPU_FIR_TIME: " << fir_time << "\n";
    std::cout << "GPU_FFT_TIME: " << fft_time << "\n";
    std::cout << "==================================\n";
}

template <typename T>
void PFB<T>::FIR(int i_batch) {
    auto e_start = std::chrono::high_resolution_clock::now(); 
    
    // Pointer offset moves forward by the exact output size per batch
    int in_offset = i_batch * N_out_blocks_batch * n_chan;
    int out_offset = i_batch * N_out_blocks_batch * n_chan;

    int threadsPerBlock = std::min(n_chan, 1024); 
    dim3 blockDim(threadsPerBlock); 
    int numGridBlocks_x = cuda::ceil_div(n_chan, threadsPerBlock); 
    int numGridBlocks_y = N_out_blocks_batch; 

    if (atomic) {
        dim3 gridDim(numGridBlocks_y, numGridBlocks_x, n_taps); 
        FIR_atomic_convolution<T><<<gridDim, blockDim>>>(d_inputData + in_offset, d_coeffs, d_outputDataComplex + out_offset, n_taps, n_chan, N_out_blocks_batch, filter_length);
    } else {
        dim3 gridDim(numGridBlocks_y, numGridBlocks_x); 
        FIR_convolution<T><<<gridDim, blockDim>>>(d_inputData + in_offset, d_coeffs, d_outputDataComplex + out_offset, n_taps, n_chan, N_out_blocks_batch, filter_length);
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    auto e_end = std::chrono::high_resolution_clock::now();
    exec_time += std::chrono::duration<double>(e_end - e_start).count();
    fir_time += std::chrono::duration<double>(e_end - e_start).count();
}

template <typename T>
void PFB<T>::FFT(int i_batch) {
    auto e_start = std::chrono::high_resolution_clock::now();
    
    int out_offset = i_batch * N_out_blocks_batch * n_chan;

    if (fft_type == CUFFT_C2C) {
        cufftExecC2C(fft_plan, (cufftComplex*)(d_outputDataComplex + out_offset), (cufftComplex*)(d_outputDataComplex + out_offset), CUFFT_FORWARD); 
    } else if (fft_type == CUFFT_Z2Z) {
        cufftExecZ2Z(fft_plan, (cufftDoubleComplex*)(d_outputDataComplex + out_offset), (cufftDoubleComplex*)(d_outputDataComplex + out_offset), CUFFT_FORWARD); 
    } 
    
    CUDA_CHECK(cudaDeviceSynchronize());
    auto e_end = std::chrono::high_resolution_clock::now();
    exec_time += std::chrono::duration<double>(e_end - e_start).count();
    fft_time += std::chrono::duration<double>(e_end - e_start).count();
}

template <typename T>
void PFB<T>::PSD(int i_batch) {
    auto e_start = std::chrono::high_resolution_clock::now();
    
    int out_offset = i_batch * N_out_blocks_batch * n_chan;

    int threadsPerBlock = std::min(n_chan, 1024); 
    dim3 blockDim(threadsPerBlock); 
    int numGridBlocks_x = cuda::ceil_div(n_chan, threadsPerBlock); 
    int numGridBlocks_y = N_out_blocks_batch; 
    dim3 gridDim(numGridBlocks_y, numGridBlocks_x); 

    PSD_integration<T><<<gridDim, blockDim>>>(d_outputDataComplex + out_offset, d_outputDataReal + out_offset, n_integrations, n_chan, N_out_blocks_batch, N_out_blocks_batch);

    CUDA_CHECK(cudaDeviceSynchronize());
    auto e_end = std::chrono::high_resolution_clock::now();
    exec_time += std::chrono::duration<double>(e_end - e_start).count();
}

template <typename T>
void PFB<T>::getOutput(T* h_outputData) {
    auto e_start = std::chrono::high_resolution_clock::now();

    // Copy ONLY the valid unpadded samples back into the main output and host
    CUDA_CHECK(cudaMemcpy(h_outputData, d_outputDataReal, valid_output_length * sizeof(T), cudaMemcpyDeviceToHost));
    
    auto e_end = std::chrono::high_resolution_clock::now();
    exec_time += std::chrono::duration<double>(e_end - e_start).count();
}