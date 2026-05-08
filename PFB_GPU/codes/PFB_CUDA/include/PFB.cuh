// This header file contains the declaration of the PFB class and its member functions.
// Since I want variable NBIT implementation, all the functions are templates, defined right here.

#pragma once
#include <iostream>
#include <cuda/std/complex>
#include <cuda_runtime_api.h>
#include <cuda/cmath>
#include "dsp.hpp"
#include "kernels.cuh" // Include the header file for CUDA kernels, as they cannot be members of a class
#include <chrono>
#include <type_traits> // Required for checking float vs double in FFT plan
#include <cufft.h>     // The cuFFT library

// macro for catching allocation errors
#define CUDA_CHECK(expr_to_check) do { \
    cudaError_t result = (expr_to_check); \
    if (result != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << ": " \
                  << cudaGetErrorString(result) << std::endl; \
        exit(1); \
    } \
} while (0)

// Class declaration. T will either be float or double depending on the precision asked for in main.cu
template <typename T>
class PFB {
    private:
        // While these parameters are private, they can be initialized through the constructor and accessed through public getter functions if needed.
        // Kept private to not alter them accidentally after initialization, as they are fundamental to the functioning of the PFB algorithm.
        // PFB parameters
        int n_taps; // number of taps in the filter
        int n_chan; // number of channels in the output
        int n_windows; // number of windows in the input data
        int n_integrations; // number of integrations for the PSD stage
        int filter_length; // total length of the filter coefficients array (M*N)
        int input_length; // total length of the input data array (M*W*N)
        int output_length; // total length of the output data array ((M*W - M + 1)*N)
        int output_integrated_length; // total length of the integrated output data array, which will be the final output of the PFB algorithm after the PSD stage (output_length / n_integrations)
        // Necessary pointers for the filter coefficients and the input data, they will only be initialised on the device.
        T* d_coeffs; // filter coefficients (real-valued)
        cuda::std::complex<T>* d_inputData; // input data (complex-valued)
        cuda::std::complex<T>* d_outputDataComplex; // output data (complex-valued): to be used for FIR, FFT stages
        T* d_outputDataReal; // output data (real-valued) to be used for the final PSD output after the completion of the PFB algorithm

        int n_time_blocks; // number of time blocks in the output data, calculated based on the input signal length and the number of taps, used for setting up kernel launch parameters
        int n_integrated_time_blocks; // number of time blocks in the integrated output data

        // --- Timing Variables ---
        double setup_time;
        double exec_time;

        // --- FFT Plan variables would go here ---
        cufftHandle fft_plan;  // pointer to the FFT plan, which will be initialized in the constructor and used in the FFT stage of the PFB algorithm
        cufftType fft_type;  // variable to hold the type of FFT plan (CUFFT_C2C for single precision, CUFFT_Z2Z for double precision)
    public:
        // Constructor and destructor
        PFB(int M, int N, int W, int n_integrations);  // Constructor to initialize the PFB parameters and allocate memory on the device
        ~PFB();  // Destructor to free the allocated memory on the device

        // Member functions for the PFB algorithm
        void FIR(std::complex<T>* h_inputData); // Function to perform the FIR filtering stage of the PFB algorithm

        void FFT(); // Function to perform the FFT stage of the PFB algorithm using the cuFFT library and the fft_plan created in the constructor.

        void PSD(); // Function to perform the PSD stage of the PFB algorithm, which will take the output from the FFT stage to compute the final power spectral density output.
        void execute_PFB(std::complex<T>* h_inputData); // Function to perform the entire PFB algorithm, which will call the FIR and FFT and PSD functions in sequence.

        // Retrieve the output data from the Device back to the Host
        void getOutput(T* h_outputData);
};

// --- IMPLEMENTATION OF CONSTRUCTOR AND DESTRUCTOR START ---
template <typename T>
PFB<T>::PFB(int M, int N, int W, int n_integrations) : n_taps(M), n_chan(N), n_windows(W), n_integrations(n_integrations), setup_time(0.0), exec_time(0.0) // Initialize the PFB parameters through the constructor initializer list (the command after the closing parenthesis of the constructor parameters)
{ 
    auto s_start = std::chrono::high_resolution_clock::now(); // START SETUP TIMING
    // The following variables are assignments of the private variables in the body; thus, they are not in the initializer list.
    filter_length = n_taps * n_chan; // Calculate the total length of the filter coefficients array based on the number of taps and channels
    input_length =  n_taps * n_windows * n_chan; // Calculate the total length of the input data array based on the number of windows, channels, and taps (units of time)
    output_length = (n_taps * n_windows - n_taps + 1) * n_chan; // Calculate the total length of the output data array based on the number of taps, windows, and channels (units of time blocks)
    n_time_blocks = output_length / n_chan; // Calculate the number of time blocks in the output data based on the total output length and the number of channels
    n_integrated_time_blocks = n_time_blocks / n_integrations; // Calculate the number of time blocks in the integrated output data based on the total integrated output length and the number of channels
    output_integrated_length = n_integrated_time_blocks * n_chan; // Calculate the total length of the integrated output data array based on the number of integrated time blocks and channels
    // Allocate memory for filter coefficients and input data on the device
    CUDA_CHECK(cudaMalloc(&d_coeffs, filter_length * sizeof(T)));  // Initialization for the d_coeffs is done by cudaMalloc itself, so initializer list is not needed here.
    CUDA_CHECK(cudaMalloc(&d_inputData, input_length * sizeof(cuda::std::complex<T>)));  // Initialization for the d_inputData is done by cudaMalloc itself
    CUDA_CHECK(cudaMalloc(&d_outputDataComplex, output_length * sizeof(cuda::std::complex<T>)));  // Initialization for the d_outputDataComplex is done by cudaMalloc itself
    CUDA_CHECK(cudaMalloc(&d_outputDataReal, output_integrated_length * sizeof(T)));  // Initialization for the d_outputDataReal is done by cudaMalloc itself
    // FFT Plan creation here.
    if (std::is_same<T, float>::value) {
        fft_type = CUFFT_C2C; // Complex-to-complex FFT for single precision
    } else if (std::is_same<T, double>::value) {
        fft_type = CUFFT_Z2Z; // Complex-to-complex FFT for double precision
    } else {
        std::cerr << "Unsupported data type for FFT plan. Only float and double are supported." << std::endl;
        exit(1);
    }
    int n[] = {n_chan};
    // Syntax: handle, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch
    cufftPlanMany(&fft_plan, 1, n, 
                              NULL, 1, n_chan, // Input layout
                              NULL, 1, n_chan, // Output layout
                              fft_type, n_time_blocks); // Number of FFTs to perform in a batch is equal to the number of time blocks in the output data
    // std::cout << "FFT plan created successfully." << std::endl;

    auto s_end = std::chrono::high_resolution_clock::now(); // END SETUP TIMING
    setup_time += std::chrono::duration<double>(s_end - s_start).count();
    std::cout << "GPU PFB initialized with M=" << n_taps << ", N=" << n_chan << ", W=" << n_windows << std::endl;
}

template <typename T>
PFB<T>::~PFB() {
    auto s_start = std::chrono::high_resolution_clock::now(); // START SETUP TIMING

    CUDA_CHECK(cudaFree(d_coeffs));
    CUDA_CHECK(cudaFree(d_inputData));
    CUDA_CHECK(cudaFree(d_outputDataComplex));
    CUDA_CHECK(cudaFree(d_outputDataReal));
    
    cufftDestroy(fft_plan); // Destroy the FFT plan to free up resources
    auto s_end = std::chrono::high_resolution_clock::now(); // END SETUP TIMING
    setup_time += std::chrono::duration<double>(s_end - s_start).count();

    // std::cout << "GPU PFB memory freed." << std::endl;


}

// --- IMPLEMENTATION OF CONSTRUCTOR AND DESTRUCTOR END ---

// --- IMPLEMENTATION OF MEMBER FUNCTIONS START ---
template <typename T>
void PFB<T>::FIR(std::complex<T>* h_inputData) {
    auto s_start = std::chrono::high_resolution_clock::now(); // START SETUP TIMING
    // Copy the input data from the host (supplied by main.cu) to the device
    CUDA_CHECK(cudaMemcpy(d_inputData, h_inputData, input_length * sizeof(std::complex<T>), cudaMemcpyHostToDevice));
    // std::cout << "Input data copied to device." << std::endl;
    // Generate the filter coefficients on the host using PFB_CPU and copy them to the device
    int half_filter_length = (filter_length + 1) / 2; // ceil division handles both even/odd lengths
    std::vector<T> win_coeffs = windowing::generate_win_coeffs<T>(n_taps, n_chan);  // n_taps and n_chan known here as they were initialized in the constructor and are private members of the class
    cudaHostRegister(win_coeffs.data(), win_coeffs.size() * sizeof(T), cudaHostRegisterDefault);
    CUDA_CHECK(cudaMemcpy(d_coeffs, win_coeffs.data(), half_filter_length * sizeof(T), cudaMemcpyHostToDevice));
    // std::cout << "Filter coefficients copied to device." << std::endl;
    cudaHostUnregister(win_coeffs.data());

    auto s_end = std::chrono::high_resolution_clock::now();
    setup_time += std::chrono::duration<double>(s_end - s_start).count();

    // --- 2. EXECUTION PHASE: Moving data and running the kernel ---
    auto e_start = std::chrono::high_resolution_clock::now();
    // --- Setting up the kernel launch parameters ---
    int threadsPerBlock = std::min(n_chan, 1024); // Number of threads per block, do not exceed 1024 threads per block if n_chan is large.

    dim3 blockDim(threadsPerBlock); // 1D thread block dimension based on the number of threads per block

    int numGridBlocks_x = cuda::ceil_div(n_chan, threadsPerBlock); // Calculate the number of blocks needed in the x-dimension of the grid based on the number of channels and threads per block
    int numGridBlocks_y = n_time_blocks; // Number of blocks needed in the y-dimension of the grid is equal to the number of time blocks in the output data

    dim3 gridDim(numGridBlocks_y, numGridBlocks_x); // 2D grid dimension based on the number of blocks needed in the x and y dimensions
    // NOTE: dimensions inverted here so x-axis dimension of grid corresponds to time, because it can handle more values.
    
    // std::cout << "Launching 2D FIR Kernel: Grid(" << gridDim.x << ", " << gridDim.y 
    //           << "), Block(" << blockDim.x << ", 1)" << std::endl;

    // Launch the FIR convolution kernel
    FIR_convolution<T><<<gridDim, blockDim>>>(d_inputData, d_coeffs, d_outputDataComplex, n_taps, n_chan, n_time_blocks, filter_length);

    // ---> ADD THIS TO WAIT FOR FIR TO FINISH <---
    CUDA_CHECK(cudaDeviceSynchronize());
    auto e_end = std::chrono::high_resolution_clock::now();
    std::cout << "GPU_FIR_EXEC_TIME: " << std::chrono::duration<double>(e_end - e_start).count() << " seconds\n";
    exec_time += std::chrono::duration<double>(e_end - e_start).count();
}

template <typename T>
void PFB<T>::FFT() {
    // This function will perform the FFT stage of the PFB algorithm using the cuFFT library and the fft_plan created in the constructor.
    // The input data for the FFT will be the output from the FIR stage, which is stored in d_outputDataComplex. The output of the FFT will overwrite this same array to save memory, as it is no longer needed after the FIR stage.
    auto e_start = std::chrono::high_resolution_clock::now();
    if (fft_type == CUFFT_C2C) {
        // std::cout << "Executing single-precision FFT..." << std::endl;
        cufftExecC2C(fft_plan, (cufftComplex*)d_outputDataComplex, (cufftComplex*)d_outputDataComplex, CUFFT_FORWARD); // Execute the FFT in-place on the device data
    } else if (fft_type == CUFFT_Z2Z) {
        // std::cout << "Executing double-precision FFT..." << std::endl;
        cufftExecZ2Z(fft_plan, (cufftDoubleComplex*)d_outputDataComplex, (cufftDoubleComplex*)d_outputDataComplex, CUFFT_FORWARD); // Execute the FFT in-place on the device data
    } else {
        std::cerr << "Unsupported FFT type." << std::endl;
    }
    
    // ---> ADD THIS TO WAIT FOR FFT TO FINISH <---
    CUDA_CHECK(cudaDeviceSynchronize());

    auto e_end = std::chrono::high_resolution_clock::now();
    std::cout << "GPU_FFT_EXEC_TIME: " << std::chrono::duration<double>(e_end - e_start).count() << " seconds\n";
    exec_time += std::chrono::duration<double>(e_end - e_start).count();
}

template <typename T>
void PFB<T>::PSD() {
    // This function will perform the PSD stage of the PFB algorithm.
    auto e_start = std::chrono::high_resolution_clock::now();
    // --- Setting up the kernel launch parameters ---
    int threadsPerBlock = std::min(n_chan, 1024); // Number of threads per block, do not exceed 1024 threads per block if n_chan is large.

    dim3 blockDim(threadsPerBlock); // 1D thread block dimension based on the number of threads per block

    int numGridBlocks_x = cuda::ceil_div(n_chan, threadsPerBlock); // Calculate the number of blocks needed in the x-dimension of the grid based on the number of channels and threads per block
    int numGridBlocks_y = n_integrated_time_blocks; // Number of blocks needed in the y-dimension of the grid is equal to the number of time blocks in the output data

    dim3 gridDim(numGridBlocks_y, numGridBlocks_x); // 2D grid dimension based on the number of blocks needed in the x and y dimensions
    // NOTE: dimensions inverted here so x-axis dimension of grid corresponds to time, because it can handle more values.
    
    // std::cout << "Launching 2D PSD Kernel: Grid(" << gridDim.x << ", " << gridDim.y 
    //           << "), Block(" << blockDim.x << ", 1)" << std::endl;

    // Launch the PSD integration kernel
    PSD_integration<T><<<gridDim, blockDim>>>(d_outputDataComplex, d_outputDataReal, n_integrations, n_chan, n_time_blocks, n_integrated_time_blocks);

    // ---> ADD THIS TO WAIT FOR PSD TO FINISH <---
    CUDA_CHECK(cudaDeviceSynchronize());
    auto e_end = std::chrono::high_resolution_clock::now();
    exec_time += std::chrono::duration<double>(e_end - e_start).count();
    std::cout << "GPU_PSD_EXEC_TIME: " << std::chrono::duration<double>(e_end - e_start).count() << " seconds\n";
}

template <typename T>
void PFB<T>::execute_PFB(std::complex<T>* h_inputData) {
    // This function will perform the entire PFB algorithm by calling the FIR, FFT, and PSD functions in sequence. For now, we have only implemented the FIR and FFT stages, so this function will call those two stages in sequence.
    FIR(h_inputData); // Call the FIR stage with the input data from the host
    FFT(); // Call the FFT stage with the output data from the FIR stage, which is stored in d_outputDataComplex on the device
    PSD(); // Call the PSD stage with the output data from the FFT stage, which is stored in d_outputDataComplex on the device, and the final output will be stored in d_outputDataReal on the device
    // Output the final accumulated times
    std::cout << "GPU_SETUP_TIME: " << setup_time << "\n";
    std::cout << "GPU_EXEC_TIME: " << exec_time << "\n";
    std::cout << "==================================\n";
}

// This a function for copying the output file from device to host, to be verified in main.cu
template <typename T>
void PFB<T>::getOutput(T* h_outputData) {
    // Data retrieval is generally considered part of execution/processing time
    auto e_start = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaMemcpy(h_outputData, d_outputDataReal, output_length * sizeof(T), cudaMemcpyDeviceToHost));
    
    auto e_end = std::chrono::high_resolution_clock::now();
    exec_time += std::chrono::duration<double>(e_end - e_start).count();
    
    // std::cout << "GPU PSD output data copied back to host." << std::endl;
}