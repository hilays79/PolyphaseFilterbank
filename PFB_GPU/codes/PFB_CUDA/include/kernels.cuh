// Separate header file for CUDA kernels as they cannot be members of a class, only launching from a class is permitted.
#pragma once

// FMA functions
// __device__ makes it callable only from a GPU. inline removes function call overhead.

// 1. Real-valued FMA for 32-bit floats
__device__ inline float custom_fma(float x, float y, float z) {
    return fmaf(x, y, z);
}

// 2. Real-valued FMA for 64-bit doubles
__device__ inline double custom_fma(double x, double y, double z) {
    return fma(x, y, z);
}

// 3. Wrapper for complex numbers
template <typename T>
__device__ inline cuda::std::complex<T> my_complex_fma(const cuda::std::complex<T>& signal, T scalar, const cuda::std::complex<T>& sum) {
    // Break the complex number into real and imaginary parts, and pass them
    // to the real-valued custom_fma. The compiler will automatically choose
    // the float or double version based on what T is!
    return cuda::std::complex<T>(
        custom_fma(signal.real(), scalar, sum.real()),
        custom_fma(signal.imag(), scalar, sum.imag())
    );
}

// 4. Wrapper for real numbers
template <typename T>
__device__ inline T my_real_fma(const T& signal, T scalar, const T& sum) {
    return T (custom_fma(signal, scalar, sum));
}

// --- FIR convolution kernel ---
template <typename T>
__global__ void FIR_convolution(const cuda::std::complex<T>* __restrict__ d_inputData, const T* __restrict__ d_coeffs, cuda::std::complex<T>* d_outputData, int n_taps, int n_chan, int num_time_blocks, int filter_length) {

    // Channel index
    int i_chan = threadIdx.x + blockIdx.y * blockDim.x; // Calculate the channel index based on the thread and block indices
    // Time block index
    int i_t = blockIdx.x;

    // Check if the channel index is within bounds
    if (i_chan < n_chan && i_t < num_time_blocks) {
        // since variable local_sum within a kernel, it is launched for each thread.
        // It is preferred to use a local variable for the accumulation of the convolution sum for each output element, as it avoids potential
        // access to the VRAM which contains the output data inside the loop.
        cuda::std::complex<T> local_sum(0.0, 0.0); // Initialize a local variable to accumulate the convolution sum for the current output element
        int t_chan = i_t * n_chan;
        int output_idx = t_chan + i_chan; // Calculate the index for the output data based on the time block and channel indices
        for (int i_tap = 0; i_tap < n_taps; ++i_tap){
            int chan_tap = n_chan * i_tap;
            int input_idx = output_idx + chan_tap; 
            
            // Standard 1D index
            int window_idx = i_chan + chan_tap; 
            
            // Fold the index over the center point using sym_window_idx.
            // Using min() avoids an 'if' statement, preventing warp divergence!
            int sym_window_idx = min(window_idx, filter_length - 1 - window_idx); 
            local_sum = my_complex_fma(d_inputData[input_idx], d_coeffs[sym_window_idx], local_sum); 
        }
        d_outputData[output_idx] = local_sum; 
    }
}

// --- FIR atomic convolution kernel: This implementation is inspired by the Curtin group's PFB implementation ---

template <typename T>
__global__ void FIR_atomic_convolution(const cuda::std::complex<T>* __restrict__ d_inputData, const T* __restrict__ d_coeffs, cuda::std::complex<T>* d_outputData, int n_taps, int n_chan, int num_time_blocks, int filter_length) {
    // Channel index
    int i_chan = threadIdx.x + blockIdx.y * blockDim.x; // Calculate the channel index based on the thread and block indices
    // Time block index
    int i_t = blockIdx.x;
    // Tap index
    int i_tap = blockIdx.z; // Each block in the z-dimension correponds to a different tap.
    int t_chan = i_t * n_chan;
    int output_idx = t_chan + i_chan; // Calculate the index for the output data based on the time block and channel indices, it is kept the same so that each thread is responsible for updating the same output element across all taps.
    int chan_tap = n_chan * i_tap;
    int input_idx = output_idx + chan_tap; // Calculate the index for the input
    int window_idx = i_chan + chan_tap; // Calculate the index for the filter coefficient
    // Check if the channel index is within bounds
    if (i_chan < n_chan && i_t < num_time_blocks && i_tap < n_taps) {
        // Calculate the symmetric window index for the filter coefficient
        int sym_window_idx = min(window_idx, filter_length - 1 - window_idx);
        // since atomicAdd works for real numbers, multiplication done separately for real and imaginary parts of the signal.
        T out_real = d_inputData[input_idx].real() * d_coeffs[sym_window_idx];
        T out_imag = d_inputData[input_idx].imag() * d_coeffs[sym_window_idx];
        T* out_ptr = reinterpret_cast<T*>(&d_outputData[output_idx]); // Get a pointer to the real and imaginary parts of the output data at the calculated output index
        atomicAdd(&out_ptr[0], out_real); // Atomically add the real part of the product to the real part of the output data at the calculated output index
        atomicAdd(&out_ptr[1], out_imag); // Atomically add the imaginary part of the product to the imaginary part of the output data at the calculated output index
    }
}


// --- PSD integration kernel ---
template <typename T>
__global__ void PSD_integration(cuda::std::complex<T>* d_inputData, T* d_outputData, int n_integrations, int n_chan, int num_time_blocks, int num_integrated_time_blocks) {
    // Channel index
    int i_chan = threadIdx.x + blockIdx.y * blockDim.x; // Calculate the channel index based on the thread and block indices
    // Integrated Time block index
    int i_t = blockIdx.x;
    int t_chan = i_t * n_chan;
    int input_jump = num_integrated_time_blocks * n_chan;
    int output_idx = t_chan + i_chan; // Calculate the index for the output data based on the time block and channel indices
    T local_sum = 0.0; // Initialize a local variable to accumulate the sum for the current integrated output element
    // Implementation for PSD integration kernel
    if (output_idx < num_integrated_time_blocks * n_chan) { // Check if the output index is within bounds for the integrated output data
    for (int i_integration = 0; i_integration < n_integrations; ++i_integration) {
        // Calculate the index for the output data based on the integration index and the output length       
        int input_idx = output_idx + i_integration * input_jump; // Calculate the index for the input data based on the output index and integration index

        // Extract real and imaginary components
        T real_part = d_inputData[input_idx].real();
        T imag_part = d_inputData[input_idx].imag();

        local_sum = my_real_fma(real_part, real_part, local_sum);
        local_sum = my_real_fma(imag_part, imag_part, local_sum);
    }
    d_outputData[output_idx] = local_sum / n_integrations; // Store the final integrated power value in the output data array at the calculated output index
    }
}