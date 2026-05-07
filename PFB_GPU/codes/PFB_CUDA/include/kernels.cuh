// Separate header file for CUDA kernels as they cannot be members of a class, only launching from a class is permitted.
#pragma once

// --- FIR convolution kernel ---
template <typename T>
__global__ void FIR_convolution(cuda::std::complex<T>* d_inputData, T* d_coeffs, cuda::std::complex<T>* d_outputData, int n_taps, int n_chan, int num_time_blocks) {

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
        int output_idx = i_t * n_chan + i_chan; // Calculate the index for the output data based on the time block and channel indices
        for (int i_tap = 0; i_tap < n_taps; ++i_tap){
            int input_idx = i_t * n_chan + i_chan + n_chan * i_tap; // Calculate the index for the input data based on the time block, channel, and tap indices
            int window_idx = i_chan + n_chan * i_tap; // Calculate the index for the filter coefficients based on the channel and tap indices
            // POSSIBLE OPTIMISATION: NEW VARIABLES FOR n_chan * i_tap and i_t * n_chan to reduce number of multiplications
            // POSSIBLE OPTIMISATION: SYMMETRIC COEFFICIENTS CAN BE EXPLOITED TO HALVE THE NUMBER OF MULTIPLICATIONS, BUT THIS IS NOT IMPLEMENTED YET.
            local_sum += d_inputData[input_idx] * d_coeffs[window_idx]; // Perform the convolution by multiplying the input data with filter coeffs and summing.
            // POSSIBLE OPTIMISATION: FMA instead of separate multiplication and addition can be used for better performance, but this is not implemented yet.
        }
        d_outputData[output_idx] = local_sum; // Store the final convolution result in the output data array at the calculated output index, DO NOT OVERWRITE THE INPUT DATA
    }
}

// --- PSD integration kernel ---
template <typename T>
__global__ void PSD_integration(cuda::std::complex<T>* d_inputData, T* d_outputData, int n_integrations, int n_chan, int num_time_blocks, int num_integrated_time_blocks) {
    // Channel index
    int i_chan = threadIdx.x + blockIdx.y * blockDim.x; // Calculate the channel index based on the thread and block indices
    // Integrated Time block index
    int i_t = blockIdx.x;
    int output_idx = i_t * n_chan + i_chan; // Calculate the index for the output data based on the time block and channel indices
    T local_sum = 0.0; // Initialize a local variable to accumulate the sum for the current integrated output element
    // Implementation for PSD integration kernel
    if (output_idx < num_integrated_time_blocks * n_chan) { // Check if the output index is within bounds for the integrated output data
    for (int i_integration = 0; i_integration < n_integrations; ++i_integration) {
        // Calculate the index for the output data based on the integration index and the output length       
        int input_idx = output_idx + i_integration * (num_integrated_time_blocks * n_chan); // Calculate the index for the input data based on the output index and integration index
        // POSSIBLE OPTIMISATION: Pool input_index and output_idx and the multiplication together.
        local_sum += cuda::std::abs(d_inputData[input_idx]) * cuda::std::abs(d_inputData[input_idx]); // Accumulate the power by summing the squared magnitudes of the complex input data elements
        // POSSIBLE OPTIMISATION: REAL**2 + IMAG**2 CAN BE USED INSTEAD OF CALLING ABS FUNCTION, BUT THIS IS NOT IMPLEMENTED YET.
        // POSSIBLE OPTIMISATION: NESTED FMA VS separate real and imag FMAs 
    }
    d_outputData[output_idx] = local_sum / n_integrations; // Store the final integrated power value in the output data array at the calculated output index    
    }
}