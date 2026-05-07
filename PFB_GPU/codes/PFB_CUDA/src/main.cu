// The main file for CUDA implementation of the PFB algorithm.
#include <iostream>
#include "test.cuh"
#include "PFB.cuh"
#include "PFB.hpp"

int main(int argc, char* argv[]) {
    int M = 4, P = 256;
    int W = 100;
    
    // --- COMMAND LINE ARGUMENTS ---
    int in_NBIT = 64;
    int out_NBIT = 32;
    bool read_from_file = true;  // If true then signal read from file, else complex phasor generated in-memory

    if (argc > 1) W = std::stoi(argv[1]);
    if (argc > 2) in_NBIT = std::stoi(argv[2]);
    if (argc > 3) out_NBIT = std::stoi(argv[3]);
    if (argc > 4) read_from_file = std::stoi(argv[4]) != 0; // Non-zero value means true

    double freq = 1.0;
    int ndim_out = 1;
    bool include_noise = false;
    std::string signal_type = "complex_phasors"; 
    int delta_period = 257, delta_start = 0;

    // --- END OF COMMAND LINE ARGUMENTS AND BASIC SETUP ---

    // --- SOME GPU SETUP INFO ---
    std::complex<double>* h_inputData = signal.data(); // Get the host pointer to the data of the generated signal vector
    int n_time_blocks = M*W - M + 1; // Calculate the number of time blocks in the output data based on the input signal length and the number of taps
    int n_integrated_time_blocks = n_time_blocks / n_integrations; // Calculate the number of time blocks in the integrated output data based on the total integrated output length and the number of channels
    int output_length = n_integrated_time_blocks * P; // Calculate the total length of the output
    std::vector<double> gpu_output(output_length); // Allocate host memory for the output data
    // Create an instance of the PFB class with double precision (64-bit)
    PFB<double> testPFB(M, P, W, n_integrations); // Create an instance of the PFB class with double precision (64-bit), n_integrations is set to 1.

    if (read_from_file) {
        std::cout << "Reading signal from file...\n";
            try {
        // SCENARIO 1: 64-bit Input -> 64-bit Output
        if (in_NBIT == 64 && out_NBIT == 64) {
            std::cout << "Reading 64-bit | Math 64-bit\n";
            auto my_pfb = [](std::vector<std::complex<double>>& d, int m, int p, int w) {
                auto result = PFB_filterbank<double>(d, m, p, w); 
                return result;
            };
            dada::run_pipeline<std::complex<double>, double>(my_pfb, signal_type, in_NBIT, out_NBIT, M, P, W, ndim_out, include_noise, freq, delta_period, delta_start);
        } 
        
        // SCENARIO 2: 64-bit Input -> 32-bit Output (Downcast)
        else if (in_NBIT == 64 && out_NBIT == 32) {
            std::cout << "Reading 64-bit | Math 32-bit (Downcasting)\n";
            auto my_pfb = [](std::vector<std::complex<double>>& d, int m, int p, int w) {
                std::vector<std::complex<float>> d_float(d.begin(), d.end()); // Safe Downcast
                auto start = std::chrono::high_resolution_clock::now();
                auto result = PFB_filterbank<float>(d_float, m, p, w); 
                auto end = std::chrono::high_resolution_clock::now();
                std::cout << "CPP_MATH_TIME:" << std::chrono::duration<double>(end - start).count() << "\n";
                return result;
            };
            dada::run_pipeline<std::complex<double>, float>(my_pfb, signal_type, in_NBIT, out_NBIT, M, P, W, ndim_out, include_noise, freq, delta_period, delta_start);
        }

        // SCENARIO 3: 32-bit Input -> 32-bit Output (Native 32-bit)
        else if (in_NBIT == 32 && out_NBIT == 32) {
            std::cout << "Reading 32-bit | Math 32-bit\n";
            auto my_pfb = [](std::vector<std::complex<float>>& d, int m, int p, int w) {
                auto start = std::chrono::high_resolution_clock::now();
                auto result = PFB_filterbank<float>(d, m, p, w); 
                auto end = std::chrono::high_resolution_clock::now();
                std::cout << "CPP_MATH_TIME:" << std::chrono::duration<double>(end - start).count() << "\n";
                return result;
            };
            dada::run_pipeline<std::complex<float>, float>(my_pfb, signal_type, in_NBIT, out_NBIT, M, P, W, ndim_out, include_noise, freq, delta_period, delta_start);
        }
        
        // Error trap
        else {
            std::cerr << "Fatal Error: Unsupported NBIT combination. Input: " << in_NBIT << ", Output: " << out_NBIT << "\n";
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "Fatal Error: " << e.what() << "\n";
    }
    } else {
        std::cout << "Generating in-memory complex phasor signal...\n";
        std::vector<std::complex<double>> signal = ts::generate_sinusoidal(M, P, W, freq, include_noise, true);
        auto start = std::chrono::high_resolution_clock::now();
        // if in_NBIT is 32, downcast the generated signal to float before processing
        if (in_NBIT == 32 && out_NBIT == 32) {
            std::vector<std::complex<float>> signal_float(signal.begin(), signal.end()); // Safe Downcast
            auto result = PFB_filterbank<float>(signal_float, M, P, W); 
        }
        else if (in_NBIT == 64 && out_NBIT == 64) {
            auto result = PFB_filterbank<double>(signal, M, P, W); 
        }
        else if (in_NBIT == 64 && out_NBIT == 32) {
            std::vector<std::complex<float>> signal_float(signal.begin(), signal.end()); // Safe Downcast
            auto result = PFB_filterbank<float>(signal_float, M, P, W); 
        }
        else {
            std::cerr << "Fatal Error: Unsupported NBIT combination for in-memory signal. Input: " << in_NBIT << ", Output: " << out_NBIT << "\n";
            return 1;
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "TOTAL_TIME:" << std::chrono::duration<double>(end - start).count() << "\n";
    }


    return 0;

int main(int argc, char* argv[])
{
    int M = 4, P = 256;
    int W = 100*1024; int n_integrations = 1;
    
    // --- COMMAND LINE ARGUMENTS ---
    int in_NBIT = 64;
    int out_NBIT = 32;
    bool read_from_file = false;  // If true then signal read from file, else complex phasor generated in-memory

    if (argc > 1) W = std::stoi(argv[1]);
    if (argc > 2) in_NBIT = std::stoi(argv[2]);
    if (argc > 3) out_NBIT = std::stoi(argv[3]);
    if (argc > 4) read_from_file = std::stoi(argv[4]) != 0; // Non-zero value means true

    double freq = 1.0;
    int ndim_out = 1;
    bool include_noise = false;
    std::string signal_type = "complex_phasors"; 
    int delta_period = 257, delta_start = 0;

    // --- END OF COMMAND LINE ARGUMENTS ---

    // --- INPUT SIGNAL GENERATION ---
    std::vector<std::complex<double>> signal;  // defined here to be used in both branches of the if-else statement and ensure it is in scope for the rest of the main function
    if (read_from_file) {
        // Code to read signal from file and populate the input data array on the device
        // This part is not implemented yet, but it will involve reading the data from a file, processing it as needed, and then copying it to the device memory.
        std::cout << "Reading input signal from file is not implemented yet. Please set read_from_file to false to generate a complex phasor signal in-memory." << std::endl;
        return 1; // Exit with an error code since reading from file is not implemented
    } else {
        std::cout << "Generating in-memory complex phasor signal...\n";
        signal = ts::generate_sinusoidal(M, P, W, freq, include_noise, true);
    }

    // --- END OF INPUT SIGNAL GENERATION ---
    std::complex<double>* h_inputData = signal.data(); // Get the host pointer to the data of the generated signal vector
    int n_time_blocks = M*W - M + 1; // Calculate the number of time blocks in the output data based on the input signal length and the number of taps
    int n_integrated_time_blocks = n_time_blocks / n_integrations; // Calculate the number of time blocks in the integrated output data based on the total integrated output length and the number of channels
    int output_length = n_integrated_time_blocks * P; // Calculate the total length of the output
    std::vector<double> gpu_output(output_length); // Allocate host memory for the output data
    // Create an instance of the PFB class with double precision (64-bit)
    PFB<double> testPFB(M, P, W, n_integrations); // Create an instance of the PFB class with double precision (64-bit), n_integrations is set to 1.
    // Call the FIR function to perform the FIR filtering stage of the PFB algorithm

    // One single call triggers the whole PFB chain!
    testPFB.execute_PFB(h_inputData);
    testPFB.getOutput(gpu_output.data());

    std::cout << "\n=== Executing CPU Reference PSD ===" << std::endl;
    double setup_time = 0.0;
    double exec_time = 0.0;
    
    // Call the CPU algorithm
    std::vector<double> cpu_output(output_length); // Allocate host memory for the CPU output data
    cpu_output = PFB_filterbank(signal, M, P, W, n_integrations);
    
    std::cout << "CPU Setup Time: " << setup_time << "s, Exec Time: " << exec_time << "s\n";

    // --- 3. COMPARE THE RESULTS ---
    std::cout << "\n=== Verifying Results ===" << std::endl;
    double max_diff = 0.0;

    for (int i = 0; i < output_length; ++i) {
        // std::abs on a complex number computes the Euclidean distance: sqrt(real^2 + imag^2)
        double diff = std::abs(gpu_output[i] - cpu_output[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }

    std::cout << "Maximum difference between GPU and CPU outputs: " << max_diff << std::endl;
}

// This main function was only used for testing the compilation and execution.
// int main()
// {
//     // must synchronize the kernel launch with the host code
//     for (int n = 1000000; n < 1000010; ++n) {
//         std::cout << "vectorSize=1024*" << n << ":" << std::endl;
//     int vectorSize = 1024*n;
//     test::unifiedMemCompare(vectorSize);
//     test::explicitMemCompare(vectorSize);
//     std::cout << "----------------------------------------" << std::endl;
//     }
//     return 0;
// }