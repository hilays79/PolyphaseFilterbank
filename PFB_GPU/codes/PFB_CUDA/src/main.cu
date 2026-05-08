#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <string>
#include <chrono>

#include "test.cuh"
#include "PFB.cuh"
#include "PFB.hpp"
#include "dada_io.hpp"

// --- SEPARATE FUNCTION TO CALCULATE MAX DIFFERENCE ---
template <typename T>
T calculate_max_difference(const std::vector<T>& gpu_out, const std::vector<T>& cpu_out) {
    T max_diff = T(0); // Initialize to 0 of whatever type T is
    for (size_t i = 0; i < gpu_out.size(); ++i) {
        T diff = std::abs(gpu_out[i] - cpu_out[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    return max_diff;
}

int main(int argc, char* argv[]) {
    int M = 4, P = 256;
    int W = 100 * 1024; 
    int n_integrations = 1;
    
    // --- COMMAND LINE ARGUMENTS ---
    int in_NBIT = 64;
    int out_NBIT = 32;
    int read_from_file = 0; // 0 = memory, 1 = specific params file, 2 = custom file

    if (argc > 1) W = std::stoi(argv[1]);
    if (argc > 2) in_NBIT = std::stoi(argv[2]);
    if (argc > 3) out_NBIT = std::stoi(argv[3]);
    if (argc > 4) read_from_file = std::stoi(argv[4]);

    double freq = 1.0;
    int ndim_out = 1;
    bool include_noise = false;
    std::string signal_type = "complex_phasors"; 
    int delta_period = 257, delta_start = 0;

    std::string custom_in_path = "Random filepath"; // Added for custom file path input when read_from_file == 2
    std::string custom_out_path = "Random output filepath"; // Added for custom output file path when read_from_file == 2

    // Helper math variables
    int n_time_blocks = M * W - M + 1;
    int n_integrated_time_blocks = n_time_blocks / n_integrations;
    int output_length = n_integrated_time_blocks * P;
    bool CPU_verification = true; // Set to true to run CPU version for verification, can be turned off for larger runs where CPU would take too long

    // ========================================================================

    // SCENARIO 1: MEMORY GENERATION (read_from_file == 0)
    if (read_from_file == 0 && in_NBIT == 64 && out_NBIT == 64) {
        std::cout << "\n=== Gen Signal | 64-bit In | 64-bit Out ===\n";
        auto signal = ts::generate_sinusoidal(M, P, W, freq, include_noise, true);  // Generate in double precision

        std::vector<double> gpu_output(output_length);
        PFB<double> testPFB(M, P, W, n_integrations);
        testPFB.execute_PFB(signal.data());
        testPFB.getOutput(gpu_output.data());

        if (CPU_verification) {
            std::vector<double> cpu_output = PFB_filterbank<double>(signal, M, P, W, n_integrations);
            double max_diff = calculate_max_difference(gpu_output, cpu_output);
            std::cout << "Max Diff: " << max_diff << "\n";
        }
    }
    else if (read_from_file == 0 && in_NBIT == 64 && out_NBIT == 32) {
        std::cout << "\n=== Gen Signal | 64-bit In | 32-bit Out ===\n";
        auto raw_signal = ts::generate_sinusoidal(M, P, W, freq, include_noise, true);
        std::vector<std::complex<float>> signal(raw_signal.begin(), raw_signal.end()); // Downcast

        std::vector<float> gpu_output(output_length);
        PFB<float> testPFB(M, P, W, n_integrations);
        testPFB.execute_PFB(signal.data());
        testPFB.getOutput(gpu_output.data());

        if (CPU_verification) {
        std::vector<float> cpu_output = PFB_filterbank<float>(signal, M, P, W, n_integrations);

            float max_diff = calculate_max_difference(gpu_output, cpu_output);
            std::cout << "Max Diff: " << max_diff << "\n";
        }
    }
    else if (read_from_file == 0 && in_NBIT == 32 && out_NBIT == 32) {
        std::cout << "\n=== Gen Signal | 32-bit In | 32-bit Out ===\n";
        auto raw_signal = ts::generate_sinusoidal(M, P, W, freq, include_noise, true);
        std::vector<std::complex<float>> signal(raw_signal.begin(), raw_signal.end()); // Downcast

        std::vector<float> gpu_output(output_length);
        PFB<float> testPFB(M, P, W, n_integrations);
        testPFB.execute_PFB(signal.data());
        testPFB.getOutput(gpu_output.data());

        if (CPU_verification) {
            std::vector<float> cpu_output = PFB_filterbank<float>(signal, M, P, W, n_integrations);

            float max_diff = calculate_max_difference(gpu_output, cpu_output);
            std::cout << "Max Diff: " << max_diff << "\n";
        }
    }

    // SCENARIO 2: READ SPECIFIC FILE (read_from_file == 1)
    else if (read_from_file == 1 && in_NBIT == 64 && out_NBIT == 64) {
        std::cout << "\n=== Read File | 64-bit In | 64-bit Out ===\n";
        std::string filepath = dada::build_filepath(true, signal_type, in_NBIT, M, P, W, include_noise, freq, delta_period, delta_start);
        auto signal = dada::read_dada_for_pfb<std::complex<double>>(filepath).data; // This is not calling the pointer, .data is just to access the vector inside the struct in dada_io.hpp

        std::vector<double> gpu_output(output_length);
        PFB<double> testPFB(M, P, W, n_integrations);
        testPFB.execute_PFB(signal.data());
        testPFB.getOutput(gpu_output.data());
        std::string out_path = dada::build_filepath(false, signal_type, out_NBIT, M, P, W, include_noise, freq, delta_period, delta_start, true); // First false is for output, last true is to save it in CUDA output folder, default is c++
        dada::save_dada(gpu_output, P, ndim_out, out_NBIT, out_path);

        if (CPU_verification) {
            std::vector<double> cpu_output = PFB_filterbank<double>(signal, M, P, W, n_integrations);
            double max_diff = calculate_max_difference(gpu_output, cpu_output);
            std::cout << "Max Diff: " << max_diff << "\n";
            std::string out_path = dada::build_filepath(false, signal_type, out_NBIT, M, P, W, include_noise, freq, delta_period, delta_start); // First false is for output, absence of last true means it will be saved in the default C++ output folder
            dada::save_dada(cpu_output, P, ndim_out, out_NBIT, out_path);
        }
    }

    else if (read_from_file == 1 && in_NBIT == 64 && out_NBIT == 32) {
        std::cout << "\n=== Read File | 64-bit In | 32-bit Out ===\n";
        std::string filepath = dada::build_filepath(true, signal_type, in_NBIT, M, P, W, include_noise, freq, delta_period, delta_start);
        auto raw_signal = dada::read_dada_for_pfb<std::complex<double>>(filepath).data; 
        std::vector<std::complex<float>> signal(raw_signal.begin(), raw_signal.end());

        std::vector<float> gpu_output(output_length);
        PFB<float> testPFB(M, P, W, n_integrations);
        testPFB.execute_PFB(signal.data());
        testPFB.getOutput(gpu_output.data());

        std::string out_path = dada::build_filepath(false, signal_type, out_NBIT, M, P, W, include_noise, freq, delta_period, delta_start, true);
        dada::save_dada(gpu_output, P, ndim_out, out_NBIT, out_path);

        if (CPU_verification) {
            std::vector<float> cpu_output = PFB_filterbank<float>(signal, M, P, W, n_integrations);
            
            auto max_diff = calculate_max_difference(gpu_output, cpu_output);
            std::cout << "Max Diff: " << max_diff << "\n";
            
            std::string cpu_out_path = dada::build_filepath(false, signal_type, out_NBIT, M, P, W, include_noise, freq, delta_period, delta_start); 
            dada::save_dada(cpu_output, P, ndim_out, out_NBIT, cpu_out_path);
        }
    }
    else if (read_from_file == 1 && in_NBIT == 32 && out_NBIT == 32) {
        std::cout << "\n=== Read File | 32-bit In | 32-bit Out ===\n";
        std::string filepath = dada::build_filepath(true, signal_type, in_NBIT, M, P, W, include_noise, freq, delta_period, delta_start);
        auto signal = dada::read_dada_for_pfb<std::complex<float>>(filepath).data; 

        std::vector<float> gpu_output(output_length);
        PFB<float> testPFB(M, P, W, n_integrations);
        testPFB.execute_PFB(signal.data());
        testPFB.getOutput(gpu_output.data());

        std::string out_path = dada::build_filepath(false, signal_type, out_NBIT, M, P, W, include_noise, freq, delta_period, delta_start, true);
        dada::save_dada(gpu_output, P, ndim_out, out_NBIT, out_path);

        if (CPU_verification) {
            std::vector<float> cpu_output = PFB_filterbank<float>(signal, M, P, W, n_integrations);
            
            auto max_diff = calculate_max_difference(gpu_output, cpu_output);
            std::cout << "Max Diff: " << max_diff << "\n";
            
            std::string cpu_out_path = dada::build_filepath(false, signal_type, out_NBIT, M, P, W, include_noise, freq, delta_period, delta_start); 
            dada::save_dada(cpu_output, P, ndim_out, out_NBIT, cpu_out_path);
        }
    }

    // ========================================================================
    // SCENARIO 3: READ CUSTOM FILE PATH (read_from_file == 2)
    // ========================================================================
    else if (read_from_file == 2 && in_NBIT == 64 && out_NBIT == 64) {
        std::cout << "\n=== Read Custom | 64-bit In | 64-bit Out ===\n";
        auto signal = dada::read_dada_for_pfb<std::complex<double>>(custom_in_path).data; 

        std::vector<double> gpu_output(output_length);
        PFB<double> testPFB(M, P, W, n_integrations);
        testPFB.execute_PFB(signal.data());
        testPFB.getOutput(gpu_output.data());

        // Assuming custom_out_path was defined at the top of your main file
        dada::save_dada(gpu_output, P, ndim_out, out_NBIT, custom_out_path);

        if (CPU_verification) {
            std::vector<double> cpu_output = PFB_filterbank<double>(signal, M, P, W, n_integrations);
            
            auto max_diff = calculate_max_difference(gpu_output, cpu_output);
            std::cout << "Max Diff: " << max_diff << "\n";
            
            std::string cpu_out_path = custom_out_path + "_cpu";
            dada::save_dada(cpu_output, P, ndim_out, out_NBIT, cpu_out_path);
        }
    }
    else if (read_from_file == 2 && in_NBIT == 64 && out_NBIT == 32) {
        std::cout << "\n=== Read Custom | 64-bit In | 32-bit Out ===\n";
        auto raw_signal = dada::read_dada_for_pfb<std::complex<double>>(custom_in_path).data; 
        std::vector<std::complex<float>> signal(raw_signal.begin(), raw_signal.end());

        std::vector<float> gpu_output(output_length);
        PFB<float> testPFB(M, P, W, n_integrations);
        testPFB.execute_PFB(signal.data());
        testPFB.getOutput(gpu_output.data());

        dada::save_dada(gpu_output, P, ndim_out, out_NBIT, custom_out_path);

        if (CPU_verification) {
            std::vector<float> cpu_output = PFB_filterbank<float>(signal, M, P, W, n_integrations);
            
            auto max_diff = calculate_max_difference(gpu_output, cpu_output);
            std::cout << "Max Diff: " << max_diff << "\n";
            
            std::string cpu_out_path = custom_out_path + "_cpu";
            dada::save_dada(cpu_output, P, ndim_out, out_NBIT, cpu_out_path);
        }
    }
    else if (read_from_file == 2 && in_NBIT == 32 && out_NBIT == 32) {
        std::cout << "\n=== Read Custom | 32-bit In | 32-bit Out ===\n";
        auto signal = dada::read_dada_for_pfb<std::complex<float>>(custom_in_path).data; 

        std::vector<float> gpu_output(output_length);
        PFB<float> testPFB(M, P, W, n_integrations);
        testPFB.execute_PFB(signal.data());
        testPFB.getOutput(gpu_output.data());

        dada::save_dada(gpu_output, P, ndim_out, out_NBIT, custom_out_path);

        if (CPU_verification) {
            std::vector<float> cpu_output = PFB_filterbank<float>(signal, M, P, W, n_integrations);
            
            auto max_diff = calculate_max_difference(gpu_output, cpu_output);
            std::cout << "Max Diff: " << max_diff << "\n";
            
            std::string cpu_out_path = custom_out_path + "_cpu";
            dada::save_dada(cpu_output, P, ndim_out, out_NBIT, cpu_out_path);
        }
    }
    else {
        std::cerr << "Invalid combination of read_from_file, in_NBIT, and out_NBIT. Please check your inputs.\n";
        return 1;
    }
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