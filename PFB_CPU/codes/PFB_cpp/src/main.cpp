#include <iostream>
#include <vector>
#include <complex>
#include <string>
#include <chrono>
#include "PFB.hpp"
#include "dada_io.hpp"
#include "dsp.hpp"

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

    if (read_from_file) {
        std::cout << "Reading signal from file...\n";
            try {
        // SCENARIO 1: 64-bit Input -> 64-bit Output
        if (in_NBIT == 64 && out_NBIT == 64) {
            std::cout << "Reading 64-bit | Math 64-bit\n";
            auto my_pfb = [](std::vector<std::complex<double>>& d, int m, int p, int w) {
                auto start = std::chrono::high_resolution_clock::now();
                auto result = PFB_filterbank<double>(d, m, p, w); 
                auto end = std::chrono::high_resolution_clock::now();
                std::cout << "CPP_MATH_TIME:" << std::chrono::duration<double>(end - start).count() << "\n";
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
}