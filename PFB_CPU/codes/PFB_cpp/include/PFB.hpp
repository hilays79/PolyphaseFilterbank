#pragma once

#include <iostream>
#include <vector>
#include <complex>
#include <chrono>
#include <fftw3.h>
#include <algorithm>
#include <iomanip>   // Required for std::setprecision
#include <limits>    // Required for std::numeric_limits
#include "dsp.hpp"
#include "FFTW.hpp"

template <typename T>
std::vector<std::complex<T>> filtering(std::vector<std::complex<T>>& signal, int n_taps, int n_chan, int n_windows, double& setup_time, double& exec_time)
{
    auto s_start = std::chrono::high_resolution_clock::now();
    
    std::vector<T> win_coeffs = windowing::generate_win_coeffs<T>(n_taps, n_chan);
    // for (int i = 0; i<20; i++)
    // {
    //     std::cout << "Input real: " << signal[i].real() << std::endl;
    //     std::cout << "Input imag: " << signal[i].imag() << std::endl;
    // }

    // for (int i = 0; i<20; i++)
    // {
    //     std::cout << "Window: " << win_coeffs[i] << std::endl;
    // }
    int n_time_blocks = n_taps * n_windows - n_taps + 1; 
    std::vector<std::complex<T>> filtered_signal(n_time_blocks * n_chan); 

    auto s_end = std::chrono::high_resolution_clock::now();
    setup_time += std::chrono::duration<double>(s_end - s_start).count();

    auto e_start = std::chrono::high_resolution_clock::now();
    
    for (int n_t = 0; n_t < n_time_blocks; ++n_t) {
        int out_offset = misc::index_2d_to_1d(n_t, 0, n_chan);
        for (int m = 0; m < n_taps; ++m) {
            int w_offset = misc::index_2d_to_1d(m, 0, n_chan);
            int s_offset = misc::index_2d_to_1d(n_t + m, 0, n_chan);
            for (int n_c = 0; n_c < n_chan; ++n_c) {
                filtered_signal[out_offset + n_c] += signal[s_offset + n_c] * win_coeffs[w_offset + n_c];
            }
        }
    }
    
    // for (int i = 0; i<28; i++)
    // {
    //     std::cout << "Convolved Real: " << filtered_signal[i].real() << std::endl;
    //     std::cout << "Convolved Imag: " << filtered_signal[i].imag() << std::endl;
    // }

    auto e_end = std::chrono::high_resolution_clock::now();
    std::cout << "CPP_FIR_EXEC_TIME: " << std::chrono::duration<double>(e_end - e_start).count() << " seconds\n";
    exec_time += std::chrono::duration<double>(e_end - e_start).count();

    return filtered_signal;
}

template <typename T>
void FFT(std::vector<std::complex<T>>& filtered_signal, int n_taps, int n_chan, int n_windows, double& setup_time, double& exec_time)
{
    auto s_start = std::chrono::high_resolution_clock::now();
    int n_time_blocks = n_taps * n_windows - n_taps + 1; 

    auto* data_ptr = reinterpret_cast<typename FFTWWrapper<T>::complex_type*>(filtered_signal.data());
    
    int n[] = {n_chan};

    auto plan = FFTWWrapper<T>::plan_many_dft(1, n, n_time_blocks,
                                              data_ptr, NULL, 1, n_chan,
                                              data_ptr, NULL, 1, n_chan,
                                              FFTW_FORWARD, FFTW_ESTIMATE);
    
    auto s_end = std::chrono::high_resolution_clock::now();
    setup_time += std::chrono::duration<double>(s_end - s_start).count();

    auto e_start = std::chrono::high_resolution_clock::now();
    
    FFTWWrapper<T>::execute(plan);
    
    auto e_end = std::chrono::high_resolution_clock::now();
    std::cout << "CPP_FFT_EXEC_TIME: " << std::chrono::duration<double>(e_end - e_start).count() << " seconds\n";
    exec_time += std::chrono::duration<double>(e_end - e_start).count();

    // Force std::cout to print all available precision for a single-precision float (9 digits)
    std::cout << std::setprecision(std::numeric_limits<float>::max_digits10);
    size_t total_size = filtered_signal.size();

    std::cout << "Shape of filtered_signal: " << total_size << std::endl;

    std::cout << "\n--- First 256 Elements ---" << std::endl;
    int first_limit = std::min((int)total_size, 256);
    for (int i = 0; i < first_limit; i++)
    {
        std::cout << "[" << i << "] Real: " << filtered_signal[i].real() 
                << ", Imag: " << filtered_signal[i].imag() << std::endl;
    }

    std::cout << "\n--- Last 256 Elements ---" << std::endl;
    if (total_size > 0) 
    {
        int last_start = std::max(0, (int)total_size - 256);
        for (int i = last_start; i < total_size; i++)
        {
            std::cout << "[" << i << "] Real: " << filtered_signal[i].real() 
                    << ", Imag: " << filtered_signal[i].imag() << std::endl;
        }
    }

    FFTWWrapper<T>::destroy_plan(plan);
}

template <typename T>
std::vector<T> PSD(std::vector<std::complex<T>>& x_pfb, int n_taps, int n_chan, int n_windows, int n_integrations, double& setup_time, double& exec_time)
{
    auto s_start = std::chrono::high_resolution_clock::now();
    
    int n_time_blocks = n_taps * n_windows - n_taps + 1;
    int valid_time_blocks = (n_time_blocks / n_integrations) * n_integrations; 
    int n_integrated_blocks = valid_time_blocks / n_integrations; 
    std::vector<T> psd(n_integrated_blocks * n_chan);
    
    auto s_end = std::chrono::high_resolution_clock::now();
    setup_time += std::chrono::duration<double>(s_end - s_start).count();

    auto e_start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < valid_time_blocks; ++i) {
        int ind_integration_block = i / n_integrations; 
        for (int j = 0; j < n_chan; ++j) {
            int index_integration = misc::index_2d_to_1d(ind_integration_block, j, n_chan);
            int index_time_block = misc::index_2d_to_1d(i, j, n_chan);
            psd[index_integration] += std::norm(x_pfb[index_time_block]) / n_integrations; 
        }
    }
    
    auto e_end = std::chrono::high_resolution_clock::now();
    std::cout << "CPP_PSD_EXEC_TIME: " << std::chrono::duration<double>(e_end - e_start).count() << " seconds\n";
    exec_time += std::chrono::duration<double>(e_end - e_start).count();

    return psd;
}

// Returns the raw complex FFT output
template <typename T>
std::vector<std::complex<T>> PFB_filterbank_complex(std::vector<std::complex<T>>& signal, int n_taps, int n_chan, int n_windows)
{
    double setup_time = 0.0;
    double exec_time = 0.0;
    
    std::cout << "CPP PFB initialized with M=" << n_taps << ", N=" << n_chan << ", W=" << n_windows << std::endl;

    std::vector<std::complex<T>> filtered_signal = filtering<T>(signal, n_taps, n_chan, n_windows, setup_time, exec_time);
    FFT<T>(filtered_signal, n_taps, n_chan, n_windows, setup_time, exec_time);
    
    return filtered_signal;
}

// Returns the PSD output of the polyphase filterbank
template <typename T>
std::vector<T> PFB_filterbank(std::vector<std::complex<T>>& signal, int n_taps, int n_chan, int n_windows, int n_integrations=1)
{
    double setup_time = 0.0;
    double exec_time = 0.0;

    std::cout << "CPP PFB initialized with M=" << n_taps << ", N=" << n_chan << ", W=" << n_windows << std::endl;

    std::vector<std::complex<T>> filtered_signal = filtering<T>(signal, n_taps, n_chan, n_windows, setup_time, exec_time);
    FFT<T>(filtered_signal, n_taps, n_chan, n_windows, setup_time, exec_time);
    std::vector<T> psd = PSD<T>(filtered_signal, n_taps, n_chan, n_windows, n_integrations, setup_time, exec_time);

    std::cout << "CPP_SETUP_TIME:" << setup_time << "\n";
    std::cout << "CPP_EXEC_TIME:" << exec_time << "\n";
    std::cout << "==================================\n";

    return psd;
}