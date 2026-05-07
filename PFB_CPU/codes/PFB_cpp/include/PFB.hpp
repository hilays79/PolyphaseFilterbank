#pragma once

#include <iostream>
#include <vector>
#include <complex>
#include <chrono>
#include <fftw3.h>

#include "dsp.hpp"
#include "FFTW.hpp"

template <typename T>
std::vector<std::complex<T>> filtering(std::vector<std::complex<T>>& signal, int n_taps, int n_chan, int n_windows, double& setup_time, double& exec_time)
{
    auto s_start = std::chrono::high_resolution_clock::now();
    
    std::vector<T> win_coeffs = windowing::generate_win_coeffs<T>(n_taps, n_chan); 
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
    
    auto e_end = std::chrono::high_resolution_clock::now();
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
    exec_time += std::chrono::duration<double>(e_end - e_start).count();

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
    exec_time += std::chrono::duration<double>(e_end - e_start).count();

    return psd;
}

template <typename T>
std::vector<T> PFB_filterbank(std::vector<std::complex<T>>& signal, int n_taps, int n_chan, int n_windows, int n_integrations=1)
{
    double setup_time = 0.0;
    double exec_time = 0.0;

    std::vector<std::complex<T>> filtered_signal = filtering<T>(signal, n_taps, n_chan, n_windows, setup_time, exec_time);
    FFT<T>(filtered_signal, n_taps, n_chan, n_windows, setup_time, exec_time); 
    std::vector<T> psd = PSD<T>(filtered_signal, n_taps, n_chan, n_windows, n_integrations, setup_time, exec_time);

    std::cout << "CPP_SETUP_TIME:" << setup_time << "\n";
    std::cout << "CPP_EXEC_TIME:" << exec_time << "\n";

    return psd;
}