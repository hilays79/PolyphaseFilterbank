#pragma once
#include <vector>
#include <fftw3.h>
#include <cmath>
#include <complex>

namespace windowing {
    
    template <typename T>
    void normalise_win(std::vector<T>& win) { 
        // Calculate Processing Gain: Sum of (c_i^2)
        T pg = 0.0;
        for (size_t i = 0; i < win.size(); ++i) {
            pg += win[i] * win[i];
        }

        T norm_factor = static_cast<T>(std::sqrt(pg));
        
        // Normalise the window coefficients
        for (size_t i = 0; i < win.size(); ++i) {
            win[i] /= norm_factor;
        }
    }

    template <typename T>
    std::vector<T> generate_hamming(int n_taps, int n_chan) {
        int n_samples = n_taps * n_chan;
        std::vector<T> win(n_samples);
        for (int i = 0; i < n_samples; ++i) {
            // Compute in double, cast final assignment to T
            win[i] = static_cast<T>(0.54 - 0.46 * std::cos(2.0 * M_PI * i / (n_samples - 1)));
        }
        return win;
    }

    template <typename T>
    std::vector<T> generate_sinc(int n_taps, int n_chan) {
        int n_samples = n_taps * n_chan;
        std::vector<T> sinc(n_samples);
        double fc = 1.0 / (2.0 * n_chan); // Normalized cutoff frequency (1/n_chan)
        double center = (n_samples - 1) / 2.0;

        for (int i = 0; i < n_samples; ++i) {
            double x = i - center;
            if (std::abs(x) < 1e-9) { 
                sinc[i] = static_cast<T>(2.0 * fc);
            } else {
                sinc[i] = static_cast<T>(std::sin(2.0 * M_PI * fc * x) / (M_PI * x));
            }
        }
        return sinc;
    }

    template <typename T>
    std::vector<T> generate_win_coeffs(int n_taps, int n_chan) {
        // Factory pattern: Generate both and combine
        std::vector<T> win = generate_hamming<T>(n_taps, n_chan);
        std::vector<T> sinc = generate_sinc<T>(n_taps, n_chan);

        // Element-wise multiplication (Equivalent to win *= sinc in Python)
        for (size_t i = 0; i < win.size(); ++i) {
            win[i] *= sinc[i];
        }
        normalise_win<T>(win); // Normalise the window coefficients
        return win;
    }

}

namespace misc
{
    // This function converts a power value to decibels (dB).
    double db(double x);

    inline int index_2d_to_1d(int i, int j, int n_cols) {
        return i * n_cols + j;
    }
}

namespace ts
{
    // Generate a sinusoidal signal with a given n_taps, n_chan, n_windows, frequency, and complex phasor or not.
    std::vector<std::complex<double>> generate_sinusoidal(int n_taps, int n_chan, int n_windows, double omega, bool include_noise, bool complex_phasor, int seed = 42);
}