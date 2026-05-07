#include <iostream>
#include <vector>
#include <cmath>
#include <complex>
#include <random>

namespace ts
{
    std::vector<std::complex<double>> generate_noise(int n_samples, int seed){
        std::vector<std::complex<double>> noise(n_samples);
        std::mt19937 gen_real(seed);
        std::mt19937 gen_imaginary(seed+1);
        std::normal_distribution<double> dist(0.5, 0.1); // Mean 0.5, standard deviation 0.1
        for (int i = 0; i < n_samples; ++i)
        {
            noise[i] = std::complex<double>(dist(gen_real), dist(gen_imaginary)); // Generate complex noise with real and imaginary parts drawn from the normal distribution
        }
        return noise;
    }


    // Generate a sinusoidal signal with a given n_taps, n_chan, n_windows, frequency, and complex phasor or not.
    std::vector<std::complex<double>> generate_sinusoidal(int n_taps, int n_chan, int n_windows, double omega, bool include_noise, bool complex_phasor, int seed = 42)
    {
        int n_samples = n_taps * n_chan * n_windows;
        double signal_amplitude = 1.0; // Amplitude of the sinusoidal signal
        std::vector<std::complex<double>> signal(n_samples);
        for (int n=0; n < n_samples; n++)
        {
            if (complex_phasor){
                signal[n] = std::polar(signal_amplitude, omega * n); // Generate the complex sinusoidal signal using polar form
            } else 
            {
                signal[n] = std::complex<double>(signal_amplitude * std::sin(omega * n), 0); // Generate the real sinusoidal signal using sine function with zero imaginary part
            }
        }

        if (include_noise)
        {
            std::vector<std::complex<double>> noise = generate_noise(n_samples, seed);
            for (int i = 0; i < n_samples; ++i)
            {
                signal[i] += noise[i]; // Add the generated noise to the sinusoidal signal
            }
        }
        return signal;
    }
}