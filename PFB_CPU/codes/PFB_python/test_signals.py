#!/usr/bin/env python3

import numpy as np
from ipdb import set_trace as stop
import numpy as np

def generate_sine_signal(n_taps, n_chan, n_windows, freq, include_noise=True, complex_sine=False):
    M = n_taps     # Number of taps
    P = n_chan     # Number of 'branches', also fft length
    W = n_windows  # Number of windows of length M*P in input time stream
    
    freq_arr = np.atleast_1d(freq)[:, np.newaxis]
    num_freqs = freq_arr.shape[0]
    total_samples = M * P * W

    samples = np.arange(total_samples)
    
    # 1. Generate the CW signal
    amp = 1
    if complex_sine:
        cw_signal = amp * np.exp(1j * freq_arr * samples)
    else:
        cw_signal = amp * np.sin(freq_arr * samples)
    
    # 2. Generate the noise
    if include_noise: 
        # Generate real noise and cast it to complex immediately
        noise = np.random.normal(loc=0.5, scale=0.1, size=(num_freqs, total_samples)).astype(complex)
        if complex_sine:
            noise_imag = np.random.normal(loc=0.5, scale=0.1, size=(num_freqs, total_samples))
            # The += operator works perfectly here since noise is already complex
            noise += 1j * noise_imag
    else: 
        # Unconditionally set the zero array to complex
        noise = np.zeros((num_freqs, total_samples), dtype=complex)

    data = noise + cw_signal # Even real data will be cast to complex when added to complex noise
    
    return np.squeeze(data) # Remove any unnecessary dimensions for compatibility with downstream processing

def generate_dirac_comb_signal(n_taps, n_chan, n_windows, delta_period, delta_start, include_noise=True, real=True, is_complex=False):
    M = n_taps           # Number of taps
    P = n_chan           # Number of 'branches', also fft length
    W = n_windows        # Number of windows of length M*P in input time stream
    total_samples = M * P * W

    # 1. Determine the complex amplitude of the Dirac pulses
    amp = 0 + 0j
    if real:
        amp += 1
    if is_complex:
        amp += 1j

    # 2. Create the complex Dirac comb signal
    dirac_comb = np.zeros(total_samples, dtype=complex)
    dirac_comb[delta_start::delta_period] = amp

    # 3. Generate the noise conditionally based on the active components
    # Start with a strictly zero complex array
    noise = np.zeros(total_samples, dtype=complex)
    
    if include_noise: 
        if real:
            # Add noise only to the real component
            noise += np.random.normal(loc=0.5, scale=0.1, size=total_samples)
        if is_complex:
            # Add noise only to the imaginary component
            noise_imag = np.random.normal(loc=0.5, scale=0.1, size=total_samples)
            noise += 1j * noise_imag

    data = noise + dirac_comb
    
    return data

if __name__ == "__main__":
    stop()