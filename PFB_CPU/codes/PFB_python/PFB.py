#!/usr/bin/env python3

# Simple implementation of a polyphase filterbank (PFB), FFT spectrometer, and brute-force spectrometer.
# PFB heavily based on Danny C. Price, Spectrometers and Polyphase Filterbanks in Radio Astronomy, 2016. arXiv 1607.03579.
import numpy as np
import scipy
from ipdb import set_trace as stop
import matplotlib.pyplot as plt
import time
import test_signals as ts
import generate_binary_data as gbd
import os

# Dynamically find the repo root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

def db(x):
    """ Convert linear value to dB value """
    return 10*np.log10(x)

def generate_win_coeffs(M, P, window_fn="hamming"):
    win_coeffs = scipy.signal.get_window(window_fn, M*P, fftbins=False)
    sinc       = scipy.signal.firwin(M * P, cutoff=1.0/P, window="rectangular")
    win_coeffs *= sinc
    return win_coeffs

def pfb_fir_frontend(x, win_coeffs, M, P):
    W = x.shape[0] // M // P
    x_p = x.reshape((W*M, P)).T
    h_p = win_coeffs.reshape((M, P)).T
    x_summed = np.zeros((P, M * W - M + 1), dtype=complex)
    for t in range(0, M*W-M + 1):
        x_weighted = x_p[:, t:t+M] * h_p
        x_summed[:, t] = x_weighted.sum(axis=1)
    return x_summed.T

def fft(x_p, P, axis=1):
    return np.fft.fft(x_p, P, axis=axis)

def pfb_filterbank(x, win_coeffs, M, P):
    x = x[:int(len(x)//(M*P))*M*P] # Ensure it's an integer multiple of win_coeffs
    x_fir = pfb_fir_frontend(x, win_coeffs, M, P)
    x_pfb = fft(x_fir, P)
    return x_pfb

def pfb_spectrometer(x, n_taps, n_chan, n_int=1, window_fn="hamming", PSD=True):
    M = n_taps
    P = n_chan
    
    # Generate window coefficients
    win_coeffs = generate_win_coeffs(M, P, window_fn)
    pg = np.sum(np.abs(win_coeffs)**2)
    win_coeffs /= pg**.5 # Normalize for processing gain
    
    # Apply frontend, take FFT to get complex voltages
    x_pfb = pfb_filterbank(x, win_coeffs, M, P)
    
    # If PSD is False, return the raw complex filterbank output immediately
    if not PSD:
        return x_pfb
        
    # Otherwise, calculate Power Spectral Density (i.e. square)
    x_psd = np.real(x_pfb * np.conj(x_pfb)) 
    
    # Trim array so we can do time integration
    valid_length = (x_psd.shape[0] // n_int) * n_int
    x_psd = x_psd[:valid_length]
    
    # Integrate over time, by reshaping and averaging over axis (efficient)
    x_psd = x_psd.reshape(valid_length // n_int, n_int, x_psd.shape[1])
    x_psd = x_psd.mean(axis=1)
    return x_psd

def brute_force_spectrometer(x, n_taps, n_chan, n_int=1, window_fn="hamming"):
    # Since PFB divides an FIR filter with M*P taps into P filters with M taps each, the total number of taps here is M*P
    M = n_taps
    P = n_chan
    num_taps = M * P
    N = len(x)
    
    # Generating a filter with a smooth bandpass response
    win_coeffs = scipy.signal.get_window(window_fn, num_taps)
    sinc = scipy.signal.firwin(num_taps, cutoff=1.0/P, window="rectangular") # Bandpass filter with channel width
    prototype_filter = win_coeffs * sinc 
    
    # Normalize for processing gain (same as the PFB spectrometer)
    pg = np.sum(np.abs(prototype_filter)**2)
    prototype_filter /= pg**0.5

    # Prepare output array (Time-integrated rows, Frequency channels)
    # We decimate by P because each channel is 1/P of the bandwidth
    # Calculate exactly how many decimated samples we get AFTER skipping the filter ramp-up
    total_decimated_samples = len(x[num_taps::P])
    num_integrated_blocks = total_decimated_samples // n_int
    x_psd = np.zeros((num_integrated_blocks, P))

    # 2. Loop through every channel (The "Inefficient" part)
    for k in range(P):
        # Shift the prototype filter to the center frequency of channel k
        # This creates a complex bandpass filter
        freq_shift = np.exp(2j * np.pi * k * np.arange(num_taps) / P)
        bandpass_filter = prototype_filter * freq_shift
        L = len(bandpass_filter)
        channel_data = np.zeros(N, dtype=complex)
        
        # Convolve the signal with this specific channel's filter
        channel_data = np.convolve(x, bandpass_filter, mode='full')[:N]
        
        # 3. Decimate and Integrate
        # In a PFB, we take 1 sample every P samples (the stride)
        decimated_data = channel_data[num_taps::P] 
        
        # Square to get power
        power = np.real(decimated_data * np.conj(decimated_data))
        
        # Time integration
        trimmed_power = power[:num_integrated_blocks * n_int]
        integrated_power = trimmed_power.reshape(-1, n_int).mean(axis=1)
        x_psd[:, k] = integrated_power

    return x_psd

def standard_fft_spectrometer(x, n_chan, n_int=1, window_fn="rectangular", PSD=True):
    P = n_chan
    
    # 1. Truncate data to ensure it perfectly divides into blocks of P
    num_samples = len(x)
    num_blocks = num_samples // P
    x_truncated = x[:num_blocks * P]
    
    # 2. Reshape into blocks (Time chunks, Frequency channels)
    # Instead of a sliding convolution, we just chop the data into discrete chunks
    x_blocks = x_truncated.reshape((num_blocks, P))
    
    # 3. Apply a simple window (NOT a sinc filter)
    # This just tapers the ends of each individual block of P samples to zero
    if window_fn != "rectangular":
        win_coeffs = scipy.signal.get_window(window_fn, P)
    if window_fn == "rectangular":
            # A rectangular window is an array of 1s
            win_coeffs = np.ones(P)

    pg = np.sum(np.abs(win_coeffs)**2)
    win_coeffs /= pg**0.5
    
    # Apply the normalized window to the data blocks
    x_blocks = x_blocks * win_coeffs
    
    # 4. Take the FFT across the channels (axis=1)
    x_fft = np.fft.fft(x_blocks, axis=1)

    if not PSD:
        return x_fft
    
    # 5. Calculate Power
    power = np.real(x_fft * np.conj(x_fft))
    
    # 6. Time Integration
    # Calculate how many full integrated blocks we can make
    num_integrated_blocks = num_blocks // n_int
    trimmed_power = power[:num_integrated_blocks * n_int]
    
    # Reshape and average over the integration time
    x_psd = trimmed_power.reshape((num_integrated_blocks, n_int, P)).mean(axis=1)
    
    return x_psd


def get_expected_input_filepath(signal_type, n_taps, n_chan, n_windows, include_noise, freq=None, delta_period=None, delta_start=None):
    """Helper to construct the expected input file path based on your naming rules."""
    savepath_base = os.path.join(REPO_ROOT, "Data", "input_files")
    
    if signal_type in ["sinusoidals", "complex_phasors"]:
        filenamestart = f"{signal_type}_freq{freq}_M{n_taps}_P{n_chan}_W{n_windows}_noise{include_noise}"
    elif signal_type == "dirac_deltas":
        filenamestart = f"{signal_type}_d{delta_period}_s{delta_start}_noise{include_noise}"
    else:
        raise ValueError(f"Unsupported signal type: {signal_type}")
        
    return os.path.join(savepath_base, signal_type, f"{filenamestart}.dada")


def run_dada_PFB(python_pfb_function, signal_type, n_taps, n_chan_in, n_chan_out, n_windows, nbit, include_noise=False, freq=None, delta_period=None, delta_start=None):
    """
    End-to-end pipeline: Checks for input file, generates if missing, reads, runs PFB, and saves output.
    
    Parameters:
    - python_pfb_function: Your actual python function for the filterbank (e.g., PFB_filterbank)
    - n_chan_in: The number of channels for the INPUT signal (usually 1)
    - n_chan_out: The number of channels your PFB will output
    """
    
    # 1. Determine where the input file *should* be
    input_filepath = get_expected_input_filepath(
        signal_type, n_taps, n_chan_out, n_windows, include_noise, freq, delta_period, delta_start
    )
    
    # 2. Generate the input file if it does not exist
    if not os.path.exists(input_filepath):
        print(f"Input file not found. Generating new signal at: {input_filepath}")
        gbd.create_binary_test_signals(
            n_taps=n_taps, 
            n_chan=n_chan_out, # weird but this is how it is currently structured. 
            n_windows=n_windows, 
            freq=freq, 
            delta_period=delta_period, 
            delta_start=delta_start, 
            nbit=nbit, 
            include_noise=include_noise, 
            signal_type=signal_type
        )
    else:
        print(f"Found existing input file: {input_filepath}")
        
    # 3. Read the data and header
    header, input_data = gbd.read_dada_file(input_filepath)
    print(f"Read input data shape: {input_data.shape}")
    
    # 4. Run the PFB
    # Note: Adjust the arguments here if your python PFB function expects different inputs
    print("Running PFB...")
    pfb_output = python_pfb_function(input_data, n_taps, n_chan_out)
    
    # 5. Save the output
    output_filepath = gbd.save_pfb_to_dada(
        pfb_data=pfb_output, 
        input_header_dict=header, 
        signal_type=signal_type, 
        n_taps=n_taps, 
        n_windows=n_windows, 
        include_noise=include_noise, 
        freq=freq, 
        delta_period=delta_period, 
        delta_start=delta_start
    )
    
    print("Pipeline complete!")
    return output_filepath, pfb_output

if __name__ == "__main__":
    M, P, W, freq = 4, 256, 100, 1.0
    delta_period, delta_start = 257, 0
    nbit = 64
    include_noise = False
    signal_type = "dirac_deltas" # Can be "sinusoidals", "complex_phasors", or "dirac_deltas"
    output_filepath, pfb_output = run_dada_PFB(
        python_pfb_function=lambda data, M, P: pfb_spectrometer(data, M, P), 
        signal_type=signal_type,
        n_taps=M,
        n_chan_in=1,
        n_chan_out=P,
        n_windows=W,
        nbit=nbit,
        include_noise=include_noise,
        freq=freq,
        delta_period=delta_period,
        delta_start=delta_start
    )
    stop()

    # data = ts.generate_sine_signal(n_taps=M, n_chan=P, n_windows=W, freq=0.1, include_noise=False) # freq in radians/sample
    # start_PFB = time.time()
    # X_psd = pfb_spectrometer(data, n_taps=M, n_chan=P, n_int=2, window_fn="hamming")
    # end_PFB = time.time()
    
    # start_brute = time.time()
    # X_psd_brute = brute_force_spectrometer(data, n_taps=M, n_chan=P, n_int=2, window_fn="hamming")
    # end_brute = time.time()

    # start_fft = time.time()
    # X_psd_fft = standard_fft_spectrometer(data, n_chan=P, n_int=2, window_fn="rectangular")
    # end_fft = time.time()

    # print(f"PFB time: {end_PFB - start_PFB}")
    # print(f"Brute force time: {end_brute - start_brute}")
    # print(f"FFT time: {end_fft - start_fft}")

    # # plt.imshow(db(X_psd)[0], cmap='viridis', aspect='auto')
    # plt.plot(db(X_psd)[0], c='#cc0000', label='PFB')
    # plt.plot(db(X_psd_brute)[0]-2, c='#0000cc', label='Brute Force (shifted down by 2 dB for visibility)')
    # plt.plot(db(X_psd_fft)[0]+2, c='#00cc00', label='FFT (shifted up by 2 dB for visibility)')
    # plt.title('Time taken for PFB=%.4f sec, Brute Force=%.4f sec, FFT=%.4f sec' % (end_PFB - start_PFB, end_brute - start_brute, end_fft - start_fft))
    # plt.ylim(-50, 30)
    # plt.xlim(-P/100, P/2)
    # plt.xlabel("Channel")
    # plt.ylabel("Power [dB]")
    # plt.legend()
    # plt.show()
    # stop()
    # # plt.colorbar()
    # # plt.xlabel("Channel")
    # # plt.ylabel("Time")    
