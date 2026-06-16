#!/usr/bin/env python3

# Simple implementation of an oversampled polyphase filterbank (PFB)
# OverPFB is based on Harris 2003 paper -> https://ieeexplore.ieee.org/document/1193158
import numpy as np
import scipy
from ipdb import set_trace as stop
import matplotlib.pyplot as plt
import time
import test_signals as ts
import generate_binary_data as gbd
import os
import PFB

# Dynamically find the repo root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

def db(x):
    """ Convert linear value to dB value """
    return 10*np.log10(x)

def generate_win_coeffs(M, P, osr, window_fn="hamming"):
    win_coeffs = scipy.signal.get_window(window_fn, M*P, fftbins=False)
    sinc       = scipy.signal.firwin(M * P, cutoff=osr/P, window="rectangular")
    win_coeffs *= sinc
    return win_coeffs

def pfb_fir_frontend(x, win_coeffs, M, P, osr):
    input_shift = int(P/osr) # This is how much we shift the window for each new output time sample. For critical sampling, this would be P. For oversampling, it's smaller.
    o_iter = (x.shape[0] - M*P) // input_shift + 1 # Number of output time samples we will produce
    print("Total steps: {}".format(o_iter))
    h_p = win_coeffs.reshape((M, P)).T
    # stop()
    x_summed = np.zeros((P, o_iter), dtype=complex)
    for t in range(o_iter):
        x_weighted = x[t*input_shift : t*input_shift+M*P].reshape((M, P)).T * h_p
        x_summed[:, t] = x_weighted.sum(axis=1)
    return x_summed.T

def fft(x_p, P, axis=1):
    return np.fft.fft(x_p, P, axis=axis)

def circular_buffer(x_c, M, P, osr):
    input_shift = int(P / osr)
    o_iter = x_c.shape[0]
    
    # Initialize the output array
    x_c_shifted = np.zeros_like(x_c)

    # Apply the shift row by row
    for t in range(o_iter):
        shift_amount = (t * input_shift) % P
        x_c_shifted[t, :] = np.roll(x_c[t, :], shift_amount)

    return x_c_shifted

def pfb_filterbank(x, win_coeffs, M, P, osr):
    x = x[:int(len(x)//(M*P))*M*P] # Ensure it's an integer multiple of win_coeffs
    x_fir = pfb_fir_frontend(x, win_coeffs, M, P, osr)
    x_fir_c = circular_buffer(x_fir, M, P, osr)
    x_pfb = fft(x_fir_c, P)
    return x_pfb

def pfb_spectrometer(x, n_taps, n_chan, osr, n_int=1, window_fn="hamming", PSD=True):
    M = n_taps
    P = n_chan
    
    # Generate window coefficients
    win_coeffs = generate_win_coeffs(M, P, osr, window_fn)
    pg = np.sum(np.abs(win_coeffs)**2)
    win_coeffs /= pg**.5 # Normalize for processing gain
    
    # Apply frontend, take FFT to get complex voltages
    x_pfb = pfb_filterbank(x, win_coeffs, M, P, osr)
    
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


def run_dada_PFB_comapre(python_pfb_function, signal_type, n_taps, n_chan_in, n_chan_out, n_windows, nbit, input_path=None, include_noise=False, freq=None, delta_period=None, delta_start=None):
    """
    End-to-end pipeline: Checks for input file, generates if missing, reads, runs PFB, and saves output.
    
    Parameters:
    - python_pfb_function: Your actual python function for the filterbank (e.g., PFB_filterbank)
    - n_chan_in: The number of channels for the INPUT signal (usually 1)
    - n_chan_out: The number of channels your PFB will output
    """
    
    # 1. Determine where the input file *should* be
    if input_path is None:
        input_filepath = get_expected_input_filepath(
            signal_type, n_taps, n_chan_out, n_windows, include_noise, freq, delta_period, delta_start
        )
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
    else:
        input_filepath = input_path
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
        output_path = "/home/hshah/src/test_data/over/test_output_python_freq2.0_M4_P256_W51200_noiseFalse_64.dada",
        include_noise=include_noise, 
        freq=freq, 
        delta_period=delta_period, 
        delta_start=delta_start
    )
    
    return output_filepath, pfb_output

if __name__ == "__main__":
    M, P, W, freq = 4, 256, 50, 1.0
    delta_period, delta_start = 257, 0
    nbit = 64
    include_noise = False
    signal_type = "dirac_deltas" # Can be "sinusoidals", "complex_phasors", or "dirac_deltas"
    osr = 32/27 # Oversampling ratio
    output_filepath, pfb_output = run_dada_PFB_comapre(
        python_pfb_function=lambda data, M, P: pfb_spectrometer(data, M, P, osr, n_int=1, window_fn="hamming", PSD=False), 
        signal_type=signal_type,
        n_taps=M,
        n_chan_in=1,
        n_chan_out=P,
        n_windows=W,
        nbit=nbit,
        input_path="/home/hshah/src/test_data/complex_phasors_freq2.0_M4_P256_W51200_noiseFalse_64.dada",
        include_noise=include_noise,
        freq=freq,
        delta_period=delta_period,
        delta_start=delta_start
    )
    stop()

    data = ts.generate_sine_signal(n_taps=M, n_chan=P, n_windows=W, freq=0.1, include_noise=False) # freq in radians/sample
    start_PFB = time.time()
    X_psd = pfb_spectrometer(data, n_taps=M, n_chan=P, osr=osr, n_int=1, window_fn="hamming")
    end_PFB = time.time()
    
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
