#!/usr/bin/env python3

import numpy as np
import scipy
from ipdb import set_trace as stop
import matplotlib.pyplot as plt
import time
import test_signals as ts
import generate_binary_data as gbd
import os

def read_dada_file(file_path, count = None, offset=0):
    """
    Reads a binary file, skips a 4096-byte header, 
    and reads the rest as float32 complex numbers.
    """
    header_size = 4096
    
    with open(file_path, 'rb') as f:
        # Skip the 4096-byte header completely
        f.seek(header_size)
        
        # Read the remaining binary data into an array
        # np.complex64 uses two 32-bit floats (8 bytes total per number)
        data = np.fromfile(f, dtype=np.complex64, count=count, offset=offset)
        
    return data

def find_shift_min_residual(original, transformed, max_lag=None):
    """
    Finds the lag index that minimizes the Mean Squared Error (MSE) 
    between the original and transformed signals.
    """
    n_points = len(original)
    if max_lag is None:
        max_lag = n_points // 2

    lags = np.arange(-max_lag, max_lag + 1)
    residuals = np.zeros(len(lags))
    
    for i, lag in enumerate(lags):
        # Slice both arrays to their overlapping regions based on the current lag
        if lag > 0:
            orig_slice = original[lag:]
            trans_slice = transformed[:-lag]
        elif lag < 0:
            orig_slice = original[:lag]
            trans_slice = transformed[-lag:]
        else:
            orig_slice = original
            trans_slice = transformed
            
        # Calculate the Mean Squared Error on the overlapping portion
        residuals[i] = np.mean((orig_slice - trans_slice)**2)

    minima_ind, properties = scipy.signal.find_peaks(-residuals, distance=60)
    minima_lags = lags[minima_ind]
    plt.figure(figsize = (5, 5))
    plt.plot(lags, residuals)
    plt.scatter(minima_lags, residuals[minima_ind], c = 'k')
    plt.yscale("log")
    plt.xlabel("Lag index [original]")
    plt.ylabel("Residual RMS")
    plt.savefig("/home/hshah/PolyphaseFilterbank/PFB_CPU/codes/PFB_python/images/inversion_lag.png", dpi=300, bbox_inches='tight')

    best_lag = lags[np.argmin(residuals)]
    
    return best_lag, residuals, lags


def correlate_signals(path1, path2, count = 4194304):
    # read signal 1 from path1
    signal1 = read_dada_file(path1, count = count)
    # read signal 2 from path2
    signal2 = read_dada_file(path2, count = count)

    pol = 1
    if pol == 1:
        signal1 = signal1[0::2]
        signal2 = signal2[0::2]
    if pol == 2:
        signal1 = signal1[1::2]
        signal2 = signal2[1::2]

    signal1 = signal1/np.mean(signal1)
    signal2 = signal2/np.mean(signal2)

    signal1_fft = np.fft.fft(signal1)
    signal2_fft = np.fft.fft(signal2)

    conj_signal = signal1_fft * np.conj(signal2_fft)
    ifft_conj_signal = np.abs(np.fft.ifft(conj_signal))

    peak_overlap_idx = np.argmax(ifft_conj_signal)
    lag_peak_offset = peak_overlap_idx
    print("index offset:{}, Time offset [us]: {}".format(lag_peak_offset, lag_peak_offset*1/128))
    
    plt_peak_range = np.arange(peak_overlap_idx - 1000, peak_overlap_idx + 1000)
    plt.figure(figsize = (5, 5))
    plt.plot(plt_peak_range, np.abs(ifft_conj_signal[plt_peak_range]))
    # plt.xscale("log")
    plt.xlabel("Lag index [original]")
    plt.ylabel("Conjk")
    plt.savefig("/home/hshah/PolyphaseFilterbank/PFB_CPU/codes/PFB_python/images/conv_lag.png", dpi=300, bbox_inches='tight')

def average_correlate_signals(path1, path2, chunk_size=4194304, num_loops=10, auto=False, auto_delay=50):
    fft_len = chunk_size // 2
    accumulated_cross_spectrum = np.zeros(fft_len, dtype=np.complex64)
    
    bytes_per_sample = 8
    
    for loop in range(num_loops):
        print(f"Processing loop {loop + 1}/{num_loops}...")
        sample_offset = loop * chunk_size
        byte_offset = sample_offset * bytes_per_sample
        
        # Read the distinct chunk for this iteration
        signal1 = read_dada_file(path1, count=chunk_size, offset=byte_offset)
        if auto:
            signal2 = signal1.copy()
            signal2 = np.roll(signal2, auto_delay)  # Introduce a delay for auto-correlation
        else:
            signal2 = read_dada_file(path2, count=chunk_size, offset=byte_offset)
        
        # Break early if we hit the end of the file and get incomplete chunks
        if len(signal1) < chunk_size or len(signal2) < chunk_size:
            print(f"Reached end of file early at loop {loop}. Stopping accumulation.")
            num_loops = loop # Update count to actual loops completed
            break
            
        pol = 1
        if pol == 1:
            signal1 = signal1[0::2]
            signal2 = signal2[0::2]
        elif pol == 2:
            signal1 = signal1[1::2]
            signal2 = signal2[1::2]
            
        # Normalize each chunk to avoid floating-point explosion
        signal1 = signal1 / np.mean(np.abs(signal1))
        signal2 = signal2 / np.mean(np.abs(signal2))
        
        # Compute FFTs
        signal1_fft = np.fft.fft(signal1)
        signal2_fft = np.fft.fft(signal2)
        
        # Accumulate the complex cross-power spectrum
        accumulated_cross_spectrum += signal1_fft * np.conj(signal2_fft)
        
    if num_loops == 0:
        print("No data was read.")
        return

    # Average the accumulated cross-power spectrum
    avg_cross_spectrum = accumulated_cross_spectrum / num_loops
    
    # Transform back to time-domain lag space
    ifft_conj_signal = np.abs(np.fft.ifft(avg_cross_spectrum))
    
    peak_overlap_idx = np.argmax(ifft_conj_signal)
    
    lag_peak_offset = peak_overlap_idx
        
    print("Averaged loops: {}".format(num_loops))
    print("index offset (sliced index): {}, Time offset [us]: {}".format(lag_peak_offset, lag_peak_offset * 1 / 128))
    
    # Adjust plot range to safely stay within array bounds
    start_idx = max(0, peak_overlap_idx - 1000)
    end_idx = min(fft_len, peak_overlap_idx + 1000)
    plt_peak_range = np.arange(start_idx, end_idx)
    
    plt.figure(figsize=(5, 5))
    plt.plot(plt_peak_range, ifft_conj_signal[plt_peak_range])
    plt.xlabel("Lag index [sliced]")
    plt.ylabel("Averaged Correlation Magnitude")
    plt.savefig("/home/hshah/PolyphaseFilterbank/PFB_CPU/codes/PFB_python/images/conv_lag_avg.png", dpi=300, bbox_inches='tight')
    plt.close()
    

if __name__ == "__main__":
    # Example usage
    path1 = "/data/pfb/J1939_Parkes_UWL_1408_float32.dada"
    path2 = "/data/ajameson/pre_Detection.dump"
    # correlation_result = correlate_signals(path1, path2, count = 268435456)
    correlation_result = average_correlate_signals(path1, path2, chunk_size=33554432, num_loops=10, auto=True, auto_delay=-100)