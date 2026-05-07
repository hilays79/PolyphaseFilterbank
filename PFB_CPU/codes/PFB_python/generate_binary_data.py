#!/usr/bin/env python3

import os
import numpy as np
import test_signals as ts
from ipdb import set_trace as stop

# Dynamically find the repo root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

def create_binary_test_signals(n_taps, n_chan, n_windows, freq, delta_period, delta_start, in_NBIT, include_noise=False, signal_type="sinusoidals", save=True):
    savepath_base = os.path.join(REPO_ROOT, "Data", "input_files")
    
    freq_str = str(freq)
    if '.' not in freq_str:
        freq_str += '.0'
        
    # 1. Generate the signal
    if signal_type == "sinusoidals":
        binary_signal = ts.generate_sine_signal(n_taps, n_chan, n_windows, freq, include_noise=include_noise, complex_sine=False)
        filenamestart = f"{signal_type}_freq{freq_str}_M{n_taps}_P{n_chan}_W{n_windows}_noise{include_noise}"
        ndim = 2
    elif signal_type == "complex_phasors":
        binary_signal = ts.generate_sine_signal(n_taps, n_chan, n_windows, freq, include_noise=include_noise, complex_sine=True)
        filenamestart = f"{signal_type}_freq{freq_str}_M{n_taps}_P{n_chan}_W{n_windows}_noise{include_noise}"
        ndim = 2
    elif signal_type == "dirac_deltas":
        binary_signal = ts.generate_dirac_comb_signal(n_taps, n_chan, n_windows, delta_period, delta_start, include_noise=include_noise, real=True, is_complex=False)
        filenamestart = f"{signal_type}_d{delta_period}_s{delta_start}_noise{include_noise}"
        ndim = 2
    else:
        raise ValueError(f"Unsupported signal type: {signal_type}")
    
    if save==False:
        return binary_signal

    # Generate filename
    filename = f"{filenamestart}.dada"
    
    filepath = os.path.join(savepath_base, signal_type, f"{in_NBIT}-bit", filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # 2. Cast to correct bit depth
    if in_NBIT == 32:
        dtype = np.complex64 if ndim == 2 else np.float32
    elif in_NBIT == 64:
        dtype = np.complex128 if ndim == 2 else np.float64
    else:
        raise ValueError("in_NBIT must be 32 or 64.")
    
    binary_signal = np.asarray(binary_signal, dtype=dtype)

    # 3. ENFORCE MEMORY ORDERING: (Time, Channel=1, Polarisation=1)
    # The total number of time samples is the total length of the array
    n_time = binary_signal.size
    binary_signal = binary_signal.reshape(n_time, 1, 1) # Reshape to (Time, Channel=1, Polarisation=1)
    binary_signal = np.ascontiguousarray(binary_signal) # Ensures C-order memory layout

    # 4. Construct Header (Explicitly NCHAN=1 for input)
    header_keys = {
        "HDR_VERSION": "1.0",
        "HDR_SIZE": "4096",
        "NCHAN": "1",          # Fixed to 1 for input data
        "NPOL": "1",
        "NDIM": str(ndim),
        "NBIT": str(in_NBIT),
        "BW": "2",
        "RESOLUTION": "2048",
        "INSTRUMENT": "dspsr",
        "OBS_OFFSET": "0",
        "UTC_START": "2025-12-17-04:11:28",
        "FREQ": "100"
    }

    header_str = "".join([f"{k:<16} {v}\n" for k, v in header_keys.items()])
    header_bytes = header_str.encode('ascii').ljust(4096, b'\0')

    with open(filepath, "wb") as f:
        f.write(header_bytes)
        f.write(binary_signal.tobytes())

    # print(f"Input data written to: {filepath} | Shape: {binary_signal.shape}")
    return filepath

def read_dada_file(filepath):
    """
    Parses a PSRDADA file, returning the header as a dictionary 
    and the data as a correctly shaped NumPy array.
    """
    header_size = 4096
    header_dict = {}

    with open(filepath, "rb") as f:
        # 1. Read and Parse the 4096-byte Header
        header_bytes = f.read(header_size)
        
        # Decode and strip null padding bytes
        header_str = header_bytes.decode('ascii').strip('\0')
        
        for line in header_str.split('\n'):
            line = line.strip()
            if line:
                parts = line.split(maxsplit=1) # Split by first whitespace block
                if len(parts) == 2:
                    key, value = parts
                    header_dict[key] = value

    # 2. Extract dimension metadata (STRICT PARSING)
        required_keys = ["NCHAN", "NPOL", "NDIM", "NBIT"]
        
        # Check if all required keys exist before trying to read them
        for key in required_keys:
            if key not in header_dict:
                raise KeyError(f"Header parsing failed: Missing strictly required key '{key}'.")

        # Now extract them safely, knowing they exist
        try:
            nchan = int(header_dict["NCHAN"])
            npol  = int(header_dict["NPOL"])
            ndim  = int(header_dict["NDIM"])
            nbit  = int(header_dict["NBIT"])
        except ValueError as e:
            # This catches the case where the key exists, but the value isn't a number (e.g., NCHAN=abc)
            raise ValueError(f"Header parsing failed: Expected an integer for dimensions, but got a string. Details: {e}")

        # 3. Determine native NumPy datatype
        if nbit == 32:
            dtype = np.complex64 if ndim == 2 else np.float32
        elif nbit == 64:
            dtype = np.complex128 if ndim == 2 else np.float64
        else:
            raise ValueError(f"Unsupported NBIT: {nbit}")

        # 4. Read raw binary payload
        raw_data = f.read()
        data = np.frombuffer(raw_data, dtype=dtype)

        # 5. Reshape based on memory layout: (Time, Channel, Polarisation)
        # Time is the slowest varying axis, so we deduce it from the array length
        n_time = len(data) // (nchan * npol)
        data = data.reshape((n_time, nchan, npol))
    return header_dict, np.squeeze(data)

def save_pfb_to_dada(pfb_data, input_header_dict, signal_type, n_taps, n_windows, include_noise=False, freq=None, delta_period=None, delta_start=None):
    """
    Saves PFB output to a .dada file, inheriting NBIT and NDIM from the input.
    Mirrors the input directory and filename structure, but saves to the output directory.
    """
    savepath_base = os.path.join(REPO_ROOT, "Data", "output_files", "python")
    
    # 1. Inherit metadata from the input header
    nbit = int(input_header_dict["NBIT"])
    ndim = int(input_header_dict["NDIM"])
    
    # 2. Determine the new NCHAN and enforce (Time, Channel, Polarisation) layout
    if pfb_data.ndim == 1:
        pfb_data = pfb_data.reshape(-1, 1, 1)
    elif pfb_data.ndim == 2:
        n_time, n_chan = pfb_data.shape
        pfb_data = pfb_data.reshape(n_time, n_chan, 1)
    elif pfb_data.ndim == 3:
        n_time, n_chan, npol = pfb_data.shape
    else:
        raise ValueError(f"Expected 1D, 2D or 3D PFB array, but got shape: {pfb_data.shape}")

    # 3. Construct the filename using the exact input logic 
    if signal_type == "sinusoidals" or signal_type == "complex_phasors":
        if freq is None:
            raise ValueError(f"'freq' parameter is required for {signal_type}")
        
        freq_str = str(freq)
        if '.' not in freq_str:
            freq_str += '.0'
            
        filenamestart = f"{signal_type}_freq{freq_str}_M{n_taps}_P{n_chan}_W{n_windows}_noise{include_noise}"
    
    elif signal_type == "dirac_deltas":
        if delta_period is None or delta_start is None:
            raise ValueError("'delta_period' and 'delta_start' are required for dirac_deltas")
        filenamestart = f"{signal_type}_d{delta_period}_s{delta_start}_noise{include_noise}"
        
    else:
        raise ValueError(f"Unsupported signal type: {signal_type}")

    filename = f"{filenamestart}.dada"
    
    filepath = os.path.join(savepath_base, signal_type, f"{nbit}-bit", filename)
    
    # Ensure the target directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # 4. Cast to the exact same datatype as the input
    if nbit == 32:
        dtype = np.complex64 if ndim == 2 else np.float32
    elif nbit == 64:
        dtype = np.complex128 if ndim == 2 else np.float64
    else:
        raise ValueError(f"Unsupported NBIT: {nbit}")
        
    pfb_data = np.asarray(pfb_data, dtype=dtype)
    pfb_data = np.ascontiguousarray(pfb_data)

    # 5. Build the new header by copying the old one
    output_header = input_header_dict.copy()
    output_header["NCHAN"] = str(n_chan)
    output_header["NPOL"]  = "1" 

    # Format into the strictly padded 4096-byte block
    header_str = "".join([f"{k:<16} {v}\n" for k, v in output_header.items()])
    header_bytes = header_str.encode('ascii')
    
    if len(header_bytes) > 4096:
        raise ValueError("Header string exceeds the required 4096 bytes.")
        
    header_bytes = header_bytes.ljust(4096, b'\0')

    # 6. Write to disk
    with open(filepath, "wb") as f:
        f.write(header_bytes)
        f.write(pfb_data.tobytes())

    print(f"Saved PFB output: {filepath} | Shape: {pfb_data.shape} | NCHAN: {n_chan}")
    return filepath

if __name__ == "__main__":
    M, P, W = 4, 256, 100
    freq = 1
    delta_period = 257
    delta_start = 0
    in_NBIT = 64
    include_noise = False
    
    # signal_type = "complex_phasors" 
    # create_binary_test_signals(M, P, W, freq, delta_period, delta_start, in_NBIT, include_noise, signal_type)
    # signal_type = "dirac_deltas"
    # create_binary_test_signals(M, P, W, freq, delta_period, delta_start, in_NBIT, include_noise, signal_type)
    
    aa, bb = read_dada_file(os.path.join(REPO_ROOT, "Data", "input_files", "complex_phasors", "64-bit", "complex_phasors_freq1.0_M4_P256_W100_noiseFalse.dada"))
    stop()