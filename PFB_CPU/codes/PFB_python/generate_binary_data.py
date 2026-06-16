#!/usr/bin/env python3

import os
import numpy as np
import test_signals as ts
from ipdb import set_trace as stop
import PFB

# Dynamically find the repo root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))

def create_binary_test_signals(n_taps, n_chan, n_windows, freq, delta_period, delta_start, in_NBIT, include_noise=False, signal_type="sinusoidals", save=True, test=False, npol=1):
    savepath_base = os.path.join(REPO_ROOT, "Data", "input_files")
    
    freq_str = str(freq)
    if '.' not in freq_str:
        freq_str += '.0'
        
    # Generate the signal
    # Validate npol
    if not isinstance(npol, int) or npol < 1:
        raise ValueError(f"npol must be an integer greater than or equal to 1. Received: {npol}")
        
    signals_list = []
    current_freq = freq
    current_delta_period = delta_period
    
    # 2. Generate npol signals, doubling the relevant parameter each iteration
    for _ in range(npol):
        if signal_type == "sinusoidals":
            sig = ts.generate_sine_signal(n_taps, n_chan, n_windows, current_freq, include_noise=include_noise, complex_sine=False)
            current_freq *= 2
        elif signal_type == "complex_phasors":
            sig = ts.generate_sine_signal(n_taps, n_chan, n_windows, current_freq, include_noise=include_noise, complex_sine=True)
            current_freq *= 2
        elif signal_type == "dirac_deltas":
            sig = ts.generate_dirac_comb_signal(n_taps, n_chan, n_windows, current_delta_period, delta_start, include_noise=include_noise, real=True, is_complex=False)
            current_delta_period *= 2
        else:
            raise ValueError(f"Unsupported signal type: {signal_type}")
            
        # Flatten to ensure we can concatenate them cleanly into a 1D array
        signals_list.append(sig)
        
    # 3. Concatenate all generated signals into a single 1D array
    binary_signal = np.concatenate(signals_list)
    
    # 4. Update metadata variables
    # Only append the npol string to the filename if npol > 1 to preserve your original naming convention
    npol_suffix = f"_npol{npol}" if npol > 1 else ""
    
    if signal_type in ["sinusoidals", "complex_phasors"]:
        filenamestart = f"{signal_type}_freq{freq_str}_M{n_taps}_P{n_chan}_W{n_windows}_noise{include_noise}{npol_suffix}"
    elif signal_type == "dirac_deltas":
        filenamestart = f"{signal_type}_d{delta_period}_s{delta_start}_noise{include_noise}{npol_suffix}"
        
    ndim = 2  # Kept as 2 for complex numbers

    
    if save==False:
        return binary_signal

    # Generate filename
    if test:
        filename = f"{filenamestart}_{str(in_NBIT)}.dada"
        filepath = os.path.join("/home/hshah/src/test_data", filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
    else:
        filename = f"{filenamestart}.dada"
        filepath = os.path.join(savepath_base, signal_type, f"{in_NBIT}-bit", filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)


    # 2. Cast to correct bit depth
    if in_NBIT == 16:
        # Create an array of named tuples, each holding two float16 values
        final_signal = np.empty(binary_signal.shape, dtype=[('re', np.float16), ('im', np.float16)])
        # Cast the original normalized values directly
        final_signal['re'] = binary_signal.real.astype(np.float16)
        final_signal['im'] = binary_signal.imag.astype(np.float16)
    elif in_NBIT == 32:
        dtype = np.complex64 if ndim == 2 else np.float32
        final_signal = np.asarray(binary_signal, dtype=dtype)
    elif in_NBIT == 64:
        dtype = np.complex128 if ndim == 2 else np.float64
        final_signal = np.asarray(binary_signal, dtype=dtype)
    else:
        raise ValueError("in_NBIT must be 16 or 32 or 64.")
    binary_signal = final_signal

    # 3. ENFORCE MEMORY ORDERING: (Time, Channel=1, Polarisation=1)
    # The total number of time samples is the total length of the array
    # n_time = binary_signal.size
    # binary_signal = binary_signal.reshape(n_time, 1, 1) # Reshape to (Time, Channel=1, Polarisation=1)
    binary_signal = np.ascontiguousarray(binary_signal) # Ensures C-order memory layout
    if in_NBIT == 64:
        bytes_per_second = int(32000000)
        if npol == 1:
            utc_date = "2064-01-0{}-12:00:00".format(int(freq))
        elif npol == 2:
            utc_date = "2064-02-0{}-12:00:00".format(int(freq))
        else:
            raise ValueError("npol {} (1 or 2) will never be used.".format(npol))
    elif in_NBIT == 32:
        bytes_per_second = int(32000000/2)
        if npol == 1:
            utc_date = "2032-01-0{}-12:00:00".format(int(freq))
        elif npol == 2:
            utc_date = "2032-02-0{}-12:00:00".format(int(freq))
        else:
            raise ValueError("npol {} (1 or 2) will never be used.".format(npol))
    elif in_NBIT == 16:
        bytes_per_second = int(32000000/4)
        if npol == 1:
            utc_date = "2016-01-0{}-12:00:00".format(int(freq))
        elif npol == 2:
            utc_date = "2016-02-0{}-12:00:00".format(int(freq))
        else:
            raise ValueError("npol {} (1 or 2) will never be used.".format(npol))
    else:
        print("in_NBIT must be 16 or 32 or 64.")

    # 4. Construct Header (Explicitly NCHAN=1 for input)
    header_keys = {
        "HDR_VERSION": "1.0",
        "HDR_SIZE": "4096",
        "NCHAN": "1",          # Fixed to 1 for input data
        "NPOL": "{}".format(npol),
        "NDIM": str(ndim),     # Dynamic
        "NBIT": str(in_NBIT),  # Dynamic
        "BW": "2",
        "RESOLUTION": "2048",
        "INSTRUMENT": "dspsr",
        "OBS_OFFSET": "0",
        "UTC_START": "{}".format(utc_date),
        "FREQ": 100,     # Central frequency probably
        "NANT": "1",
        "NBEAM": "1",
        "NBIN": "1",
        "TSAMP": "0.5",
        "DSB": "0",
        "ORDER": "SFPT",
        "STATE": "Analytic",
        "OSAMP_NUMERATOR": "1",
        "OSAMP_DENOMINATE": "1",
        "OSAMP_DENOMINATOR": "1",
        "PICOSECONDS": "0",
        "BYTES_PER_SECOND": "{}".format(bytes_per_second*npol),
        "ENDIAN": "LITTLE",
        "ENCODING": "TWOSCOMPLEMENT",
        "REPRESENTATION": "FloatingPoint",
        "CAL_SIGNAL": "0"
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
        required_keys = ["NCHAN", "NPOL", "NDIM", "NBIT", "ORDER"]
        
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
            order = str(header_dict["ORDER"])
        except ValueError as e:
            # This catches the case where the key exists, but the value isn't a number (e.g., NCHAN=abc)
            raise ValueError(f"Header parsing failed: Expected an integer for dimensions, but got a string. Details: {e}")

        # 3. Determine native NumPy datatype
        if nbit == 16:
            # No native complex32. Use a structured dtype to group half2 fields.
            dtype = np.dtype([('real', np.float16), ('imag', np.float16)]) if ndim == 2 else np.float16
        elif nbit == 32:
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
        if order=="SFPT":
            if npol == 2:
                n_time = len(data) // (nchan * npol)
                data = data.reshape((npol, nchan, n_time))
        elif order=="TSPF":
            if npol==2:
                n_time = len(data) // (nchan * npol)
                n_chunks = npol * n_time
                # 1. Break the 1D array into a 2D array of chunks
                chunks = data.reshape(n_chunks, nchan)

                # 2. Slice out every 2nd chunk starting from index 0 (odds) and index 1 (evens)
                # Then flatten them back into 1D rows
                row_1 = chunks[0::2].flatten()
                row_2 = chunks[1::2].flatten()

                # 3. Stack them into a final 2D array
                data = np.vstack((row_1, row_2))
        else:
            raise ValueError("Read Ordering not implemented yet.")
    return header_dict, np.squeeze(data)

def save_pfb_to_dada(pfb_data, input_header_dict, signal_type, n_taps, n_windows, output_path=None, include_noise=False, freq=None, delta_period=None, delta_start=None):
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

    if output_path is None:
        filename = f"{filenamestart}.dada"
        
        filepath = os.path.join(savepath_base, signal_type, f"{nbit}-bit", filename)
        
        # Ensure the target directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
    else:
        filepath = output_path
        print("Saving to", filepath)

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
    freq = 2
    delta_period = 257
    delta_start = 0
    in_NBIT = 64
    include_noise = False
    
    # signal_type = "complex_phasors"
    # create_binary_test_signals(M, P, 51200, freq, delta_period, delta_start, 16, include_noise, signal_type, test=True, npol=1)
    # create_binary_test_signals(M, P, 51200, freq, delta_period, delta_start, 32, include_noise, signal_type, test=True, npol=1)
    # create_binary_test_signals(M, P, 51200, freq, delta_period, delta_start, 64, include_noise, signal_type, test=True, npol=1)

    # signal_type = "dirac_deltas"
    # create_binary_test_signals(M, P, W, freq, delta_period, delta_start, in_NBIT, include_noise, signal_type)
    # aa, bb = read_dada_file(os.path.join(REPO_ROOT, "Data", "input_files", "test_data", "complex_phasors_freq1.0_M4_P256_W102400_noiseFalse.dada"))
    # aa, bb = read_dada_file(os.path.join(REPO_ROOT, "Data", "input_files", "complex_phasors", "64-bit", "complex_phasors_freq1.0_M4_P256_W100_noiseFalse.dada"))
    # fourier_header, fourier_out = read_dada_file("/home/hshah/src/test_data/complex_phasors_freq1.0_M4_P256_W102400_noiseFalse_npol2_32.dada")
    # fourier_header, fourier_out = read_dada_file("/home/hshah/src/test_data/RAM/2032-02-01-12:00:00_0000000000000000.000000.dada")

    # header, input_data = read_dada_file(/home/hshah/src/test_data/complex_phasors_freq1.0_M4_P256_W102400_noiseFalse_32.dada)
    # py_out = PFB.pfb_spectrometer(input_data, n_taps=M, n_chan=P)
    # py_time = time.perf_counter() - py_start


    stop()