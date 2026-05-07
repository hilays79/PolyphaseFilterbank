#!/usr/bin/env python3

import os
import numpy as np
import generate_binary_data as gbd
from ipdb import set_trace as stop
import matplotlib.pyplot as plt
import time
import subprocess
import PFB

# Dynamically find the repo root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

def get_output_filepath(language, signal_type, n_taps, n_chan, n_windows, include_noise, nbit, freq=None, delta_period=None, delta_start=None):
    """Helper to construct the expected output file path for either Python or C++."""
    savepath_base = os.path.join(REPO_ROOT, "Data", "output_files", language)
    
    freq_str = str(freq)
    if freq is not None and '.' not in freq_str:
        freq_str += '.0'
    
    if signal_type in ["sinusoidals", "complex_phasors"]:
        filenamestart = f"{signal_type}_freq{freq_str}_M{n_taps}_P{n_chan}_W{n_windows}_noise{include_noise}"
    elif signal_type == "dirac_deltas":
        filenamestart = f"{signal_type}_d{delta_period}_s{delta_start}_noise{include_noise}"
    else:
        raise ValueError(f"Unsupported signal type: {signal_type}")
        
    return os.path.join(savepath_base, signal_type, f"{nbit}-bit", f"{filenamestart}.dada")

def load_comparison_arrays(signal_type, n_taps, n_chan_out, n_windows, out_NBIT_python, out_NBIT_cpp, include_noise=False, freq=None, delta_period=None, delta_start=None):
    """
    Finds, parses, and returns both the Python and C++ output arrays for a specific test run.
    """
    
    # 1. Build the paths (passing the correct bit depth for each)
    py_path = get_output_filepath(
        "python", signal_type, n_taps, n_chan_out, n_windows, include_noise, out_NBIT_python, freq, delta_period, delta_start
    )
    cpp_path = get_output_filepath(
        "c++", signal_type, n_taps, n_chan_out, n_windows, include_noise, out_NBIT_cpp, freq, delta_period, delta_start
    )
    
    # 2. Check if they exist before parsing
    if not os.path.exists(py_path):
        raise FileNotFoundError(f"Python output missing. Did you run the Python pipeline?\nExpected: {py_path}")
    if not os.path.exists(cpp_path):
        raise FileNotFoundError(f"C++ output missing. Did you run the C++ pipeline?\nExpected: {cpp_path}")
        
    # 3. Read the data
    print(f"Loading Python data from: .../python/{signal_type}/{out_NBIT_python}-bit/{os.path.basename(py_path)}")
    py_header, py_data = gbd.read_dada_file(py_path)
    
    print(f"Loading C++ data from:    .../c++/{signal_type}/{out_NBIT_cpp}-bit/{os.path.basename(cpp_path)}")
    cpp_header, cpp_data = gbd.read_dada_file(cpp_path)
    
    return py_data, cpp_data

def calculate_comparison_metrics(py_array, cpp_array):
    """Calculate and print comparison metrics between the two arrays."""
    if py_array.shape != cpp_array.shape:
        raise ValueError(f"Shape mismatch: Python array shape {py_array.shape} vs C++ array shape {cpp_array.shape}")
    
    # Calculate element-wise differences
    if np.iscomplexobj(py_array) and np.all(py_array.imag == 0):
        py_array = py_array.real
    difference = py_array - cpp_array
    abs_difference = np.abs(difference)
    
    # Metrics
    max_diff = np.max(abs_difference)
    mean_diff = np.mean(abs_difference)
    mse = np.mean(difference**2)
    
    print(f"Max Absolute Difference: {max_diff}")
    print(f"Mean Absolute Difference: {mean_diff}")
    print(f"Mean Squared Error: {mse}")

def run_benchmark(in_NBIT_python, in_NBIT_cpp, out_NBIT_python, out_NBIT_cpp):
    windows = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200] 
    M, P, freq = 4, 256, 1.0
    signal_type = "complex_phasors" 
    delta_period, delta_start = 257, 0
    include_noise = False

    cpp_executable = os.path.join(REPO_ROOT, "codes", "PFB_cpp", "build", "pfb_app")
    cpp_build_dir = os.path.join(REPO_ROOT, "codes", "PFB_cpp", "build")

    print(f"{'W':<7} | {'Py Time (s)':<12} | {'C++ Tot (s)':<12} | {'C++ Set (s)':<12} | {'C++ Exe (s)':<12} | {'C++ Set/Exec':<12} | {'Speedup':<9} | {'Max Diff':<10}")
    print("-" * 105)

    for W in windows:
        
        freq_str = str(freq)
        if '.' not in freq_str:
            freq_str += '.0'
        filenamestart = f"{signal_type}_freq{freq_str}_M{M}_P{P}_W{W}_noise{include_noise}"
        
        input_filepath_py = os.path.join(REPO_ROOT, "Data", "input_files", signal_type, f"{in_NBIT_python}-bit", f"{filenamestart}.dada")
        
        if not os.path.exists(input_filepath_py):
            gbd.create_binary_test_signals(
                n_taps=M, n_chan=P, n_windows=W, freq=freq,
                delta_period=delta_period, delta_start=delta_start,
                in_NBIT=in_NBIT_python, include_noise=include_noise, signal_type=signal_type
            )
        input_filepath_cpp = os.path.join(REPO_ROOT, "Data", "input_files", signal_type, f"{in_NBIT_cpp}-bit", f"{filenamestart}.dada")    
        if not os.path.exists(input_filepath_cpp):
            gbd.create_binary_test_signals(
                n_taps=M, n_chan=P, n_windows=W, freq=freq,
                delta_period=delta_period, delta_start=delta_start,
                in_NBIT=in_NBIT_cpp, include_noise=include_noise, signal_type=signal_type
            )
        
        # Load Python input data
        header, input_data = gbd.read_dada_file(input_filepath_py)
        
        py_start = time.perf_counter()
        py_out = PFB.pfb_spectrometer(input_data, n_taps=M, n_chan=P)
        py_time = time.perf_counter() - py_start

        # Dynamically pass W, in_NBIT_cpp, and out_NBIT_cpp to the C++ executable
        result = subprocess.run(
            [cpp_executable, str(W), str(in_NBIT_cpp), str(out_NBIT_cpp)], 
            cwd=cpp_build_dir,
            capture_output=True, 
            text=True
        )
        
        cpp_total = None
        cpp_setup = None
        cpp_exec = None
        
        for line in result.stdout.split('\n'):
            if line.startswith("CPP_MATH_TIME:"):
                cpp_total = float(line.split(":")[1])
            elif line.startswith("CPP_SETUP_TIME:"):
                cpp_setup = float(line.split(":")[1])
            elif line.startswith("CPP_EXEC_TIME:"):
                cpp_exec = float(line.split(":")[1])
                
        if cpp_total is None:
            print(f"Error running C++ for W={W}.\n{result.stderr}\n{result.stdout}")
            continue

        cpp_output_filepath = get_output_filepath(
            "c++", signal_type, M, P, W, include_noise, out_NBIT_cpp, freq, delta_period, delta_start
        )
        
        if os.path.exists(cpp_output_filepath):
            _, cpp_out = gbd.read_dada_file(cpp_output_filepath)
            max_diff = np.max(np.abs(np.squeeze(py_out) - np.squeeze(cpp_out)))
            diff_str = f"{max_diff:.2e}" 
        else:
            diff_str = "File Missing"
            
        speedup = py_time / cpp_total
        set_exec_ratio = cpp_setup / cpp_exec
        speedup_str = f"{speedup:.2f}x"
        
        print(f"{W:<7} | {py_time:<12.6f} | {cpp_total:<12.6f} | {cpp_setup:<12.6f} | {cpp_exec:<12.6f} | {set_exec_ratio:<12.6f} | {speedup_str:<9} | {diff_str:<10}")
        
if __name__ == "__main__":
    M, P, W, freq = 4, 256, 100, 1
    delta_period, delta_start = 257, 0
    include_noise = False
    signal_type = "complex_phasors"
    
    in_NBIT_python = 64
    out_NBIT_python = 64 
    in_NBIT_cpp = 32
    out_NBIT_cpp = 32
    
    # py_array, cpp_array = load_comparison_arrays(
    #     signal_type=signal_type,
    #     n_taps=M,
    #     n_chan_out=P,
    #     n_windows=W,
    #     out_NBIT_python=out_NBIT_python,
    #     out_NBIT_cpp=out_NBIT_cpp,       
    #     include_noise=False,
    #     freq=freq,
    #     delta_period=delta_period,
    #     delta_start=delta_start
    # )
    
    # calculate_comparison_metrics(py_array, cpp_array)
    
    run_benchmark(in_NBIT_python, in_NBIT_cpp, out_NBIT_python, out_NBIT_cpp) 
    stop()
