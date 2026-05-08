#!/usr/bin/env python3

import os
import numpy as np
import generate_binary_data as gbd
from ipdb import set_trace as stop
import time
import subprocess
import PFB
import sys
import io
import matplotlib.pyplot as plt

# Dynamically find the repo root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))

def get_output_filepath(language, signal_type, n_taps, n_chan, n_windows, include_noise, nbit, freq=None, delta_period=None, delta_start=None):
    """Helper to construct the expected output file path for Python, C++, or CUDA."""
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
    Finds, parses, and returns the Python, C++, and CUDA output arrays for a specific test run.
    """
    py_path = get_output_filepath("python", signal_type, n_taps, n_chan_out, n_windows, include_noise, out_NBIT_python, freq, delta_period, delta_start)
    cpp_path = get_output_filepath("c++", signal_type, n_taps, n_chan_out, n_windows, include_noise, out_NBIT_cpp, freq, delta_period, delta_start)
    cuda_path = get_output_filepath("CUDA", signal_type, n_taps, n_chan_out, n_windows, include_noise, out_NBIT_cpp, freq, delta_period, delta_start)
    
    if not os.path.exists(py_path):
        raise FileNotFoundError(f"Python output missing. Expected: {py_path}")
    if not os.path.exists(cpp_path):
        raise FileNotFoundError(f"C++ output missing. Expected: {cpp_path}")
    if not os.path.exists(cuda_path):
        raise FileNotFoundError(f"CUDA output missing. Expected: {cuda_path}")
        
    print(f"Loading Python data from: .../python/{signal_type}/{out_NBIT_python}-bit/{os.path.basename(py_path)}")
    py_header, py_data = gbd.read_dada_file(py_path)
    
    print(f"Loading C++ data from:    .../c++/{signal_type}/{out_NBIT_cpp}-bit/{os.path.basename(cpp_path)}")
    cpp_header, cpp_data = gbd.read_dada_file(cpp_path)

    print(f"Loading CUDA data from:   .../CUDA/{signal_type}/{out_NBIT_cpp}-bit/{os.path.basename(cuda_path)}")
    cuda_header, cuda_data = gbd.read_dada_file(cuda_path)
    
    return py_data, cpp_data, cuda_data

def calculate_comparison_metrics(py_array, cpp_array, cuda_array):
    """Calculate and print comparison metrics between the three arrays."""
    # Ensure shapes match
    py_sq = np.squeeze(py_array)
    cpp_sq = np.squeeze(cpp_array)
    cuda_sq = np.squeeze(cuda_array)

    if np.iscomplexobj(py_sq) and np.all(py_sq.imag == 0):
        py_sq = py_sq.real

    diff_P_C = np.max(np.abs(py_sq - cpp_sq))
    diff_P_G = np.max(np.abs(py_sq - cuda_sq))
    diff_C_G = np.max(np.abs(cpp_sq - cuda_sq))

    print(f"Max Diff (Python vs C++):  {diff_P_C}")
    print(f"Max Diff (Python vs CUDA): {diff_P_G}")
    print(f"Max Diff (C++ vs CUDA):    {diff_C_G}")

def run_benchmark(in_NBIT_python, in_NBIT_cpp, out_NBIT_python, out_NBIT_cpp, run_python=True):
    windows = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, 102400, 204800, 409600]
    M, P, freq = 4, 256, 1.0
    signal_type = "complex_phasors" 
    delta_period, delta_start = 257, 0
    include_noise = False

    cuda_executable = os.path.join(REPO_ROOT, "PFB_GPU", "codes", "PFB_CUDA", "build", "PFB_app")
    cuda_build_dir = os.path.join(REPO_ROOT, "PFB_GPU", "codes", "PFB_CUDA", "build")

    # Updated Header: Column name changed to Exe(C/G) to reflect execution time ratio
    print(f"{'Data (GB)':<9} | {'IN':<2}/{'OUT':<3} | {'Py Time':<9} | {'C++ Set':<9} | {'C++ Exe':<9} | {'GPU Set':<9} | {'GPU Exe':<9} | {'Spd(C/Py)':<9} | {'Spd(G/Py)':<9} | {'Exe(C/G)':<8} | {'Diff P-C':<8} | {'Diff P-G':<8} | {'Diff C-G':<8}")
    print("-" * 145)

    for W in windows:
        data_bits = M * P * W * in_NBIT_cpp
        data_gb = data_bits / (8 * (1024**3))
        data_gb_str = f"{data_gb:.4f}"
        
        freq_str = str(freq)
        if '.' not in freq_str:
            freq_str += '.0'
        filenamestart = f"{signal_type}_freq{freq_str}_M{M}_P{P}_W{W}_noise{include_noise}"
        
        input_filepath_py = os.path.join(REPO_ROOT, "Data", "input_files", signal_type, f"{in_NBIT_python}-bit", f"{filenamestart}.dada")
        
        # Ensure data exists (gbd calls omitted for brevity, keeping your logic)
        if not os.path.exists(input_filepath_py):
            gbd.create_binary_test_signals(n_taps=M, n_chan=P, n_windows=W, freq=freq, delta_period=delta_period, delta_start=delta_start, in_NBIT=in_NBIT_python, include_noise=include_noise, signal_type=signal_type)
        input_filepath_cpp = os.path.join(REPO_ROOT, "Data", "input_files", signal_type, f"{in_NBIT_cpp}-bit", f"{filenamestart}.dada")    
        if not os.path.exists(input_filepath_cpp):
            gbd.create_binary_test_signals(n_taps=M, n_chan=P, n_windows=W, freq=freq, delta_period=delta_period, delta_start=delta_start, in_NBIT=in_NBIT_cpp, include_noise=include_noise, signal_type=signal_type)
        
        py_time_str = "NaN"
        py_out = None
        py_time = 0
        
        if run_python:
            header, input_data = gbd.read_dada_file(input_filepath_py)
            py_start = time.perf_counter()
            py_out = PFB.pfb_spectrometer(input_data, n_taps=M, n_chan=P)
            py_time = time.perf_counter() - py_start
            py_time_str = f"{py_time:.5f}"

        result = subprocess.run([cuda_executable, str(W), str(in_NBIT_cpp), str(out_NBIT_cpp), "1"], cwd=cuda_build_dir, capture_output=True, text=True)
        
        cpp_setup, cpp_exec, gpu_setup, gpu_exec = None, None, None, None
        for line in result.stdout.split('\n'):
            line = line.strip()
            if line.startswith("CPP_SETUP_TIME:"): cpp_setup = float(line.split(":")[1].strip())
            elif line.startswith("CPP_EXEC_TIME:"): cpp_exec = float(line.split(":")[1].strip())
            elif line.startswith("GPU_SETUP_TIME:"): gpu_setup = float(line.split(":")[1].strip())
            elif line.startswith("GPU_EXEC_TIME:"): gpu_exec = float(line.split(":")[1].strip())
                
        if None in (cpp_setup, cpp_exec, gpu_setup, gpu_exec):
            print(f"Error running App for W={W}.\n{result.stderr}")
            continue

        cpp_total = cpp_setup + cpp_exec
        gpu_total = gpu_setup + gpu_exec

        cpp_output_filepath = get_output_filepath("c++", signal_type, M, P, W, include_noise, out_NBIT_cpp, freq, delta_period, delta_start)
        cuda_output_filepath = get_output_filepath("CUDA", signal_type, M, P, W, include_noise, out_NBIT_cpp, freq, delta_period, delta_start)
        
        diff_pc_str, diff_pg_str, diff_cg_str = "NaN", "NaN", "Missing"

        if os.path.exists(cpp_output_filepath) and os.path.exists(cuda_output_filepath):
            _, cpp_out = gbd.read_dada_file(cpp_output_filepath)
            _, cuda_out = gbd.read_dada_file(cuda_output_filepath)
            cpp_sq, cuda_sq = np.squeeze(cpp_out), np.squeeze(cuda_out)
            max_cg = np.max(np.abs(cpp_sq - cuda_sq))
            diff_cg_str = f"{max_cg:.1e}"

            if run_python and py_out is not None:
                py_sq = np.squeeze(py_out)
                diff_pc_str = f"{np.max(np.abs(py_sq - cpp_sq)):.1e}" 
                diff_pg_str = f"{np.max(np.abs(py_sq - cuda_sq)):.1e}"
            
        speedup_cpp = f"{py_time / cpp_total:.1f}x" if run_python else "NaN"
        speedup_gpu = f"{py_time / gpu_total:.1f}x" if run_python else "NaN"
        
        # Calculation changed to use execution times only
        exe_ratio_gpu_cpp = f"{cpp_exec / gpu_exec:.1f}x"
        
        print(f"{data_gb_str:<9} | {in_NBIT_cpp:<2}/{out_NBIT_cpp:<3} | {py_time_str:<9} | {cpp_setup:<9.5f} | {cpp_exec:<9.5f} | {gpu_setup:<9.5f} | {gpu_exec:<9.5f} | {speedup_cpp:<9} | {speedup_gpu:<9} | {exe_ratio_gpu_cpp:<8} | {diff_pc_str:<8} | {diff_pg_str:<8} | {diff_cg_str:<8}")

def benchmarking_plots(in_NBIT_python, in_NBIT_cpp, out_NBIT_python, out_NBIT_cpp, run_python=True):
    """Runs the benchmark, captures the terminal output, parses it, and generates log-log plots."""
    
    # 1. Capture the printed output from run_benchmark
    captured_output = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = captured_output

    try:
        run_benchmark(in_NBIT_python, in_NBIT_cpp, out_NBIT_python, out_NBIT_cpp, run_python=run_python)
    finally:
        sys.stdout = original_stdout # Restore normal printing

    # Extract the string and print it so you can still see the table in your terminal!
    output_str = captured_output.getvalue()
    print(output_str)

    # 2. Parse the table output
    data_gb = []
    py_time = []
    cpp_total = []
    gpu_total = []
    cpp_exe = []
    gpu_exe = []

    lines = output_str.strip().split('\n')
    
    start_idx = 0
    for i, line in enumerate(lines):
        if line.startswith('---'):
            start_idx = i + 1
            break
            
    for line in lines[start_idx:]:
        if not line.strip(): continue
        
        parts = [p.strip() for p in line.split('|')]
        if len(parts) < 7: continue 
        
        try:
            gb_val = float(parts[0])
            py_val = float(parts[2]) if parts[2] != 'NaN' else np.nan
            c_set = float(parts[3])
            c_exe = float(parts[4])
            g_set = float(parts[5])
            g_exe = float(parts[6])
            
            data_gb.append(gb_val)
            py_time.append(py_val)
            cpp_total.append(c_set + c_exe)
            gpu_total.append(g_set + g_exe)
            cpp_exe.append(c_exe)
            gpu_exe.append(g_exe)
        except ValueError:
            continue

    # 3. Create the plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # --- NEW: Calculate Reference Lines (Slope = 1 for linear O(N) scaling) ---
    if len(data_gb) > 0:
        ref_x = np.array([min(data_gb), max(data_gb)])
        # Slope of 1: y is directly proportional to x
        ref_y_total = cpp_total[0] * (ref_x / data_gb[0])
        ref_y_exe = cpp_exe[0] * (ref_x / data_gb[0])

    # Subplot 1: Total Times vs Data Size
    if run_python and not np.isnan(py_time).all():
        ax1.plot(data_gb, py_time, marker='o', label='Python Total Time')
    ax1.plot(data_gb, cpp_total, marker='s', label='C++ Total Time')
    ax1.plot(data_gb, gpu_total, marker='^', label='CUDA Total Time')
    
    if len(data_gb) > 0:
        ax1.plot(ref_x, ref_y_total, 'k--', alpha=0.6, label='O(N) Reference (Slope=1)')

    ax1.set_ylabel('Total Time (s)', fontsize=12)
    ax1.set_title('Total Processing Time vs Input Data Size', fontsize=13)
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    
    # Remove grid and make ticks bigger
    ax1.grid(False)
    ax1.tick_params(axis='both', which='major', labelsize=12, length=8, width=1.5)
    ax1.tick_params(axis='both', which='minor', length=5, width=1)
    ax1.legend()

    # Subplot 2: Execution Times vs Data Size
    ax2.plot(data_gb, cpp_exe, marker='s', color='tab:orange', label='C++ Execution Time')
    ax2.plot(data_gb, gpu_exe, marker='^', color='tab:green', label='CUDA Execution Time')
    
    if len(data_gb) > 0:
        ax2.plot(ref_x, ref_y_exe, 'k--', alpha=0.6, label='O(N) Reference (Slope=1)')

    ax2.set_ylabel('Execution Time (s)', fontsize=12)
    ax2.set_xlabel('Input Data Size (GB)', fontsize=12)
    ax2.set_title('Execution Time vs Input Data Size (excluding setup overhead)', fontsize=13)
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    
    # Remove grid and make ticks bigger
    ax2.grid(False)
    ax2.tick_params(axis='both', which='major', labelsize=12, length=8, width=1.5)
    ax2.tick_params(axis='both', which='minor', length=5, width=1)
    ax2.legend()

    # Suptitle Logic
    if run_python:
        sup_title = f"PFB Benchmark Scaling\nPython ({in_NBIT_python}-bit In / {out_NBIT_python}-bit Out) vs C++/CUDA ({in_NBIT_cpp}-bit In / {out_NBIT_cpp}-bit Out)"
    else:
        sup_title = f"PFB Benchmark Scaling\nC++/CUDA ({in_NBIT_cpp}-bit In / {out_NBIT_cpp}-bit Out) [Python Disabled]"
    
    fig.suptitle(sup_title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('images/benchmark_scaling_plots_{}_{}_{}_{}_optimised.png'.format(in_NBIT_python, in_NBIT_cpp, out_NBIT_python, out_NBIT_cpp), dpi=300)
    plt.show()

if __name__ == "__main__":
    M, P, W, freq = 4, 256, 100, 1
    delta_period, delta_start = 257, 0
    include_noise = False
    signal_type = "complex_phasors"
    
    in_NBIT_python = 64
    out_NBIT_python = 64 
    in_NBIT_cpp = 64
    out_NBIT_cpp = 64
    
    # Easily toggle Python execution on or off here
    run_python_baseline = True
    
    # Now call the plotting function instead of run_benchmark directly
    benchmarking_plots(in_NBIT_python, in_NBIT_cpp, out_NBIT_python, out_NBIT_cpp, run_python=run_python_baseline)