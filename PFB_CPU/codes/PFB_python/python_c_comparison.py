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
from matplotlib.colors import LogNorm

master_data_key = 'manualFFT_transpose_tap8_P512'
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

def load_comparison_arrays_path(path1, path2):
    """
    Finds, parses, and returns the Python, C++, and CUDA output arrays for a specific test run.
    """
        
    path1_header, path1_data = gbd.read_dada_file(path1)
    path2_header, path2_data = gbd.read_dada_file(path2)
    
    return path1_data, path2_data

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

def benchmarking_ntap_nchan_chunk(in_NBIT_cpp, out_NBIT_cpp):
    """Runs benchmarks varying n_taps, n_chan_out, and chunk_size while keeping data size constant."""
    M_values = [1, 2, 4, 8, 16, 32, 64]
    P_values = [32, 64, 128, 256, 512, 1024, 2048]
    # M_values = [1, 2]
    # P_values = [32, 64]
    chunk_sizes = [100/2**13, 100/2**12, 100/2**11, 100/2**10]
    
    # Constants required for input file generation
    freq = 1.0
    signal_type = "complex_phasors"
    delta_period, delta_start = 257, 0
    include_noise = False

    num_P = len(P_values)
    num_M = len(M_values)
    num_chunks = len(chunk_sizes)
    
    # Arrays to hold the three metrics: dimensions (P, M, chunks, Yours(0)/Curtin(1))
    setup_results = np.zeros((num_P, num_M, num_chunks, 2)) 
    exec_results = np.zeros((num_P, num_M, num_chunks, 2)) 
    diff_results = np.zeros((num_P, num_M, num_chunks, 2)) 

    for c_idx, c in enumerate(chunk_sizes):
        print(f"\n--- Benchmarking with chunk_size={c:.4f} GB ---")
        for m_idx, M in enumerate(M_values):
            for p_idx, P in enumerate(P_values):
                # Calculate W to keep data size constant
                W = int((c * (1024**3 * 8)) / (M * P * in_NBIT_cpp))
                print(f"Testing M={M}, P={P}, calculated W={W} for chunk size {c:.4f} GB")
                if W == 0:
                    setup_results[p_idx, m_idx, c_idx, :] = np.nan
                    exec_results[p_idx, m_idx, c_idx, :] = np.nan
                    diff_results[p_idx, m_idx, c_idx, :] = np.nan
                    continue

                # Ensure the correct input file is generated
                freq_str = str(freq)
                if '.' not in freq_str:
                    freq_str += '.0'
                filenamestart = f"{signal_type}_freq{freq_str}_M{M}_P{P}_W{W}_noise{include_noise}"
                input_filepath_cpp = os.path.join(REPO_ROOT, "Data", "input_files", signal_type, f"{in_NBIT_cpp}-bit", f"{filenamestart}.dada")    
                
                if not os.path.exists(input_filepath_cpp):
                    gbd.create_binary_test_signals(n_taps=M, n_chan=P, n_windows=W, freq=freq, delta_period=delta_period, delta_start=delta_start, in_NBIT=in_NBIT_cpp, include_noise=include_noise, signal_type=signal_type)

                # Run Yours (Non-Atomic)
                your_setup, your_exec, your_diff = run_cuda_benchmark(in_NBIT_cpp, out_NBIT_cpp, M, P, W, atomic=False, n_batches=1)
                
                # Run Curtin (Atomic)
                curtin_setup, curtin_exec, curtin_diff = run_cuda_benchmark(in_NBIT_cpp, out_NBIT_cpp, M, P, W, atomic=True, n_batches=1)

                # Record times: index 0 is Yours, index 1 is Curtin
                setup_results[p_idx, m_idx, c_idx, 0], setup_results[p_idx, m_idx, c_idx, 1] = your_setup, curtin_setup
                exec_results[p_idx, m_idx, c_idx, 0], exec_results[p_idx, m_idx, c_idx, 1] = your_exec, curtin_exec
                diff_results[p_idx, m_idx, c_idx, 0], diff_results[p_idx, m_idx, c_idx, 1] = your_diff, curtin_diff
                
    # --- Inner Helper Function to Plot the 2x4 Grids ---
    def plot_metric_grid(data_array, title, colorbar_label, save_label):
        fig, axs = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
        fig.suptitle(title, fontsize=18)

        # Calculate global min and max for consistent color scaling across all subplots
        valid_data = data_array[~np.isnan(data_array)]
        if len(valid_data) > 0:
            vmin = np.nanmin(valid_data)*0.8
            vmax = np.nanmax(valid_data)*1.2
            # LogNorm requires strictly positive values
            if vmin <= 0:
                positive_data = valid_data[valid_data > 0]
                vmin = np.nanmin(positive_data) if len(positive_data) > 0 else 1e-15
                vmax = max(vmax, vmin * 10) # Ensure vmax is strictly greater than vmin
        else:
            vmin, vmax = 1e-3, 1.0 # Fallback if all data is NaN

        for row in range(2):
            execution_type = "Non-Atomic" if row == 0 else "Atomic (Curtin)"
            
            for col in range(4):
                ax = axs[row, col]
                data = data_array[:, :, col, row]
                
                # Apply the globally calculated vmin and vmax
                im = ax.pcolormesh(data, cmap='viridis', shading='auto', norm=LogNorm(vmin=vmin, vmax=vmax))
                
                # Setup Axes Ticks
                ax.set_yticks(np.arange(num_P) + 0.5)
                ax.set_xticks(np.arange(num_M) + 0.5)
                ax.set_yticklabels(P_values)
                ax.set_xticklabels(M_values)
                
                # Labels and Titles
                if row == 1: 
                    ax.set_xlabel('Number of Taps (M)', fontsize=12)
                if col == 0: 
                    ax.set_ylabel('Output Channels (P)', fontsize=12)
                    
                ax.set_title(f"{execution_type}\nChunk Size: {chunk_sizes[col]:.4f} GB", fontsize=13)
                
                # Add a shared colorbar to the right side of the entire grid
                if row == 1 and col == 3:
                     fig.colorbar(im, ax=axs[:, :], orientation='vertical', label=colorbar_label, fraction=0.046, pad=0.04)

        plt.savefig("images/grid_benchmark_{}_{}_{}.png".format(save_label, in_NBIT_cpp, out_NBIT_cpp), dpi=300, bbox_inches='tight')

    # Create the three plots sequentially
    plot_metric_grid(setup_results, "GPU Setup Time Benchmark", "Setup Time (s)", "setup")
    plot_metric_grid(exec_results, "GPU Execution Time Benchmark", "Execution Time (s)", "execution")
    plot_metric_grid(diff_results, "Maximum Precision Difference (C++ vs GPU)", "Max Diff", "difference")
                
def run_cuda_benchmark(in_NBIT_cpp, out_NBIT_cpp, M, P, W, atomic, n_batches=1, verify_diff=True):
    """
    Runs the CUDA PFB executable and extracts performance metrics.
    """
    cuda_executable = os.path.join(REPO_ROOT, "PFB_GPU", "codes", "PFB_CUDA", "build", "PFB_app")
    cuda_build_dir = os.path.join(REPO_ROOT, "PFB_GPU", "codes", "PFB_CUDA", "build")

    atomic_flag = "1" if atomic else "0"

    cmd = [
        cuda_executable,
        str(W),
        str(in_NBIT_cpp),
        str(out_NBIT_cpp),
        "1",  
        str(M),
        str(P),
        str(n_batches),
        atomic_flag
    ]

    result = subprocess.run(cmd, cwd=cuda_build_dir, capture_output=True, text=True)

    gpu_setup = None
    gpu_exec = None
    gpu_fir = None
    gpu_fft = None
    max_diff = None

    for line in result.stdout.split('\n'):
        line = line.strip()
        if line.startswith("GPU_SETUP_TIME:"):
            gpu_setup = float(line.split(":")[1].strip())
        elif line.startswith("GPU_EXEC_TIME:"):
            gpu_exec = float(line.split(":")[1].strip())
        elif line.startswith("GPU_FIR_TIME:"):
            gpu_fir = float(line.split(":")[1].strip())
        elif line.startswith("GPU_FFT_TIME:"):
            gpu_fft = float(line.split(":")[1].strip())
        elif verify_diff and line.startswith("Max Diff:"): 
            max_diff = float(line.split(":")[1].strip())

    # Only check for max_diff if verify_diff is True
    if gpu_setup is None or gpu_exec is None or gpu_fir is None or gpu_fft is None or (verify_diff and max_diff is None):
        print(f"Error parsing CUDA output for W={W}, atomic={atomic}.")
        print("--- STDERR ---")
        print(result.stderr)
        print("--- STDOUT ---")
        print(result.stdout)
    
    return gpu_setup, gpu_exec, gpu_fir, gpu_fft, max_diff


def benchmarking_batch_chunk(in_NBIT_cpp, out_NBIT_cpp, M=4, P=256, verify_diff=True):
    """
    Runs benchmarks varying chunk_size and n_batches, holding M and P constant.
    Plots a 1x3 (or 1x2) grid of Heatmaps for GPU Setup, GPU Exec, and optionally Max Diff.
    """

    
    # Chunk sizes from 100/2**14 to 100/2**10
    chunk_sizes_base = [100 / (2**i) for i in range(10, 7, -1)]
    # Plotting x-axis requires 2x chunk size for complex data
    plot_x = [c * 2 for c in chunk_sizes_base]
    
    n_batches_vals = np.arange(1, 100)
    
    freq = 1.0
    signal_type = "complex_phasors"
    delta_period, delta_start = 257, 0
    include_noise = False
    atomic = False

    num_chunks = len(chunk_sizes_base)
    num_batches = len(n_batches_vals)

    # Initialize result arrays
    setup_results = np.zeros((num_batches, num_chunks))
    exec_results = np.zeros((num_batches, num_chunks))
    fir_results = np.zeros((num_batches, num_chunks))
    fft_results = np.zeros((num_batches, num_chunks))
    if verify_diff:
        diff_results = np.zeros((num_batches, num_chunks))

    print("\n=== Starting Batch vs Chunk Size Benchmark ===")
    
    for c_idx, c in enumerate(chunk_sizes_base):
        # Calculate W to match the target chunk size memory footprint
        W = int((c * (1024**3 * 8)) / (M * P * in_NBIT_cpp))
        N_out_blocks_total = (M * W) - M + 1
        
        print(f"\nTesting Base Chunk: {c:.6f} GB | Plot X: {plot_x[c_idx]:.6f} GB | W={W}")
        
        if W <= 0:
            setup_results[:, c_idx] = np.nan
            exec_results[:, c_idx] = np.nan
            fir_results[:, c_idx] = np.nan
            fft_results[:, c_idx] = np.nan
            if verify_diff:
                diff_results[:, c_idx] = np.nan
            continue

        # Ensure the correct input file is generated
        freq_str = str(freq) if '.' in str(freq) else f"{freq}.0"
        filenamestart = f"{signal_type}_freq{freq_str}_M{M}_P{P}_W{W}_noise{include_noise}"
        input_filepath_cpp = os.path.join(REPO_ROOT, "Data", "input_files", signal_type, f"{in_NBIT_cpp}-bit", f"{filenamestart}.dada")    
        
        if not os.path.exists(input_filepath_cpp):
            gbd.create_binary_test_signals(n_taps=M, n_chan=P, n_windows=W, freq=freq, delta_period=delta_period, delta_start=delta_start, in_NBIT=in_NBIT_cpp, include_noise=include_noise, signal_type=signal_type)

        for b_idx, b in enumerate(n_batches_vals):
            # The executable will fail if n_batches > N_out_blocks_total, skip these cases
            if b > N_out_blocks_total:
                setup_results[b_idx, c_idx] = np.nan
                exec_results[b_idx, c_idx] = np.nan
                fir_results[b_idx, c_idx] = np.nan
                fft_results[b_idx, c_idx] = np.nan
                if verify_diff:
                    diff_results[b_idx, c_idx] = np.nan
                continue

            g_setup, g_exec, g_fir, g_fft, max_diff = run_cuda_benchmark(in_NBIT_cpp, out_NBIT_cpp, M, P, W, atomic, n_batches=b, verify_diff=verify_diff)
            
            setup_results[b_idx, c_idx] = g_setup
            exec_results[b_idx, c_idx] = g_exec
            fir_results[b_idx, c_idx] = g_fir
            fft_results[b_idx, c_idx] = g_fft
            if verify_diff:
                diff_results[b_idx, c_idx] = max_diff
            
            # Print progress every 20 batches
            if b % 20 == 0:
                print(f"  Processed {b}/100 batches...")

    # --- Plotting ---
    num_plots = 5 if verify_diff else 4
    fig, axs = plt.subplots(1, num_plots, figsize=(7 * num_plots, 6), constrained_layout=True)
    fig.suptitle(f"PFB Scaling: Number of Batches vs. Complex Chunk Size (M={M}, P={P})", fontsize=16, fontweight='bold')

    metrics = [
        (setup_results, "GPU Setup Time (s)", "Setup Time (s)"),
        (exec_results, "GPU Exec Time (s)", "Execution Time (s)"),
        (fir_results, "GPU FIR Time (s)", "FIR Time (s)"),
        (fft_results, "GPU FFT Time (s)", "FFT Time (s)")
    ]
    
    if verify_diff:
        metrics.append((diff_results, "Max Difference (C++ vs GPU)", "Max Diff"))

    # If only 1 metric is ever passed, axs won't be iterable, but num_plots >= 2 guarantees it is an array.
    for i, (data_array, title, cb_label) in enumerate(metrics):
        ax = axs[i]
        
        # Calculate robust LogNorm limits ignoring NaNs and zeros
        valid_data = data_array[~np.isnan(data_array)]
        if len(valid_data) > 0:
            positive_data = valid_data[valid_data > 0]
            vmin = np.nanmin(positive_data) if len(positive_data) > 0 else 1e-10
            vmax = np.nanmax(valid_data)
            if vmax <= vmin: vmax = vmin * 10
        else:
            vmin, vmax = 1e-3, 1.0
            
        # Create X, Y grids for pcolormesh
        X, Y = np.meshgrid(plot_x, n_batches_vals)
        
        im = ax.pcolormesh(X, Y, data_array, cmap='viridis', shading='nearest', norm=LogNorm(vmin=vmin, vmax=vmax))
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Complex Data Chunk Size (GB)", fontsize=12)
        ax.set_ylabel("Number of Batches", fontsize=12)
        ax.set_xscale('log')
        
        fig.colorbar(im, ax=ax, label=cb_label)

    plt.savefig(f"images/benchmark_batch_chunk_{in_NBIT_cpp}_{out_NBIT_cpp}_verify{verify_diff}_{master_data_key}.png", dpi=300)
    np.savez(f"benchmark_data/batch_chunk_{in_NBIT_cpp}_{out_NBIT_cpp}_verify{verify_diff}_{master_data_key}.npz", n_batches=n_batches_vals, chunk_sizes=plot_x, setup=setup_results, exec=exec_results, fir=fir_results, fft=fft_results, diff=diff_results if verify_diff else None)
    print("\nPlot saved to images/benchmark_batch_chunk_{}_{}.png".format(in_NBIT_cpp, out_NBIT_cpp))
    plt.show()

def plot_batched_results(in_NBIT_cpp, out_NBIT_cpp, verify_diff=True):
    filepath = f"benchmark_data/batch_chunk_{in_NBIT_cpp}_{out_NBIT_cpp}_verify{verify_diff}_{master_data_key}.npz"
    if not os.path.exists(filepath):
        print(f"Error: Benchmark data file not found at {filepath}. Please run benchmarking_batch_chunk() first.")
        return
    data = np.load(filepath)
    n_batches = data['n_batches']
    chunk_sizes = data['chunk_sizes']
    setup_results = data['setup']
    exec_results = data['exec']
    fir_results = data['fir']
    fft_results = data['fft']
    diff_results = data['diff'] if verify_diff else None
    # Find minimum execution time across all batches for each chunk size
    min_exec_times = np.nanmin(exec_results, axis=0)
    min_exec_indices = np.nanargmin(exec_results, axis=0)
    best_batches = n_batches[min_exec_indices]
    fig = plt.figure(figsize=(6, 6))
    plt.plot(chunk_sizes, exec_results[0, :], marker='o', label='Unbatched Execution Time')
    plt.plot(chunk_sizes, min_exec_times, marker='s', label='Best Batched Execution Time')
    for i, c in enumerate(chunk_sizes):
        plt.annotate(f"{best_batches[i]} batches", (chunk_sizes[i], min_exec_times[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Complex Data Chunk Size (GB)", fontsize=12)
    plt.ylabel("Execution Time (s)", fontsize=12)
    plt.title(f"Best Batched Execution Time vs Unbatched\n(M={M}, P={P}, {in_NBIT_cpp}-bit In / {out_NBIT_cpp}-bit Out)", fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.savefig(f"images/best_batched_vs_unbatched_{in_NBIT_cpp}_{out_NBIT_cpp}_verify{verify_diff}_{master_data_key}.png", dpi=300)
    print(f"\nPlot saved to images/best_batched_vs_unbatched_{in_NBIT_cpp}_{out_NBIT_cpp}_verify{verify_diff}_{master_data_key}.png")
    plt.show()

if __name__ == "__main__":
    M, P, W, freq = 4, 256, 100, 1
    delta_period, delta_start = 257, 0
    include_noise = False
    signal_type = "complex_phasors"
    
    in_NBIT_python = 64
    out_NBIT_python = 64 
    in_NBIT_cpp = 32
    out_NBIT_cpp = 32
    
    # Easily toggle Python execution on or off here
    # run_python_baseline = True
    # run_benchmark(in_NBIT_python, in_NBIT_cpp, out_NBIT_python, out_NBIT_cpp, run_python=run_python_baseline)

    # -----------------------------------------------------------------------------
    # Critically sampled PFB
    # # Print some comparisons between different implementations
    # # Absolute maximum difference (32-bit RAM and CUDA)
    # path1 = "/home/hshah/src/test_data/RAM/2032-01-01-12:00:00_0000000000000000.000000.dada"
    # path2 = "/home/hshah/src/test_data/CUDA/2032-01-01-12:00:00_0000000000000000.000000.dada"

    # array1, array2 = load_comparison_arrays_path(path1, path2)
    # print("Absolute maximum difference npol=1 (32-bit RAM and CUDA): ", str(np.max(np.abs(array1-array2))))

    # path1 = "/home/hshah/src/test_data/RAM/2032-02-01-12:00:00_0000000000000000.000000.dada"
    # path2 = "/home/hshah/src/test_data/CUDA/2032-02-01-12:00:00_0000000000000000.000000.dada"

    # array1, array2 = load_comparison_arrays_path(path1, path2)
    # print("Absolute maximum difference npol=2 (32-bit RAM and CUDA): ", str(np.max(np.abs(array1-array2))))
    # stop()

    # # Absolute maximum difference (64-bit RAM and CUDA)
    # path1 = "/home/hshah/src/test_data/RAM/2064-01-01-12:00:00_0000000000000000.000000.dada"
    # path2 = "/home/hshah/src/test_data/CUDA/2064-01-01-12:00:00_0000000000000000.000000.dada"

    # array1, array2 = load_comparison_arrays_path(path1, path2)
    # print("Absolute maximum difference (64-bit RAM and CUDA): ", str(np.max(np.abs(array1-array2))))

    # # Absolute maximum difference (16-bit CUDA and 32-bit CUDA)
    # path1 = "/home/hshah/src/test_data/CUDA/2016-01-01-12:00:00_0000000000000000.000000.dada"
    # path2 = "/home/hshah/src/test_data/CUDA/2032-01-01-12:00:00_0000000000000000.000000.dada"

    # array1, array2 = load_comparison_arrays_path(path1, path2)
    # array1_complex = array1['real'].astype(np.float32) + 1j * array1['imag'].astype(np.float32)

    # print("Absolute maximum difference (16-bit CUDA and 32-bit CUDA): ", str(np.max(np.abs(array1_complex - array2))))

    # # Absolute maximum difference (16-bit CUDA and 64-bit CUDA)
    # path1 = "/home/hshah/src/test_data/CUDA/2016-01-01-12:00:00_0000000000000000.000000.dada"
    # path2 = "/home/hshah/src/test_data/CUDA/2064-01-01-12:00:00_0000000000000000.000000.dada"

    # array1, array2 = load_comparison_arrays_path(path1, path2)
    # array1_complex = array1['real'].astype(np.float32) + 1j * array1['imag'].astype(np.float32)

    # print("Absolute maximum difference (16-bit CUDA and 64-bit CUDA): ", str(np.max(np.abs(array1_complex - array2))))

    # # Absolute maximum difference (32-bit RAM pol2 and 2 32-bit RAM pol1)
    # p2_path = "/home/hshah/src/test_data/RAM/2032-02-01-12:00:00_0000000000000000.000000.dada"
    # p1_path1 = "/home/hshah/src/test_data/RAM/2032-01-01-12:00:00_0000000000000000.000000.dada"
    # p1_path2 = "/home/hshah/src/test_data/RAM/2032-01-02-12:00:00_0000000000000000.000000.dada"

    # p2_array, p1_array1 = load_comparison_arrays_path(p2_path, p1_path1)
    # p2_array, p1_array2 = load_comparison_arrays_path(p2_path, p1_path2)
    # stack_p1_array = np.vstack([p1_array1, p1_array2])

    # print("Absolute maximum difference (32-bit RAM pol2 and 2 32-bit RAM pol1): ", str(np.max(np.abs(p2_array - stack_p1_array))))

    # # Absolute maximum difference (64-bit RAM pol2 and 2 64-bit RAM pol1)
    # p2_path = "/home/hshah/src/test_data/RAM/2064-02-01-12:00:00_0000000000000000.000000.dada"
    # p1_path1 = "/home/hshah/src/test_data/RAM/2064-01-01-12:00:00_0000000000000000.000000.dada"
    # p1_path2 = "/home/hshah/src/test_data/RAM/2064-01-02-12:00:00_0000000000000000.000000.dada"

    # p2_array, p1_array1 = load_comparison_arrays_path(p2_path, p1_path1)
    # p2_array, p1_array2 = load_comparison_arrays_path(p2_path, p1_path2)
    # stack_p1_array = np.vstack([p1_array1, p1_array2])

    # print("Absolute maximum difference (64-bit RAM pol2 and 2 64-bit RAM pol1): ", str(np.max(np.abs(p2_array - stack_p1_array))))

    # # Absolute maximum difference (32-bit RAM and CUDA pol2)
    # path1 = "/home/hshah/src/test_data/RAM/2032-02-01-12:00:00_0000000000000000.000000.dada"
    # path2 = "/home/hshah/src/test_data/CUDA/2032-02-01-12:00:00_0000000000000000.000000.dada"

    # array1, array2 = load_comparison_arrays_path(path1, path2)
    # print("Absolute maximum difference (32-bit RAM and CUDA, npol=2): ", str(np.max(np.abs(array1-array2))))

    # # Absolute maximum difference (64-bit RAM and CUDA)
    # path1 = "/home/hshah/src/test_data/RAM/2064-02-01-12:00:00_0000000000000000.000000.dada"
    # path2 = "/home/hshah/src/test_data/CUDA/2064-02-01-12:00:00_0000000000000000.000000.dada"

    # array1, array2 = load_comparison_arrays_path(path1, path2)
    # print("Absolute maximum difference (64-bit RAM and CUDA, npol=2): ", str(np.max(np.abs(array1-array2))))

    # # Absolute maximum difference (16-bit CUDA and 64-bit CUDA, npol=2)
    # path1 = "/home/hshah/src/test_data/CUDA/2016-02-01-12:00:00_0000000000000000.000000.dada"
    # path2 = "/home/hshah/src/test_data/CUDA/2064-02-01-12:00:00_0000000000000000.000000.dada"

    # array1, array2 = load_comparison_arrays_path(path1, path2)
    # array1_complex = array1['real'].astype(np.float32) + 1j * array1['imag'].astype(np.float32)

    # print("Absolute maximum difference (16-bit CUDA and 64-bit CUDA, npol=2): ", str(np.max(np.abs(array1_complex - array2))))

    # -----------------------------------------------------------------------------
    # over sampled PFB

    # Absolute maximum difference (64-bit RAM and CUDA)
    path1 = "/home/hshah/src/test_data/over/RAM/2064-01-01-12:00:00_0000000000000000.000000.dada"
    path2 = "/home/hshah/src/test_data/over/test_output_python_freq1.0_M4_P256_W51200_noiseFalse_64.dada"

    array1, array2 = load_comparison_arrays_path(path1, path2)

    print("Absolute maximum difference (64-bit RAM and Python):", str(np.max(np.abs(array1 - array2[:array1.shape[0]]))))

    # Absolute maximum difference (64-bit RAM pol2 and 2 64-bit RAM pol1)
    p2_path = "/home/hshah/src/test_data/over/RAM/2064-02-01-12:00:00_0000000000000000.000000.dada"
    p1_path1 = "/home/hshah/src/test_data/over/test_output_python_freq1.0_M4_P256_W51200_noiseFalse_64.dada"
    p1_path2 = "/home/hshah/src/test_data/over/test_output_python_freq2.0_M4_P256_W51200_noiseFalse_64.dada"

    p2_array, p1_array1 = load_comparison_arrays_path(p2_path, p1_path1)
    p2_array, p1_array2 = load_comparison_arrays_path(p2_path, p1_path2)
    stack_p1_array = np.vstack([p1_array1, p1_array2])

    print("Absolute maximum difference (64-bit RAM and Python, npol=2 (pol1)):", str(np.max(np.abs(p2_array[0] - p1_array1[:p2_array[0].shape[0]]))))
    print("Absolute maximum difference (64-bit RAM and Python, npol=2 (pol2)):", str(np.max(np.abs(p2_array[1] - p1_array2[:p2_array[1].shape[0]]))))

    # Absolute maximum difference (64-bit RAM and CUDA)
    path1 = "/home/hshah/src/test_data/over/RAM/2064-02-01-12:00:00_0000000000000000.000000.dada"
    path2 = "/home/hshah/src/test_data/over/CUDA/2064-02-01-12:00:00_0000000000000000.000000.dada"

    array1, array2 = load_comparison_arrays_path(path1, path2)
    print("Absolute maximum difference (64-bit RAM and CUDA, npol=2): ", str(np.max(np.abs(array1-array2))))

    # Absolute maximum difference (64-bit RAM and CUDA)
    path1 = "/home/hshah/src/test_data/over/RAM/2064-01-01-12:00:00_0000000000000000.000000.dada"
    path2 = "/home/hshah/src/test_data/over/CUDA/2064-01-01-12:00:00_0000000000000000.000000.dada"

    array1, array2 = load_comparison_arrays_path(path1, path2)
    print("Absolute maximum difference (64-bit RAM and CUDA, npol=1): ", str(np.max(np.abs(array1-array2))))

    path1 = "/home/hshah/src/test_data/over/RAM/2032-02-01-12:00:00_0000000000000000.000000.dada"
    path2 = "/home/hshah/src/test_data/over/RAM/2064-02-01-12:00:00_0000000000000000.000000.dada"

    array1, array2 = load_comparison_arrays_path(path1, path2)

    print("Absolute maximum difference (32-bit RAM and 64-bit RAM (npol=2)):", str(np.max(np.abs(array1 - array2))))

    path1 = "/home/hshah/src/test_data/over/RAM/2032-02-01-12:00:00_0000000000000000.000000.dada"
    path2 = "/home/hshah/src/test_data/over/CUDA/2032-02-01-12:00:00_0000000000000000.000000.dada"

    array1, array2 = load_comparison_arrays_path(path1, path2)

    print("Absolute maximum difference (32-bit RAM and 32-bit CUDA (npol=2)):", str(np.max(np.abs(array1 - array2))))

    # ## IF THE OUTPUT FROM OSAMP HAS BEEN GENERATED WITH 1/1 RATIO

    # path1 = "/home/hshah/src/test_data/over/RAM/2064-02-01-12:00:00_0000000000000000.000000.dada"
    # path2 = "/home/hshah/src/test_data/over/CUDA/2064-02-01-12:00:00_0000000000000000.000000.dada"

    # array1, array2 = load_comparison_arrays_path(path1, path2)

    # print("Absolute maximum difference (over 64-bit RAM and over 64-bit CUDA (npol=2)):", str(np.max(np.abs(array1 - array2))))

    # path1 = "/home/hshah/src/test_data/over/RAM/2032-02-01-12:00:00_0000000000000000.000000.dada"
    # path2 = "/home/hshah/src/test_data/over/CUDA/2032-02-01-12:00:00_0000000000000000.000000.dada"

    # array1, array2 = load_comparison_arrays_path(path1, path2)

    # print("Absolute maximum difference (over 32-bit RAM and over 32-bit CUDA (npol=2)):", str(np.max(np.abs(array1 - array2))))

    # path1 = "/home/hshah/src/test_data/over/RAM/2064-02-01-12:00:00_0000000000000000.000000.dada"
    # path2 = "/home/hshah/src/test_data/CUDA/2064-02-01-12:00:00_0000000000000000.000000.dada"

    # array1, array2 = load_comparison_arrays_path(path1, path2)
    # stop()

    # print("Absolute maximum difference (over 64-bit RAM and crit 64-bit CUDA (npol=2)):", str(np.max(np.abs(array1 - array2))))

    
    # stop()
    # Now call the plotting function instead of run_benchmark directly
    # benchmarking_plots(in_NBIT_python, in_NBIT_cpp, out_NBIT_python, out_NBIT_cpp, run_python=run_python_baseline)
    # benchmarking_ntap_nchan_chunk(in_NBIT_cpp, out_NBIT_cpp)
    # benchmarking_batch_chunk(in_NBIT_cpp, out_NBIT_cpp, M=8, P=512, verify_diff=False)
    # plot_batched_results(in_NBIT_cpp, out_NBIT_cpp, verify_diff=False)
