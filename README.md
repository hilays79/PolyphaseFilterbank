# PolyphaseFilterbank

## Overview

* This repository provides a high-performance computing (HPC) implementation of a Polyphase Filter Bank (PFB) tailored for advanced digital signal processing (DSP).
* The project focuses on complex data processing by channelising a high-speed time-domain signal into multiple narrower-band frequency channels.
* It achieves this using a computationally efficient combination of a polyphase FIR filter and a Fast Fourier Transform (FFT).
* Explored optimisations on GPU: L2 cache memory management using batching, L1 cache throughput using tranposed kernels, and redistribution of thread compute using different kernel grid sizes and convolution operations.
* Development of a more advanced and optimised algorithm has been moved to a private repository. Contact author at Hilay.Shah@anu.edu.au for more information.

### The PFB Algorithm

* **C++ (CPU) Implementation:**
* Employs standard vectorisation to perform overlapping, windowing, and FFT operations.
* The input time-series data is multiplied by a windowing function (such as a Hamming or Hann window) across multiple taps, and then fed into an FFT (typically leveraging FFTW) to extract the frequency channels.
* Both 32-bit and 64-bit data types are supported.


* **CUDA (GPU) Implementation:**
* Built for speed using massive parallelisation.
* It leverages custom CUDA kernels for the polyphase filter and windowing stages.
* Tranposed FIR filter kernels are tested to maximise L1 cache throughput.
* Batching is implemented to minimise flushing of data from L2 cache to DRAM during the execution of the kernel.
* Batchsize of ~40% the L2 cache size provides maximum performance due to minimal flushing.
* The heavily optimised cuFFT library is then used to execute batched FFTs directly on the device, maximising throughput for large-scale data processing or real-time physical simulations. My batching implementation also optimises cuFFT execution of the FFT.
* Both 32-bit and 64-bit data types are supported. 32-bit preferred due to high FP32 FLOPS performance on GPUs (~30X speedup over the C++ CPU implementation).



---

## Dependencies

Ensure your environment is configured with the necessary compilers and libraries before building:

**C++ & System Dependencies:**

* CMake (>= 3.15)
* C++ Compiler (GCC, Clang, or MSVC) supporting C++17 or later
* FFTW3 (for CPU FFT operations)

**CUDA Dependencies:**

* NVIDIA CUDA Toolkit (>= 11.0)
* `nvcc` compiler (included in the toolkit)
* cuFFT library (included in the toolkit)

**Python (for Benchmarking):**

* Python 3.8+
* `numpy`
* `scipy`
* `matplotlib` (for visualizing filter responses and output data)

---

## Cloning the Repository

* Clone the repository and its sub-directories directly from GitHub:

```bash
git clone https://github.com/hilays79/PolyphaseFilterbank.git
cd PolyphaseFilterbank

```

---

## Building with CMake

* The project utilises CMake to manage and compile both the CPU (`PFB_CPU`) and GPU (`PFB_GPU`) source codes.

1. **Create a dedicated build directory** inside CPU and GPU root directories: `PFB_cpp` and `PFB_CUDA`, respectively.
```bash
mkdir build && cd build

```


2. **Generate the necessary Makefiles** (CMake will automatically detect your C++ and CUDA toolchains and throw errors if necessary libraries are not installed):
* On MacOS (for C++):



```bash
    CXX=g++-15 CC=gcc-15 cmake ..
```
* On Linux (for both C++ and CUDA):
     
```bash
    cmake ..
```

3. **Compile the executables:**
   ```bash
   make

```

* This will generate the `pfb_app` and `PFB_app` binary inside the `build/` directories of C++ and CUDA implementations, respectively.

---

## Executable Usage

* Once compiled successfully, the executables will be generated in their respective build subdirectories.
* You can run them by passing the required algorithm parameters.
* Note that the CUDA implementation has the ability to use both C++ (CPU) and CUDA (GPU) implementations to compare the results and execution times. Thus, the C++ implementation need not to executed separately.
* In `main.cu` of the `PFB_CUDA` implementation, making `CPU_verification = true;` results in the CUDA code compiling both C++ and CUDA implementations.
* If the `read_from_file` is enabled and input binary files are absent in the `Data/` directory, benchmarking tests using Python can be run to generate the files.
* Alternatively, `create_binary_test_signals()` in the code `generate_binary_data.py` in `PFB_python/` directory has the ability to create binary data formatted test signals.

### Example execution (only C++ CPU, not necessary if CPU_verification is turned on in CUDA implementation):

```bash
./pfb_app 100 64 64 0

```

* The **first argument** refers to the number of `n_taps * n_chan` blocks in the input data.
* The **second and third arguments** are the data sizes of the input (64-bit/32-bit) and the output data (64-bit/32-bit).
* The **fourth argument** is whether to generate the signal (`0`) or use input binary dada files/`read_from_file` (`1`).

### Example execution (CUDA GPU):

```bash
./PFB_app 100 64 64 1 4 256 1 0

```

* The **first argument** refers to the number of `n_taps * n_chan` blocks in the input data.
* The **second and third arguments** are the data sizes of the input (64-bit/32-bit) and the output data (64-bit/32-bit).
* The **fourth argument** is whether to generate the signal (`0`) or use input binary dada files/`read_from_file` (`1`).
* The **fifth and sixth arguments** are `n_taps` and `n_chan` for polyphase filtering, respectively.
* The **seventh argument** is the number of batches to execute polyphase filtering and FFT in.
* The **eighth argument** disables (`0`) or enables (`1`) atomic convolution kernel for the FIR filtering stage. Our benchmarking indicates that disabling it is faster.

---

## Python Benchmarking

* To validate accuracy and compare the execution speed across the different environments, you can utilise the provided Python scripts.
* These benchmarks test execution times, setup times, compares CPU and GPU implementations.
* The primary testing and benchmarking suite is handled by `python_c_comparison.py`. This script handles the end-to-end pipeline:

1. It checks for the necessary binary test signals (e.g., complex phasors) in the `Data/input_files/` directory and generates them if they don't exist.
2. It runs the Python PFB implementation (if run_python enabled), C++ PFB implementation (if CPU_verification=true in main.cu of PFB_CUDA), CUDA PFB implementation, and records the time.
3. The execution of the CUDA compiled `PFB_app` binary (handling the relative paths automatically) and its setup and execution times recording is done automatically.
4. Make sure `CPU_verification` is `true` if one desires to benchmark both CPU and GPU times.

### Execution

* **Step 1:** Navigate to the Python codes directory and ensure you have python3 and pip installed:
```console
cd PFB_python/

```


* **If you prefer to work in a virtual environment:**

```console
  virtualenv -p python3 .venv
  source .venv/bin/activate

```

* **Step 2:** If you do not have the dependencies installed, run:
```console
pip install numpy scipy matplotlib ipdb

```


* **Step 3:** Run the benchmarking script:
```console
python python_c_comparison.py

```


* If all steps have been followed correctly, the user should see the terminal output that looks like the following benchmark run on my machine with RTX5070Ti GPU.

| Data (GB) | IN/OUT | Py Time | C++ Set | C++ Exe | GPU Set | GPU Exe | Spd(C/Py) | Spd(G/Py) | Exe(C/G) | Diff P-C | Diff P-G | Diff C-G |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0004 | 32/32 | 0.41289 | 0.00045 | 0.00020 | 0.00115 | 0.00095 | 635.9x | 196.5x | 0.2x | 9.0e-05 | 6.3e-05 | 1.2e-04 |
| 0.0008 | 32/32 | 0.00375 | 0.00058 | 0.00035 | 0.00124 | 0.00097 | 4.0x | 1.7x | 0.4x | 1.2e-04 | 9.0e-05 | 1.5e-04 |
| 0.0015 | 32/32 | 0.00647 | 0.00095 | 0.00066 | 0.00135 | 0.00098 | 4.0x | 2.8x | 0.7x | 1.2e-04 | 9.0e-05 | 1.5e-04 |
| 0.0031 | 32/32 | 0.01328 | 0.00162 | 0.00122 | 0.00159 | 0.00099 | 4.7x | 5.1x | 1.2x | 1.2e-04 | 9.3e-05 | 1.5e-04 |
| 0.0061 | 32/32 | 0.02761 | 0.00305 | 0.00249 | 0.00192 | 0.00099 | 5.0x | 9.5x | 2.5x | 1.2e-04 | 9.3e-05 | 1.5e-04 |
| 0.0122 | 32/32 | 0.05621 | 0.00562 | 0.00581 | 0.00248 | 0.00106 | 4.9x | 15.9x | 5.5x | 1.2e-04 | 9.3e-05 | 1.5e-04 |
| 0.0244 | 32/32 | 0.11198 | 0.01093 | 0.01209 | 0.00342 | 0.00131 | 4.9x | 23.6x | 9.2x | 1.2e-04 | 9.3e-05 | 1.5e-04 |
| 0.0488 | 32/32 | 0.22099 | 0.02285 | 0.02418 | 0.00545 | 0.00172 | 4.7x | 30.8x | 14.1x | 1.2e-04 | 9.3e-05 | 1.8e-04 |
| 0.0977 | 32/32 | 0.44544 | 0.04213 | 0.05000 | 0.00983 | 0.00246 | 4.8x | 36.2x | 20.4x | 1.2e-04 | 1.2e-04 | 1.8e-04 |
| 0.1953 | 32/32 | 0.89004 | 0.07893 | 0.09599 | 0.01809 | 0.00399 | 5.1x | 40.3x | 24.1x | 1.2e-04 | 1.2e-04 | 1.8e-04 |
| 0.3906 | 32/32 | 1.77462 | 0.15684 | 0.18719 | 0.03409 | 0.00697 | 5.2x | 43.2x | 26.9x | 1.2e-04 | 1.2e-04 | 1.8e-04 |
| 0.7812 | 32/32 | 5.81942 | 0.32366 | 0.38186 | 0.06491 | 0.01314 | 8.2x | 74.6x | 29.1x | 1.2e-04 | 1.2e-04 | 1.8e-04 |
| 1.5625 | 32/32 | 15.14191 | 0.62367 | 0.77989 | 0.12402 | 0.02517 | 10.8x | 101.5x | 31.0x | 1.2e-04 | 1.2e-04 | 1.8e-04 |

* This is run with the most basic GPU implementation, and even that is 31X faster than the C++ CPU implementation and 101.5X faster than the Python CPU implementation.
* Further optimisations with batching, and transposed kernels provide additional benefits.
* The user can play around with the other plotting scripts and advanced benchmarking to generate more useful and detailed comparisons with batching.

```

```