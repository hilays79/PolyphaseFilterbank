# PolyphaseFilterbank

## Overview

This repository provides a high-performance computing (HPC) implementation of a Polyphase Filter Bank (PFB) tailored for advanced digital signal processing (DSP). The project focuses on complex data processing by channelising a high-speed time-domain signal into multiple narrower-band frequency channels. It achieves this using a computationally efficient combination of a polyphase FIR filter and a Fast Fourier Transform (FFT).

### The PFB Algorithm

* **C++ (CPU) Implementation:** Employs standard vectorization to perform overlapping, windowing, and FFT operations. The input time-series data is multiplied by a windowing function (such as a Hamming or Hann window) across multiple taps, and then fed into an FFT (typically leveraging FFTW) to extract the frequency channels. Both 32-bit and 64-bit data types are supported.
* **CUDA (GPU) Implementation:** Built for speed using massive parallelization. It leverages custom CUDA kernels for the polyphase filter and windowing stages. Tranposed FIR filter kernels are tested to maximise L1 cache throughput. Batching is implemented to minimise flushing of data from L2 cache to DRAM during the execution of the kernel. Batchsize of ~40% the L2 cache size provides maximum performance due to minimal flushing. The heavily optimized cuFFT library is then used to execute batched FFTs directly on the device, maximizing throughput for large-scale data processing or real-time physical simulations. Both 32-bit and 64-bit data types are supported. 32-bit preferred due to high FP32 FLOPS performance on GPUs (~30X speedup over the CPU implementation).

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

Clone the repository and its sub-directories directly from GitHub:

```bash
git clone https://github.com/hilays79/PolyphaseFilterbank.git
cd PolyphaseFilterbank

```

---

## Building with CMake

The project utilizes CMake to seamlessly manage and compile both the CPU (`PFB_CPU`) and GPU (`PFB_GPU`) source codes.

1. **Create a dedicated build directory** inside CPU and GPU root directories: `PFB_cpp` and `PFB_CUDA`, respectively.
```bash
mkdir build && cd build

```


2. **Generate the necessary Makefiles** (CMake will automatically detect your C++ and CUDA toolchains):
* On MacOS:



```bash
    CXX=g++-15 CC=gcc-15 cmake ..
```
*   On Linux:
```bash
    cmake ..
```
3.  **Compile the executables:**
```bash
    make
```
    > This will generate the `pfb_app` and `PFB_app` binary inside the `build/` directories of C++ and CUDA implementations, respectively.

---

## Executable Usage

Once compiled successfully, the executables will be generated in their respective build subdirectories. You can run them by passing the required algorithm parameters. 

Note that the CUDA implementation has the ability to use both C++ (CPU) and CUDA (GPU) implementations to compare the results and execution times. Thus, the C++ implementation need not to executed separately. 

In `main.cu` of the `PFB_CUDA` implementation, making `CPU_verification = true;` results in the CUDA code compiling both C++ and CUDA implementations.

If the `read_from_file` is enabled and input binary files are absent in the `Data/` directory, benchmarking tests using Python can be run. `create_binary_test_signals()` in the code `generate_binary_data.py` in `PFB_python/` directory has the ability to create binary data formatted test signals.

### Example execution (only C++ CPU):

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

To validate accuracy and compare the execution speed across the different environments, you can utilize the provided Python scripts. These benchmarks test execution times, setup times, compares CPU and GPU implementations.

The primary testing and benchmarking suite is handled by `python_c_comparison.py`. This script handles the end-to-end pipeline:

1. It checks for the necessary binary test signals (e.g., complex phasors) in the `Data/input_files/` directory and generates them if they don't exist.
2. It runs the Python PFB implementation and records the time.
3. It executes the CUDA compiled `PFB_app` binary (handling the relative paths automatically) and records its setup and execution times.
4. Make sure `CPU_verification` is `true` if one desires to benchmark both CPU and GPU times.

### Execution

**Step 1:** Navigate to the Python codes directory and ensure you have python3 and pip installed:

```console
cd PFB_python/

```

**If you prefer to work in a virtual environment:**

```console
virtualenv -p python3 .venv
source .venv/bin/activate

```

**Step 2:** If you do not have the dependencies installed, run:

```console
pip install numpy scipy matplotlib ipdb

```

**Step 3:** Run the benchmarking script:

```console
python python_c_comparison.py

```

Several different functions are provided to perform benchmarking between different tap sizes, channels, batch sizes, and create relevant plots.

```

```