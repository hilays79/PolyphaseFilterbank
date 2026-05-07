<h1 align="center">Fourier Space: Polyphase Filter Bank (PFB) Benchmarking</h1>

This repository contains tools and implementations for benchmarking a Polyphase Filter Bank (PFB) spectrometer. It features both a baseline Python implementation and an optimized C++ implementation to compare processing speeds, setup times, and numerical accuracy.

The project is structured with the Python and C++ source codes maintained as independent Git submodules, alongside a structured `Data` directory for binary signal inputs and outputs (.dada files).

---

##  1. Cloning the Repository

**Clone the repository with all submodules:**
```console
git clone https://github.com/hilays79/PFB_CPU
```

**Navigate into the project:**
```console
cd PFB_CPU
```

> *(If you accidentally cloned without the flag, you can fetch the submodules by running: `git submodule update --init --recursive`)*

---

##  2. Directory Structure & Data

The repository tracks the following core structure:

-  `codes/PFB_cpp/`: Submodule containing the C++ implementation.
-  `codes/PFB_python/`: Submodule containing the Python implementation and the main benchmarking scripts.
-  `Data/`: Directory structure for inputs and outputs. 

> **Note on Data:** To keep the repository lightweight, massive binary `.dada` files and animation `.mp4` files are strictly ignored by Git. The `Data` folder contains placeholder files to preserve the necessary directory tree. The benchmarking script will automatically generate the required binary input files on the fly if they are missing.

---

##  3. Compiling the C++ Backend

Before running any benchmarks, you must compile the C++ executable. The build process uses CMake.
**Step 1:** Create C++ build directory if it does not exist
```console
mkdir codes/PFB_cpp/build
```

**Step 1:** Navigate to the C++ build directory
```console
cd codes/PFB_cpp/build
```

**Step 2:** Configure the project with CMake

On MacOS
```console
CXX=g++-15 CC=gcc-15 cmake .. ..
```
On Linux
```console
cmake .. ..
```

**Step 3:** Compile the executable
```console
make
```

This will generate the `pfb_app` binary inside the `build/` directory.

---

##  4. Running the Benchmarks

The primary testing and benchmarking suite is handled by `python_c_comparison.py`. This script handles the end-to-end pipeline:

1. It checks for the necessary binary test signals (e.g., complex phasors) in the `Data/input_files/` directory and generates them if they don't exist.
2. It runs the Python PFB implementation and records the time.
3. It executes the compiled C++ `pfb_app` binary (handling the relative paths automatically) and records its setup and execution times.
4. It compares the output `.dada` files from both languages to ensure mathematical parity (calculating the maximum absolute difference).

### Execution

**Step 1:** Navigate to the Python codes directory and ensure you have python3 and pip installed
```console
cd ../../PFB_python/
```
**If you prefer to work in a virtual environment
```console
virtualenv -p python3 .venv
source .venv/bin/activate
```

**Step 2:** If you do not have the dependencies installed, run
```console
pip install numpy scipy matplotlib ipdb
```

**Step 3:** Run the benchmarking script
```console
python python_c_comparison.py
```

### Expected Output

The script will output a table comparing the performance across varying window sizes (`W`).

```text
W       | Py Time (s)  | C++ Tot (s)  | C++ Set (s)  | C++ Exe (s)  | C++ Set/Exec | Speedup   | Max Diff  
---------------------------------------------------------------------------------------------------------
100     | ...          | ...          | ...          | ...          | ...          | ...x      | ...
```

---

##  Dependencies

- **Python:** pip, NumPy, SciPy, Matplotlib (for visualization/testing), `ipdb` (for debugging).
- **C++:** CMake (v3.10+), a standard C++ compiler (GCC) with C++17 support.
