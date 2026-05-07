# Minimum version 3.18 is recommended for modern CUDA support
cmake_minimum_required(VERSION 3.18)

# Define the project and enable both C++ and CUDA languages
project(PFB_CUDA LANGUAGES CXX CUDA)

# --- Build Configuration ---
# Set Default Build Type to Release (-O3) for maximum performance
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "Setting build type to: ${CMAKE_BUILD_TYPE}")
message(STATUS "Using C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "Using CUDA compiler: ${CMAKE_CUDA_COMPILER}")

# --- Find External Dependencies (From CPU Project) ---
# Finding the PkgConfig tool
find_package(PkgConfig REQUIRED)

# Locating the FFTW3 library (both single and double precision)
pkg_check_modules(FFTW3 REQUIRED fftw3 fftw3f)

# --- DIRECTORIES ---
# Define the path to your CPU repository (relative to CMakeLists.txt)
set(CPU_PATH "../../../PFB_CPU/codes/PFB_cpp")

# --- SOURCE FILES ---
# 1. Grab all local CUDA and C++ files
file(GLOB GPU_SRCS "src/*.cu" "src/*.cpp")

# 2. Grab all CPP files from the CPU repo
file(GLOB CPU_SRCS "${CPU_PATH}/src/*.cpp")

# 3. Filter out the CPU main.cpp to avoid "multiple definition of main"
list(FILTER CPU_SRCS EXCLUDE REGEX "main\\.cpp$")

# --- TARGET EXECUTABLE ---
# Combine sources and define the output executable
add_executable(PFB_app ${GPU_SRCS} ${CPU_SRCS})

# --- COMPILER FLAGS & ARCHITECTURE ---
set_target_properties(PFB_app PROPERTIES
    CUDA_ARCHITECTURES "120"
    CXX_STANDARD 17           # Updated to 17 to match your CPU project
    CXX_STANDARD_REQUIRED ON
)

# --- INCLUDES ---
target_include_directories(PFB_app PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CPU_PATH}/include
    ${FFTW3_INCLUDE_DIRS}     # Added FFTW3 includes
)

# --- LINK DIRECTORIES ---
# Only needed if FFTW3 is installed in a non-standard location found by PkgConfig
target_link_directories(PFB_app PRIVATE 
    ${FFTW3_LIBRARY_DIRS}
)

# --- LIBRARIES ---
# Link both the CUDA FFT library (cufft) and the CPU FFT library (FFTW3)
target_link_libraries(PFB_app PRIVATE 
    cufft                     # CUDA library for GPU FFTs
    ${FFTW3_LIBRARIES}        # CPU library required by your inherited .cpp files
)