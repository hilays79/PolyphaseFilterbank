# Specify the minimum version of CMake required to build this project
cmake_minimum_required(VERSION 3.10)

# Project name and language
project(PFB CXX)

# --- Set Default Build Type to Release (-O3) for maximum performance ---
set(CMAKE_BUILD_TYPE Release)
message(STATUS "Setting default build type to: Release (-O3)")

# echo the C++ compiler being used
message(STATUS "Using C++ compiler: ${CMAKE_CXX_COMPILER}")

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find relevant linked libraries (e.g., FFTW)
# Finding the PkgConfig tool
find_package(PkgConfig REQUIRED)
# Locating the FFTW3 library (both single and double precision) using PkgConfig
pkg_check_modules(FFTW3 REQUIRED fftw3 fftw3f)

# Adding path for header files
include_directories(include)

# Collecting all source files in SOURCES variable
file(GLOB SOURCES "src/*.cpp")

# Create an executable from the source files
add_executable(pfb_app ${SOURCES})

# 7. Link FFTW3
target_include_directories(pfb_app PRIVATE ${FFTW3_INCLUDE_DIRS})
target_link_directories(pfb_app PRIVATE ${FFTW3_LIBRARY_DIRS})
target_link_libraries(pfb_app PRIVATE ${FFTW3_LIBRARIES})