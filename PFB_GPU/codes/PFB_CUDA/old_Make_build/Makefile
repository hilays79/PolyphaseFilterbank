# ---------------------------------------------------
# Simple CUDA Makefile
# ---------------------------------------------------

# The CUDA compiler
NVCC = nvcc

# Compiler flags
NVCC_FLAGS = -arch=compute_120 -code=sm_120 -O3

# The name of your output executable
OBJ_DIR = objs
TARGET = $(OBJ_DIR)/PFB_app

# --- DIRECTORIES ---
# Define the path to your CPU repository
CPU_PATH = ../../../PFB_CPU/codes/PFB_cpp

# Include directories for header files (Using the new variable!)
INCLUDES = -Iinclude -I$(CPU_PATH)/include

# --- SOURCE FILES ---
# 1. Grab all the CUDA files
GPU_SRCS = $(wildcard src/*.cu)

# 2. Grab all CPP files from the CPU repo, filtering out the CPU main.cpp
CPU_SRCS = $(filter-out %main.cpp, $(wildcard $(CPU_PATH)/src/*.cpp))

# 3. Combine them
SOURCES = $(GPU_SRCS) $(CPU_SRCS)

# ---------------------------------------------------

# 'all' is the default target that runs when you just type 'make'
all: $(TARGET)

# Rule to compile the source code into the executable
$(TARGET): $(SOURCES)
	mkdir -p $(OBJ_DIR)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(SOURCES) -o $(TARGET)

# Rule to compile (if needed) and immediately run the program
run: $(TARGET)
	./$(TARGET)

# Rule to delete the compiled executable to keep the directory clean
clean:
	rm -f $(TARGET)