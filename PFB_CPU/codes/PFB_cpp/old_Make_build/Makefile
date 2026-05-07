# This Makefile works on MacOS.
# To use this, move this to the root directory.
# In the root directory, run "mkdir objs"
# Then run "make build" to compile
# or "make run" to both compile and run the executable (stored in objs/)

# --- Compiler Settings ---
CXX = g++-15
CXXFLAGS = -std=c++17 -O3 -Iinclude -I/opt/homebrew/include

# --- Library Linking ---
# LDFLAGS: Tells the compiler WHERE to look for libraries
LDFLAGS = -L/opt/homebrew/lib

# LDLIBS: Tells the compiler WHICH libraries to link
LDLIBS = -lfftw3 -lfftw3f

# --- File Paths ---
OBJ_DIR = objs
SOURCES = $(wildcard src/*.cpp) # wildcard finds all .cpp files in the src directory
EXECUTABLE = $(OBJ_DIR)/PFB_app
.PHONY: build # build is not a directory, but a command

# --- Build Targets ---

# command to compile the code
build:
	@echo "Compiling the code..."
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $(EXECUTABLE) $(LDFLAGS) $(LDLIBS)

# command to run the executable
run: build
	@echo "Running the executable..."
	./$(EXECUTABLE)

# command to clean the build files
clean:
	@echo "Cleaning build files..."
	rm -rf $(OBJ_DIR)/*
