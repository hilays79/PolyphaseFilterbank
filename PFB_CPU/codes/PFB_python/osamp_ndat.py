#!/usr/bin/env python3

import math

def get_buffer_info(i: int, ntaps: int, nchannels: int, ndat: int, input_step: int):
    """
    Calculates the number of steps in input window i (n_i) 
    and the integers making up the buffer range.
    """
    if i < 1:
        raise ValueError("i must be an integer >= 1 (assuming n_0 = 0)")

    sum_n = 0
    n_k = 0

    # Iteratively calculate n_k up to the requested i
    for k in range(1, i + 1):
        # The fractional term representing total steps possible up to block k
        base_term = (k * ndat - ntaps * nchannels) / input_step
        
        # Calculate new steps for the current window k
        n_k = math.floor(base_term) - sum_n + 1
        
        # Keep a running total of steps taken (equivalent to the sum notation)
        sum_n += n_k

    # After the loop, n_k holds the value for n_i
    n_i = n_k

    # Calculate the bounds for the buffer list
    # start = (\sum_{1}^{i} n_i) * input_step
    # end = i * ndat
    start_val = sum_n * input_step + 1
    end_val = i * ndat

    # Generate the list of integers.
    # We use a standard Python half-open range [start, end) which perfectly matches 
    # zero-indexed array slicing for buffers.
    buffer_integers = list(range(start_val, end_val))

    return n_i, buffer_integers

# ==========================================
# Example Usage:
# Based on the examples at the bottom of your page:
# i=6 => 120-8/3 means ndat=20, ntaps*nchannels=8, input_step=3
# ==========================================
if __name__ == "__main__":
    test_ntaps = 4
    test_nchannels = 256
    test_ndat = 216*6
    test_input_step = 216
    
    for test_i in range(1, 20):
        n_i_result, buffer_result = get_buffer_info(
            i=test_i, 
            ntaps=test_ntaps, 
            nchannels=test_nchannels, 
            ndat=test_ndat, 
            input_step=test_input_step
        )
        
        # Format the buffer output as a tuple (start, end)
        buffer_range = (buffer_result[0], buffer_result[-1] + 1) if buffer_result else "Empty"
        
        print(f"n_{test_i} = {n_i_result}, Buffer {test_i} range start to end: {buffer_range}, length {buffer_range[1]-buffer_range[0]+1}")