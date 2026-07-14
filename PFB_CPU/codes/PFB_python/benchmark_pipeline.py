#!/usr/bin/env python3

import os
import numpy as np
from ipdb import set_trace as stop
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import re
import pandas as pd
import re

def calculate_benchmark_data_rate(filepath):
    # --- Regex Patterns for Global Parameters ---
    # Matches: "taps: 4, output channels: 256"
    taps_chan_pattern = re.compile(r"taps:\s*(\d+),\s*output channels:\s*(\d+)")
    
    # Matches: "OSR = 32/27" or "OSR:32/27" (handles both fractions and standard decimals)
    osr_pattern = re.compile(r"OSR\s*[:=]\s*([\d.]+)(?:/([\d.]+))?")
    
    # Matches: "Best ndat should be 1227744 with an expected data rate >= 17.147 Gsamples/s"
    best_ndat_pattern = re.compile(r"Best ndat should be\s*(?:>=|=)?\s*(\d+)\s*with an expected data rate\s*(?:>=|=)?\s*([\d.]+)")    
    # --- Regex Pattern for Iteration Data ---
    log_pattern = re.compile(
        r"ndat:\s*(?P<ndat>\d+),\s*"
        r"target_ndat:\s*(?P<target_ndat>\d+),\s*"
        r"npol:\s*(?P<npol>\d+),\s*"
        r"ndim:\s*(?P<ndim>\d+),\s*"
        r"nbit:\s*(?P<nbit>\d+),\s*"
        r"iters:\s*(?P<iters>\d+),\s*"
        r"time:\s*(?P<time>[\d.]+)"
    )

    # Dictionary to hold the global parameters as we find them
    metadata = {
        'oversampled': False,
        'osr': 1.0,          # Defaults to 1.0 for critically sampled
        'ntaps': None,
        'nchannels': None,
        'best_ndat': None,
        'expected_data_rate_gsps': None # Added to capture the final metric
    }
    
    parsed_data = []

    with open(filepath, 'r') as file:
        for line in file:
            # 1. Check for global metadata in the current line
            if "oversampled PFB" in line or "Oversampled PFB" in line:
                metadata['oversampled'] = True
                
            osr_match = osr_pattern.search(line)
            if osr_match:
                numerator = float(osr_match.group(1))
                if osr_match.group(2): # If it's formatted as a fraction (e.g., 32/27)
                    denominator = float(osr_match.group(2))
                    metadata['osr'] = numerator / denominator
                else: # If it's just a float (e.g., OSR: 1.185)
                    metadata['osr'] = numerator
                    
            taps_chan_match = taps_chan_pattern.search(line)
            if taps_chan_match:
                metadata['ntaps'] = int(taps_chan_match.group(1))
                metadata['nchannels'] = int(taps_chan_match.group(2))
                
            best_ndat_match = best_ndat_pattern.search(line)
            if best_ndat_match:
                metadata['best_ndat'] = int(best_ndat_match.group(1))
                metadata['expected_data_rate_gsps'] = float(best_ndat_match.group(2))

            # 2. Check for the recurring iteration data
            match = log_pattern.search(line)
            if match:
                ndat = int(match.group('ndat'))
                target_ndat = int(match.group('target_ndat'))
                npol = int(match.group('npol'))
                ndim = int(match.group('ndim'))
                nbit = int(match.group('nbit'))
                iters = int(match.group('iters'))
                time_s = float(match.group('time'))
                
                # Calculate the data size in bits and bytes
                bitsize = nbit * ndim * npol
                total_bits = target_ndat * bitsize
                total_bytes = total_bits / 8
                
                # Convert to Gigabytes (GB)
                total_gb = total_bytes / 1e9 
                
                # Calculate Data Rate (GB/s)
                data_rate_gbs = total_gb / time_s if time_s > 0 else 0
                
                # Append the core row data
                parsed_data.append({
                    'ndat': ndat,
                    'target_ndat': target_ndat,
                    'npol': npol,
                    'ndim': ndim,
                    'nbit': nbit,
                    'iters': iters,
                    'time': time_s,
                    'data_rate_gbs': data_rate_gbs
                })
                
    # 3. Inject the global metadata into every row of the parsed data
    for row in parsed_data:
        row.update(metadata)
            
    return parsed_data

def plot_parsed_data(filepath):
    # Load and immediately sort the data by ndat
    df = pd.DataFrame(calculate_benchmark_data_rate(filepath)).sort_values(by='ndat')
    
    transform = "Oversampled" if df['oversampled'].iloc[0] else "Critically Sampled"
    title = (f"{transform} PFB | npol: {df['npol'].iloc[0]}, nbit: {df['nbit'].iloc[0]}, ndim: {df['ndim'].iloc[0]}\n"
             f"ntaps: {df['ntaps'].iloc[0]}, nchannels: {df['nchannels'].iloc[0]}, OSR: {df['osr'].iloc[0]:.3f}")

    plt.figure(figsize=(5,5))
    
    # plt.plot draws solid lines. Adding marker='.' adds the scatter points on top.
    plt.plot(df['ndat'], df['data_rate_gbs'], label="Data Rate", color='C0', marker='.')
    
    b1 = int(df['best_ndat'].iloc[0])
    plt.axvline(b1, color='red', ls='--', alpha=0.6, label=f'Best ndat ({b1})')

    plt.xscale("log")
    plt.xlabel("ndat [samples]")
    plt.ylabel("Data Rate [GB/s]")
    plt.title(title)
    plt.legend()
    plt.savefig("images/pipeline_benchmark.png", dpi=300, bbox_inches='tight')
    print("\nPlot saved to images/pipeline_benchmark.png")
    plt.close()

def plot_compare_parsed_data(filepath1, filepath2, label1, label2):
    # Load and immediately sort both dataframes by ndat
    df1 = pd.DataFrame(calculate_benchmark_data_rate(filepath1)).sort_values(by='ndat')
    df2 = pd.DataFrame(calculate_benchmark_data_rate(filepath2)).sort_values(by='ndat')
    
    transform = "Oversampled" if df1['oversampled'].iloc[0] else "Critically Sampled"
    title = (f"{transform} PFB | npol: {df1['npol'].iloc[0]}, nbit: {df1['nbit'].iloc[0]}, ndim: {df1['ndim'].iloc[0]}\n"
             f"ntaps: {df1['ntaps'].iloc[0]}, nchannels: {df1['nchannels'].iloc[0]}, OSR: {df1['osr'].iloc[0]:.3f}")

    plt.figure(figsize=(5,5))
    
    # Plot sorted data with lines and markers
    plt.plot(df1['ndat'], df1['data_rate_gbs'], label=label1, color='C0', marker='.')
    plt.plot(df2['ndat'], df2['data_rate_gbs'], label=label2, color='C1', marker='.')
    
    b1, b2 = int(df1['best_ndat'].iloc[0]), int(df2['best_ndat'].iloc[0])
    
    plt.axvline(b1, color='C0', ls='--', alpha=0.6, label=f'Best {label1} ({b1})')
    plt.axvline(b2, color='C1', ls='--', alpha=0.6, label=f'Best {label2} ({b2})')

    plt.xscale("log")
    plt.xlabel("ndat [samples]")
    plt.ylabel("Data Rate [GB/s]")
    plt.title(title)
    plt.legend()
    plt.savefig("images/pipeline_benchmark_compare.png", dpi=300, bbox_inches='tight')
    print("\nPlot saved to images/pipeline_benchmark_compare.png")
    plt.close()

if __name__ == "__main__":
    filepath = "/home/hshah/PolyphaseFilterbank/PFB_CPU/codes/PFB_python/images/"
    over = False
    if over == True:
        over_key = "over"
    else:
        over_key = "critical"
    plot_parsed_data(filepath+'benchmark_pipeline_{}_unfused.txt'.format(over_key))
    plot_compare_parsed_data(filepath+'benchmark_pipeline_{}_unfused.txt'.format(over_key),
    filepath+'benchmark_pipeline_{}_fused.txt'.format(over_key), "unfused", "fused")
    stop()
