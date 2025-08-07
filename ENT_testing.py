# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 09:30:59 2025

@author: 22391643
"""

#!/usr/bin/env python3
import math
import argparse
from collections import Counter

def bits_to_bytes(bit_str: str) -> bytes:
    """
    Packs an ASCII bit-string into bytes, dropping any trailing bits
    if len(bit_str) % 8 != 0.
    """
    bit_str = bit_str.strip()
    usable_len = (len(bit_str) // 8) * 8
    bit_str = bit_str[:usable_len]
    byte_vals = [int(bit_str[i:i+8], 2) for i in range(0, usable_len, 8)]
    return bytes(byte_vals)

def shannon_entropy(data: bytes) -> float:
    """
    Shannon entropy in bits per byte (max 8.0).
    """
    counts = Counter(data)
    total = len(data)
    return -sum((cnt/total) * math.log2(cnt/total) for cnt in counts.values())

def chi_square_stat(data: bytes) -> float:
    """
    Chi-square statistic against uniform 0–255.
    """
    counts = Counter(data)
    N = len(data)
    expected = N / 256
    return sum((counts[i] - expected)**2 / expected for i in range(256))

def arithmetic_mean(data: bytes) -> float:
    """
    Average byte value (should be ≈127.5).
    """
    return sum(data) / len(data)

def monte_carlo_pi(data: bytes) -> float:
    """
    Approximate π by interpreting successive byte-pairs as (x,y) in [0,1]^2.
    """
    N = len(data) // 2
    inside = 0
    for i in range(N):
        x = data[2*i]   / 255
        y = data[2*i+1] / 255
        if x*x + y*y <= 1:
            inside += 1
    return 4 * inside / N

def serial_correlation(data: bytes) -> float:
    """
    Serial correlation coefficient over the byte stream.
    """
    N = len(data)
    mean = sum(data) / N
    num = sum((data[i] - mean)*(data[(i+1)%N] - mean) for i in range(N))
    den = sum((data[i] - mean)**2 for i in range(N))
    return num / den

def analyse_seed_file(filename: str):
    # Read bit-string
    with open(filename, 'r') as f:
        bit_str = f.read()
    data = bits_to_bytes(bit_str)
    if len(data) < 16:
        raise ValueError("Not enough data (need at least 16 bytes).")

    # Compute metrics
    entropy    = shannon_entropy(data)
    chi_sq     = chi_square_stat(data)
    mean_val   = arithmetic_mean(data)
    pi_est     = monte_carlo_pi(data)
    ser_corr   = serial_correlation(data)

    # Print report
    print(f"Analysis of '{filename}':\n" + "-"*40)
    print(f"Total bytes analyzed       : {len(data)}")
    print(f"Shannon entropy            : {entropy:.6f} bits/byte (max 8.0)")
    print(f"Chi-square statistic       : {chi_sq:.3f} (ideal chi_sq ≈ 255)")
    print(f"Arithmetic mean            : {mean_val:.3f} (ideal ~ 127.5)")
    print(f"Monte Carlo π approximation: {pi_est:.6f} (ideal π ≈ 3.141593)")
    print(f"Serial correlation coeff.  : {ser_corr:.6f} (ideal ~ 0)")
    print("-"*40)

analyse_seed_file('0.0001.seed')