# -*- coding: utf-8 -*-
"""
Runs ENT test for given file

@author: 22391643
"""

import math
from collections import Counter

def bits_to_bytes(bit_str: str):
    '''
    Helper function that divides binary into bytes of size 8, dropping left 
    over bits

    Parameters
    ----------
    bit_str : string of binary numbers

    Returns
    -------
    byts : bytes of length 8 representing a value between 0 and 255

    '''
    bit_str = bit_str.strip()
    
    # Split String into bytes
    usable_len = (len(bit_str) // 8) * 8
    bit_str = bit_str[:usable_len]
    
    # Assign a value to each byte
    byte_vals = [int(bit_str[i:i+8], 2) for i in range(0, usable_len, 8)]
    result = bytes(byte_vals)
    
    return result

def shannon_entropy(data: bytes):
    """
    Calculates Shannon's Entropy for each byte
    """
    # Number of byte values that appear
    counts = Counter(data)
    
    # Number of bytes in data
    total = len(data)
    
    #Calculation
    result = - sum((cnt/total) * math.log2(cnt / total) for cnt in counts.values())
    
    return result

def chi_square_stat(data: bytes):
    """
    Calculates Chi-square statistic for each byte 
    """
    # Number of byte values that appear
    counts = Counter(data)
    
    # Number of bytes in data
    N = len(data)
    
    # Calculation 
    expected = N / 256
    return sum((counts[i] - expected)**2 / expected for i in range(256))

def arithmetic_mean(data: bytes):
    """
    Calculates average byte value 
    """
    # Calculation
    result = sum(data) / len(data)
    
    return result

def monte_carlo_pi(data: bytes):
    """
    Approximates π by interpreting successive byte-pairs as (x,y) in unit 
    square [0, 1] x [0, 1]
    """
    # Ensures every byte has a pair
    N = len(data) // 2
    
    # Count the number of pairs that lie within quarter-circle with radius = 1
    inside = 0
    for i in range(N):
        x = data[2 * i] / 255
        y = data[2 * i + 1] / 255
        if x*x + y*y <= 1:
            inside += 1
    
    # Calculation
    result = 4 * inside / N
    
    return result

def serial_correlation(data: bytes):
    """
    Calculates the correlation between a byte and its' successor
    """
    
    # Total Number of bytes
    N = len(data)
    
    # Average byte value
    mean = sum(data) / N
    
    # Calculation
    num = sum((data[i] - mean)*(data[(i+1)%N] - mean) for i in range(N))
    den = sum((data[i] - mean)**2 for i in range(N))
    result = num/den
    
    return result

def analyse_seed_file(filename: str):
    '''
    

    Parameters
    ----------
    filename : File where data in binary string form is saved

    Returns
    -------
    ENT Test

    '''
    
    # Read bit-string
    with open(filename, 'r') as f:
        raw = f.read()
    bit_str = ''.join(ch for ch in raw if ch in ('0','1'))
    data = bits_to_bytes(bit_str)
    if len(data) < 16:
        raise ValueError("Not enough data - need at least 16 bytes.")

    # Compute tests
    entropy    = shannon_entropy(data)
    chi_sq     = chi_square_stat(data)
    mean_val   = arithmetic_mean(data)
    pi_est     = monte_carlo_pi(data)
    ser_corr   = serial_correlation(data)

    # Print ENT test
    print(f"Analysis of '{filename}':\n" + "-"*40)
    print(f"Total bytes analyzed       : {len(data)}")
    print(f"Shannon entropy            : {entropy:.6f} bits/byte (max 8.0)")
    print(f"Chi-square statistic       : {chi_sq:.3f} (ideal chi_sq ≈ 255)")
    print(f"Arithmetic mean            : {mean_val:.3f} (ideal ~ 127.5)")
    print(f"Monte Carlo π approximation: {pi_est:.6f} (ideal π ≈ 3.141593)")
    print(f"Serial correlation coeff.  : {ser_corr:.6f} (ideal ~ 0)")
    print("-"*40)
    
    return

analyse_seed_file('0.786.seed')