# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 08:57:06 2025

@author: 22391643
"""
import numpy as np
from Numba_Logistic_Map import digital_logistic_map_numba

def float_to_binary_frac(x: float, k: int):
    """
    Convert a float in (0,1) to its binary-fraction string of length k,
    padding with zeros if the fraction terminates early.

    Args:
        x (float): A value strictly between 0 and 1.
        k (int): Number of binary digits to produce.

    Returns:
        str: A binary string of length k (no leading '0.').
    """
    if not (0 < x < 1):
        raise ValueError(f"x must be >0 and <1, got {x}")
    bits = []
    for _ in range(k):
        x *= 2
        if x >= 1.0:
            bits.append('1')
            x -= 1.0
        else:
            bits.append('0')
    return ''.join(bits)

def array_to_binary_string(arr: np.ndarray, k: int, filename: str):
    """
    Convert each float in `arr` (strictly between 0 and 1) to a fixed-length
    binary-fraction string of k bits, concatenate all results, print and return.

    Args:
        arr (np.ndarray): Array of floats in (0,1).
        k (int): Number of bits for each float's binary representation.

    Returns:
        str: Concatenated binary strings.
    """
    if not np.all((arr > 0) & (arr < 1)):
        raise ValueError("All elements must be strictly >0 and <1.")

    chunks = [float_to_binary_frac(x, k) for x in arr]
    result = ''.join(chunks)
    
    # Ensure the filename ends with .seed
    if not filename.lower().endswith('.seed'):
        filename += '.seed'

    # Write to file
    with open(filename, 'w') as f:
        f.write(result)
    
    return result

x = 0.314
n = 100
k = 64

xN = digital_logistic_map_numba(x, n, k)
array_to_binary_string(xN, k, 'test')
total_bits = k * len(xN)
