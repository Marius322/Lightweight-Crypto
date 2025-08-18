# -*- coding: utf-8 -*-
"""
Takes float numbers created by Numba_Logistic_Map and converts them to binary
and saves them into a file for ENT/NIST testing

@author: 22391643
"""

import numpy as np
from Digital_Logistic_Map import digital_logistic_map_mixed

def write_bits_seed(bits, x, k):
    """
    bits: 1-D iterable of 0/1 (e.g., list/np.array)
    x   : base name for the output file; file will be f"{x}.seed"
    """
    
    b = np.asarray(bits, dtype=np.uint8).ravel()
    
    # Catching errors
    if b.size == 0:
        raise ValueError("No bits provided.")
    if np.any((b != 0) & (b != 1)):
        raise ValueError("All elements must be 0 or 1.")
    
    bit_str = ''.join('1' if v else '0' for v in b.tolist())
    
    k = k -16
    fname = f"{k}_{x:.3f}.seed"
    with open(fname, 'w', encoding='utf-8') as f:
        f.write(bit_str)
    
    return fname, len(bit_str)

n = 82_000 # 105,000 for k = 48; 82,000 for k = 64; 140,000 for k = 32, 60,000 for k = 80
k = 80
x = 0.001
X1 = np.linspace(0.001, 0.999, 20) #only half used
X2 = np.linspace(0.56, 0.99, 10)
T = 101