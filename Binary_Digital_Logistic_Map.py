# -*- coding: utf-8 -*-
"""
Takes float numbers created by Numba_Logistic_Map and converts them to binary
and saves them into a file for ENT/NIST testing

@author: 22391643
"""

import numpy as np
from Functions import transform_logistic_map
from Numba_Logistic_Map import digital_logistic_map_numba
import math

def float_to_binary_frac(x: float, k: int):
    '''
    Helper function that converts a float number to binary, where the binary 
    number is k digits long

    Parameters
    ----------
    x : Float number being converted to binary
    
    k : Number of digits in binary representation of x

    Returns
    -------
    strng : Binary representation of x with length k in string format

    '''
    
    # Error catching
    if not (0 < x < 1):
        raise ValueError(f"x must be >0 and <1, got {x}")
        
    # Nudge exact 1.0 down to avoid acc == 2^k
    if x == 1.0:
        x = math.nextafter(1.0, 0.0)
    
    #Creating binary string
    acc = int(math.floor(math.ldexp(float(x), k)))  # acc = ⌊x·2^k⌋
    
    strng = format(acc, f"0{k}b")
    
    return strng

def array_to_binary_string(array: np.ndarray, k: int, filename: str):
    '''
    Converts the array of float numbers into a single binary string, saved on a
    file for statistical testing.

    Parameters
    ----------
    strng : The binary string of a single float number within xN array
        
    k : Maximum number of digits in binary string 
        
    filename : Name of file that the final concatenated string is saved on

    Returns
    -------
    result : A single binary string representining all elements in the array
        
    total_bits : The total number of bits in the result string

    '''
    
    # Defining array
    arr = np.asarray(array, dtype=np.float64)
    
    # Error catching
    if not np.all((arr > 0) & (arr < 1)):
        raise ValueError("All elements must be strictly >0 and <1.")
        
    # Clamp into [0,1], then nudge any exact 1.0 down by one ULP
    arr = np.clip(arr, 0.0, 1.0)
    if np.any(arr == 1.0):
        arr = arr.copy()
        arr[arr == 1.0] = np.nextafter(1.0, 0.0)
    
    # Convert float numbers to binary and stitch strings together
    accs = np.floor(np.ldexp(arr, k)).astype(np.uint64)
    chunks = [format(int(a), f"0{k}b") for a in accs]
    result = ''.join(chunks)
    
    # Ensure filename ends with .seed
    if not filename.lower().endswith('.seed'):
        filename += '.seed'

    # Create File
    with open(filename, 'w') as f:
        f.write(result)
        
    # Calculate total_bits
    total_bits = len(result)
    
    return result, total_bits


n = 82_000 # 105,000 for d = 5, k = 48; 82,000 for d = 5, k = 64
k = 64
#X = np.linspace(0.001, 0.999, 20)

'''
for i in range(len(X)):
    
   xN = digital_logistic_map_numba(X[i], n, k)
   yN = transform_logistic_map(xN, T = 101, d = 5)
   _, total_bits = array_to_binary_string(yN, k, f"{X[i]}")
'''

x = 0.001
xN = digital_logistic_map_numba(x, n, k)
yN = transform_logistic_map(xN, T = 101, d = 5)
_, total_bits = array_to_binary_string(yN, k, f"Test{x}")