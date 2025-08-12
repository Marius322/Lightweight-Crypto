# -*- coding: utf-8 -*-
"""
Takes float numbers created by Numba_Logistic_Map and converts them to binary
and saves them into a file for ENT/NIST testing

@author: 22391643
"""

import numpy as np
from Functions import Logistic_map
from Numba_Logistic_Map import digital_logistic_map_numba

def float_to_binary_frac(x: float, k: int):
    '''
    Helper function that converts a float number to binary, where the binary 
    number is at most k digits long

    Parameters
    ----------
    x : Float number being converted to binary
    
    k : Maximum number of digits in binary representation of x

    Returns
    -------
    strng : Binary representation of x with maximimum length k in string format

    '''
    
    # Error catching
    if not (0 < x < 1):
        raise ValueError(f"x must be >0 and <1, got {x}")
    
    #Creating binary string
    bits = []
    for _ in range(k):
        x *= 2
        if x >= 1.0:
            bits.append('1')
            x -= 1.0
        else:
            bits.append('0')
        
    strng = ''.join(bits)
    
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
    
    # Error catching
    if not np.all((array > 0) & (array < 1)):
        raise ValueError("All elements must be strictly >0 and <1.")
    
    # Convert float numbers to binary and stitch strings together
    chunks = [float_to_binary_frac(x, k) for x in array]
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

x = 0.111
n = 1000
k = 64

Log = Logistic_map(x, n)
xN = digital_logistic_map_numba(x, n, k)
_, total_bits = array_to_binary_string(xN, k, f"{x}")
