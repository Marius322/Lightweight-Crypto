# -*- coding: utf-8 -*-
"""
Digital Logistic Map that uses numpy arrays and works with numba

@author: 22391643
"""

import Numba_Map_Generator as NMG
import numpy as np

# Function

def unwrapped_digital_logistic_map(x, n, k, 
                                   op_keys,
                                   entry_op,
                                   head_idx,
                                   tail_strt, tail_len, tail_idxs,
                                   col_strt, col_len, col_idxs,
                                   inv_powers, C):
    
    # Catching Errors
    if x <= 0 or x >= 1:
        raise ValueError('x must lie between 0 and 1')
    if k <= 0:
        raise ValueError('k must be greater than 0')
    if n <= 0:
        raise ValueError('n must be greater than 1')
     
    # Initialising arrays
    xN = np.zeros(n+1, dtype = np.longdouble)
    xN[0] = x
    
    # Iterate n steps 
    for it in range(n):
        
        # If we've fallen out of (0,1), stop early
        if x == 0 or x == 1:
            print(f"ERROR - {xN[it-1]} went outside accepted range after {it} iterations")
            return xN[:it]
        
        # Convert current x to its k-bit fractional binary a[0..k-1]
        a = np.zeros(k, int)
        z = x
        for i in range(k):
            z *= 2
            if z >= 1:
                a[i] = 1
                z -= 1
            else:
                a[i] = 0
        
        # Compute A_i_j ops and initialise all other ops
        
        
    