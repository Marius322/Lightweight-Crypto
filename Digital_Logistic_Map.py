# -*- coding: utf-8 -*-
"""
This is a digital logistic map that works for any k and is not parallerised

@author: 22391643
"""

import numpy as np
import Digital_Map_Generator as DMG

def digital_logistic_map(x: float, n: int, k: int):
    '''
    Inputs:
      x = real number between 0 and 1
      n = number of iterations
      k = number of digits 

    Outputs:
      xN = logistic sequence of real numbers
    '''
    # Error checking
    if x <= 0 or x >= 1:
        print('ERROR - x must lie between 0 and 1')
        return
    if k <= 0:
        print('ERROR - k must be greater or equal to 1')
        return
    if n <= 0:
        print('ERROR - n must be greater or equal to 1')
        return

    # Initialize output array 
    xN = np.zeros(n+1, dtype = np.longdouble)
    xN[0] = x
    C = 2*k-2
    powers = 2.0 ** -np.arange(1, C+1)

    # Pull in our four parsed maps once per call ---
    all_cols, aij_defs, top_defs, rec_defs = DMG.generate_parsed_maps(k)
    formulas = {**top_defs, **rec_defs}

    # Iterate n steps 
    for l in range(n):
        # If we've fallen out of (0,1), stop early
        if x == 0 or x == 1:
            print(f"ERROR - {xN[l-1]} went outside accepted range after {l} iterations")
            return xN[:l]

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
        bit_cache = {}
        for key, (head_key, tail_keys) in aij_defs.items():
            # head_key == (1,i,j)
            _, i, j = head_key
            bit_cache[head_key] = a[i-1] ^ a[j-1]

        # Compute all other ops
        for r in range(4, C):
            for d,i,j in all_cols[r]:
                 if d != 1:
                     key = (d,i,j)
                     head, tails = formulas[key]
                     if bit_cache.get(head, 0) == 0:
                         bit_cache[key] = 0
                     else:
                         xorv = 0
                         for t in tails:
                             xorv ^= bit_cache.get(t, 0)
                         bit_cache[key] = bit_cache[head] & xorv
                     
        # Build the XCol array by XOR‑summing each column of all_cols 
        XCol = np.zeros(C, dtype=int)
        for r, entries in all_cols.items():
            acc = 0
            for (d,i,j) in all_cols[r]:
                acc ^= bit_cache.get((d,i,j), 0)
            XCol[C - 1 - r] = acc
        
        # Convert back to real
        x = np.dot(XCol, powers)
        xN[l+1] = x
    
    return xN

