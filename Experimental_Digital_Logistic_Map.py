# -*- coding: utf-8 -*-
"""
This is a digital map that is optimised using Numba 

@author: 22391643
"""
import numpy as np
import Digital_Map_Generator as DMG

def listed_digital_logistic_map(x: float, n: int, k: int):
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
        raise ValueError('x must lie between 0 and 1')
    if k <= 0:
        raise ValueError('k must be greater than 0')
    if n <= 0:
        raise ValueError('n must be greater than 0')

    # Initialize arrays 
    xN = np.zeros(n+1, dtype = np.longdouble)
    xN[0] = x
    C = 2*k - 2
    XCol = np.empty(C, dtype = int) 
    powers = 2.0 ** -np.arange(1, C+1)

    # Pull in our four parsed maps once per call 
    all_cols, aij_defs, top_defs, rec_defs = DMG.generate_parsed_maps(k)
    formulas = {**top_defs, **rec_defs}
    
    # build a flat list of cache keys and index map
    all_keys = list(aij_defs.keys()) + list(formulas.keys())
    key2idx = { key: idx for idx, key in enumerate(all_keys) }
    cache_len = len(all_keys)
    
    # iterate n steps 
    for l in range(n):
        # if we've fallen out of (0,1), stop early
        if x == 0 or x == 1:
            print(f"ERROR - {xN[l-1]} went outside accepted range after {l} iterations")
            return xN[:l]

        # Step 2: convert current x to its k-bit fractional binary a[0..k-1]
        a = np.zeros(k, int)
        z = x
        for i in range(k):
            z *= 2
            if z >= 1:
                a[i] = 1
                z -= 1
            else:
                a[i] = 0
            
    # Compute A_i_j ops
    bit_cache = cache_len*[0]
    for key, (head_key, tail_keys) in aij_defs.items():
        # head_key == (1,i,j)
        _, i, j = head_key
        idx = key2idx[head_key]
        bit_cache[idx] = a[i-1] ^ a[j-1]
        
    # Compute all Nn_i_j ops
    for r in range(4, C):
        for d,i,j in all_cols[r]:
            if d != 1:
                key = (d,i,j)
                head, tails = formulas[key]
                if bit_cache.get(head, 0) == 0:
                    bit_cache[key] = 0    
    
    

                
