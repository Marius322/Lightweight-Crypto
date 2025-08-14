# -*- coding: utf-8 -*-
"""
Digital Logistic Map that uses numpy arrays and works with numba

@author: 22391643
"""

import Numba_Map_Generator as NMG
import numpy as np
import math
from numba import njit

@njit
def unwrapped_digital_logistic_map_numba(x, n, k, 
                                   op_keys,
                                   entry_op,
                                   head_idx,
                                   tail_strt, tail_len, tail_idxs,
                                   col_strt, col_len, col_idxs,
                                   C, M, a0):
    '''
    Digitised version of logistic map that is numba friendly
    
    Parameters
    ----------
    x : starting number 
    
    n : number of iterations
    
    k : number of digits in the binary representation of x
    -------

    Returns
    -------
    xN : sequence of float numbers determined by the digitised version of the 
         logistic map
         
    A : sequence of bits determined by the digitised version of the logistic  
        map
    '''
    
    # Initialising arrays
    xN = np.zeros(n+1, dtype = np.float64)
    xN[0] = x
    
    a = a0
    depths = op_keys[:, 0]
    i_idx  = op_keys[:, 1] - 1
    j_idx  = op_keys[:, 2] - 1
    
    bit_cache = np.empty(M, dtype = np.uint8)
    XCol = np.empty(C, dtype = np.uint8)
    
    # Iterate n steps 
    for it in range(n):
        
        # Initialise final values for all ops and XCol
        bit_cache.fill(0)
        XCol.fill(0)
        
        # Compute A_i_j ops        
        bits = a[i_idx] ^ a[j_idx]
        bit_cache[:] = bits * (depths == 1)
        
        # Compute Nn_i_j ops
        for cl in range(C):
            start = col_strt[cl] 
            length = col_len[cl]
            for ee in range(start, start + length): # ee = each entry
                op_id = entry_op[ee]
                
                # Avoid touching A_i_j ops
                if op_keys[op_id, 0] == 1:
                    continue
               
                #Safe Prune
                h = head_idx[ee] 
                if h < 0 or bit_cache[h] == 0:
                   bit_cache[op_id] = 0 
                   continue
               
                # XOR all tail entries
                t_strt = tail_strt[ee]
                t_len = tail_len[ee]
                xor = np.uint8(0)
                for te in range(t_strt, t_strt + t_len): # te = tail entry
                    xor ^= bit_cache[tail_idxs[te]]
                
                # Head AND Tail
                bit_cache[op_id] = bit_cache[h] & xor            
            
            # XOR bit_cache[op_id] along along each column to create XCol
            XCol_entry = np.uint8(0)
            
            for idx in range(start, start + length):
                XCol_entry ^= bit_cache[col_idxs[idx]]
            XCol[C - 1 -cl] = XCol_entry  
        
        # Convert back to real
        x = np.float64(0.0)
        for j in range(C - 1, -1, -1): 
            # Horner Method
            x = 0.5 * (x + np.float64(XCol[j]))
        
        xN[it + 1] = x 
        
        # Set a to be XCol for next iteration
        a[:] = XCol[:k]
        
        # If we've fallen out of (0,1), stop early
        if x <= 0 or x >= 1:
            print(f"ERROR - {xN[it-1]} went outside accepted range after {it} iterations")
            return xN[:it]
                
    return xN

def digital_logistic_map_numba(x: float, n: int, k: int):
    """
    Wrapper function that calls the actual digital logistic map with only x, n
    and k as inputs
    """
   
    # Catching Errors
    if x <= 0 or x >= 1:
        raise ValueError('x must lie between 0 and 1')
        return
    if k <= 0:
        raise ValueError('k must be greater than 0')
        return
    if n <= 0:
        raise ValueError('n must be greater than 1')
        return
    
    # Convert input x to its k-bit fractional binary a[0..k-1]
    a0 = np.zeros(k, dtype = np.uint8)
    acc = int(math.floor(math.ldexp(float(x), k)))  # acc = ⌊x·2^k⌋
    
    for j in range(k):
        # Most-significant fractional bit first
        a0[j] = (acc >> (k - 1 - j)) & 1
        
    # pull in for this k
    (op_keys,
     entry_op,
     head_idx,
     tail_strt, tail_len, tail_idxs,
     col_strt, col_len, col_idxs,
     C, M) = NMG.generate_listed_map(k)
    
    xN = unwrapped_digital_logistic_map_numba(
        x, n, k,
        op_keys,
        entry_op,
        head_idx,
        tail_strt, tail_len, tail_idxs,
        col_strt, col_len, col_idxs,
        C, M, a0
    )
    
    return xN

x = 0.314
n = 19
k = 64

xN = digital_logistic_map_numba(x, n, k)
