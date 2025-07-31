# -*- coding: utf-8 -*-
"""
This is a digital map that is optimised (?) using Numba 

@author: 22391643
"""

import numpy as np
import Numba_Map_Generator_V1 as NMG
from numba import njit, prange

@njit(parallel = True)
def _numba_digital_logistic_map(x: float, n: int, k: int, 
op_keys,
entry_op,
head_idx,
tail_ptr,
tail_len,
tail_idxs,
orders,
col_ptr,
col_len,
inv_powers,
num_chunks,
col_ops):
    
    ''' 
    Inputs:  
      x = real number between 0 and 1
      n = number of iterations
      k = number of digits
      prune_layers = number of layers to pre-load for pruning

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
    
    # Step 1: initialize array
    xN = np.empty(n+1, dtype = np.float64)
    xN[0] = x
    E = entry_op.shape[0]
    C = col_ptr.shape[0]
    
    # Precompute maximum operator depth
    max_depth = 0
    for i in range(op_keys.shape[0]):
        if op_keys[i, 0] > max_depth:
            max_depth = op_keys[i, 0]
            
    # iterate n steps 
    for l in range(n):
        # if we've fallen out of (0,1), stop early
        if x == 0 or x == 1:
            print(f"ERROR - {xN[l-1]} went outside accepted range after {l} iterations")
            return xN[:l]

        # Step 2: convert current x to its k-bit fractional binary a[0..k-1]
        a = np.empty(k, dtype = np.uint8)
        z = x
        for i in range(k):
            z *= 2
            if z >= 1:
                a[i] = 1
                z -= 1
            else:
                a[i] = 0
                
        # Step 3: Compute operators 
        
        # Creating array of A_i_j values
        # Compute A_i_j operators
        M = op_keys.shape[0] # represents an array containing all operators/keys
        bit_cache = np.empty(M, dtype=np.uint8)
        for idx in prange(M):
            d = op_keys[idx, 0]
            if d == 1:
                # A_i_j operator
                i = op_keys[idx, 1] - 1
                j = op_keys[idx, 2] - 1
                bit_cache[idx] = a[i] ^ a[j]
            else:
                bit_cache[idx] = 0  # initialize others to zero
        
        # Computing all other operators
                
        for depth in range(2, max_depth+1):
            for e in prange(E):
                # only root entries
                if orders[e] != 0:
                    continue
                op_id = entry_op[e]
                if op_keys[op_id, 0] != depth:
                    continue
                # safe-prune
                h = head_idx[e]
                if h < 0 or bit_cache[h] == 0:
                    bit_cache[op_id] = 0
                else:
                    # compute XOR of tails & AND with head
                    ptr = tail_ptr[e]
                    L   = tail_len[e]
                    xorv = np.uint8(0)
                    for j in range(L):
                        xorv ^= bit_cache[tail_idxs[ptr + j]]
                    bit_cache[op_id] = bit_cache[h] & xorv
            
        # Step 4: Create XCol array by XORing all column terms
        C    = col_ptr.shape[0]
        XCol = np.zeros(C, dtype=np.uint8)
        for r in range(C):
            ptr = col_ptr[r]
            length = col_len[r]
            acc = np.uint8(0)
            for j in range(length):
                acc ^= bit_cache[ col_ops[ptr + j] ]
            XCol[C - 1 - r] = acc
        
        # Step 5: extend state by appending a bits 
        a_ext = np.zeros(2*k-2, dtype = np.uint8)
        a_ext[k-2:] = a

        # Step 6: XOR the two 2k‑vectors 
        XCol ^= a_ext

        # Step 7: convert back to real and store 
        s = 0.0
        for j in range(C):
        # inv_powers[j] is already float64
            s += XCol[j] * inv_powers[j]
        x = s
        xN[l+1] = x
     
    print(xN)    
    return  

def Numba_digital_logistic_map(x: float, n: int, k: int):

    # Get arrays (cached)
    arrays = NMG.prepare_k(k)

    return _numba_digital_logistic_map(
        x, n, k,
        arrays['op_keys'],
        arrays['entry_op'],
        arrays['head_idx'],
        arrays['tail_ptr'],
        arrays['tail_len'],
        arrays['tail_idxs'],
        arrays['orders'],
        arrays['col_ptr'],
        arrays['col_len'],
        arrays['inv_powers'],
        arrays['num_chunks'],
        arrays['col_ops']
    )
        
Numba_digital_logistic_map(0.314, 100, 64)


    
       
        

        

    
    
    
 
        

    


