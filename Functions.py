# -*- coding: utf-8 -*-
"""
Gives Logistic Map

@author: 22391643
"""
import numpy as np
from numba import njit, prange
import Numba_Map_Generator as NMG
import math

def Logistic_map(x, n):
    
    #catching errors
    if x<=0 or x>=1:
        print('Error')
    else:
        X=np.zeros(n+1)
        X[0]=x
        for j in range(n):
             x=4.0*x*(1.0-x)
             X[j+1]=x
             
        return X

@njit(parallel = True)
def transform_logistic_map(xN, T, d):
    
    def inverse_transform(x):
        
        y = (1/np.pi)*(np.arcsin(2*x - 1))
        return y
    
    xN = xN[int(T):]
    yN = np.zeros(len(xN), dtype = np.float64)
    
    for i in prange (len(xN)):
        
        #Catching rounding errors
        t = np.float64(2.0)*xN[i] - np.float64(1.0)
        if t > 1.0: t = 1.0
        if t < -1.0: t = -1.0
        
        yN[i] = inverse_transform(xN[i]) + np.float64(0.5)
        
    yN = yN[::d]
    
    return yN


@njit
def unwrapped_digital_logistic_map_numba_bits(x, n, k, 
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
    a = a0.copy()
    A  = np.empty((n+1, k), dtype=np.uint8)
    A[0, :] = a[:] 
    
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
        
        # Set a to be XCol for next iteration
        a[:] = XCol[:k]
        
        # Collect Outputs
        A[it + 1, :] = a[:]
        
        # If we've fallen out of (0,1), stop early
        if x <= 0 or x >= 1:
            print(f"ERROR - {A[it-1,:]} went outside accepted range after {it} iterations")
            return A[:it,:]
                
    return A

def digital_logistic_map_numba_bits(x: float, n: int, k: int):
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
    
    A = unwrapped_digital_logistic_map_numba_bits(
        x, n, k,
        op_keys,
        entry_op,
        head_idx,
        tail_strt, tail_len, tail_idxs,
        col_strt, col_len, col_idxs,
        C, M, a0
    )
    
    return A

def write_bits_seed(bits, x):
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
    
    fname = f"{x}.seed"
    with open(fname, 'w', encoding='utf-8') as f:
        f.write(bit_str)
    
    return fname, len(bit_str)



