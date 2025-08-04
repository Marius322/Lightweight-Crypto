# -*- coding: utf-8 -*-
"""
Generator that holds Numba formulas, used to further optimise digital map

@author: 22391643
"""

import numpy as np
import itertools

# Column generators

def generate_chunk1_arrays(k):
    '''
    
    Inputs
    ----------
    k : Number of digits in binary number

    Returns
    -------
    chunk1_array : List containing all (1, i, j) entries

    '''
    
    C = 2 * k - 2
    chunk1_array = [None] * C
    chunk1_array[0] = []  # placeholder for empty first column

    for cl in range(1, C):
        col = []
        
        # Case 1 - Col(1) to Col(k-1)
        if 1 <= cl <= k - 1:
            base_i = k - cl
            base_j = k
         
        # Case 2 - Col(k) to Col(C)
        else:
            base_i = 1
            base_j = 2 * k - 1 - cl
        
        for n in range (0, k):
            i = base_i + n
            j = base_j - n
            if i >= j:
                break
            else:
                col.append((1, i, j))
                n += 1 
        
        # Creating full array
        chunk1_array[cl] = col if col else []

    return chunk1_array
  
def generate_chunk2_arrays(k):
    '''

    Inputs
    ----------
    k : Number of digits in a binary number

    Returns
    -------
    chunk2_array : List containing all (2, i, j) tuples
    '''
    
    C = 2 * k - 2
    chunk2_array = [None] * C
    chunk2_array[0], chunk2_array[1] = [], []
    chunk2_array[2], chunk2_array[3] = [], [] # placeholder for empty first column
    
    for cl in range(4, C):
        col = []
        
        # Case 1 - Col(4) to Col(k)
        if cl <= k:
            base_i, base_j = k - cl + 1, k
        
        # Case 2 - Col(k) to Col(C)
        else:
            base_i, base_j = 1, 2*k - cl
            
        n = 0
        for n in range (0, k):
            i = base_i + n
            j = base_j - n
            if i >= j:
                break
            else:
                col.append((2, i, j))
                n += 1 
                
        # Creating full array
        chunk2_array[cl] = col if col else []
        
    return chunk2_array
        
def generate_chunk3_arrays(k, chunk2_array):  
    
    C = 2*k - 2
    chunk3_array = [None]*C
    for i in range(min(7, C)):
        chunk3_array[i] = []
    used_tuples = set()
    col = [[] for _ in range(C)]
    
    for cl in range(7, C):
        
        prev2 = chunk2_array[cl - 1]
        prev3 = col[cl - 1]
        target = len(prev2) + len(prev3) - 2
        
        if target <= 0:
            col[cl] = []
            chunk3_array[cl] = []
            continue
        
        new_entries = []
        for d, i, j in itertools.chain(prev2, prev3):
            key = (d + 1, i, j)
            if j > i and key not in used_tuples:
                used_tuples.add(key)
                new_entries.append(key)
                if len(new_entries) == target:
                    break
                    
        # Creating full array
        col[cl] = new_entries
        chunk3_array[cl] = new_entries

    return chunk3_array    

        
            
            
        
    
    
    

