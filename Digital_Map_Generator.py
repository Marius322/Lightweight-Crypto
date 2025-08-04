# -*- coding: utf-8 -*-
"""
Creates the formulas used for digital map
Automatically includes the left-shift-by-two (drops two columns on left)
Gives 2k-2 length output

@author: 22391643
"""

from concurrent.futures import ThreadPoolExecutor
import math
from functools import lru_cache

# Column generators 

def generate_col1_columns(k):
    '''
    
    Inputs
    ----------
    k : Number of digits in binary number

    Returns
    -------
    col1_dict : A dict containing all A_i_j ops assigned to their columns

    '''
    col1_dict = {}
    for cl in range(2 * k - 2):
        col1_entries = []
        
        # Case 1: A_i_j does not exist for Col(0)
        if cl == 0:
            col1_dict[cl] = col1_entries
            continue
        
        # Case 2: A_i_j from Col(1) to Col(k-1)
        if 1 <= cl <= k - 1:
            base_i, base_j = k - cl, k
            n = 0

        # Case 3: A_i_j from Col(k) to Col(2k - 3)
        else:  
            base_i, base_j = 1, 2 * k - 1 - cl
            n = 0
            
        while True:
            i = base_i + n
            j = base_j - n
            if j > i:
                col1_entries.append([1, i, j])
                n += 1
            else:
                break
        
        col1_dict[cl] = col1_entries
    return col1_dict

def generate_col2_columns(k):
    '''
    
    Inputs
    ----------
    k : Number of digits in binary number

    Returns
    -------
    col2_dict : a dict containing all N2_i_j ops assigned to their columns

    '''
    col2_dict = {}
    for r in range(2 * k - 2):
        col2_entries = []
        
        # Case 1: N2_i_j does not exist for Col(0) to Col(3)
        if r < 4:
            col2_dict[r] = col2_entries
            continue
        
        # Case 2: N2_i_j from Col(4) to Col(k)
        if r <= k:
            base_i, base_j = k - r + 1, k
        
        # Case 3: N2_i_j from Col(k + 1) to Col(2k-2)
        else:
            base_i, base_j = 1, 2 * k - r
        n = 0
        
        while True:
            i = base_i + n
            j = base_j - n
            if j > i:
                col2_entries.append([2, i, j])
                n += 1
            else:
                break
        
        col2_dict[r] = col2_entries
    return col2_dict

def generate_col3_columns(k, col2_dict):
    '''
    

    Inputs
    ----------
    k : Number of digits in a binary number
    col2_dict : The dict containing all N2_i_j ops assigned to their column

    Returns
    -------
    col3_dict : A dict containing all Nn_i_j ops (n>2) assigned to their 
                column

    '''
    col3_dict = {}
    used_keys = set()
    
    for r in range(7, 2 * k-2):
        prev2 = col2_dict.get(r - 1, [])
        prev3 = col3_dict.get(r - 1, [])
        target = len(prev2) + len(prev3) - 2
        
        if target <= 0:
            col3_dict[r] = []
            continue
        new_entries = []
        
        for d, i, j in prev2:
            key = (d + 1, i, j)
            if j > i and key not in used_keys:
                used_keys.add(key)
                new_entries.append([*key])
            if len(new_entries) == target:
                break
        
        if len(new_entries) < target:
            for d, i, j in prev3:
                key = (d + 1, i, j)
                if j > i and key not in used_keys:
                    used_keys.add(key)
                    new_entries.append([*key])
                if len(new_entries) == target:
                    break
        
        col3_dict[r] = new_entries
    return col3_dict

def generate_all_columns(k):
    '''
    

    Inputs
    ----------
    k : number of digits in binary number

    Returns
    -------
    all_cols : A dict containing all columns and ops for digital logistic map

    '''
    col1 = generate_col1_columns(k)
    col2 = generate_col2_columns(k)
    col3 = generate_col3_columns(k, col2)
    all_cols = {r: col1.get(r, []) + col2.get(r, []) + col3.get(r, [])
            for r in range(2 * k - 2)} 
    return all_cols

# Operator Generators

def generate_aij_dict(k):
    '''
    

    Inputs
    ----------
    k : Number of digits in binary number

    Returns
    -------
    aij : A python dict containing all A_i_j ops

    '''
    col1 = generate_col1_columns(k)
    aij = {}
    for entries in col1.values():
        for _, i, j in entries:
            key = (1,i,j)
            aij[key] = ( (1,i,j), [] )   # head is itself, no tail
    return aij

def generate_top_head_dict(k):
    '''
    
    Inputs
    ----------
    k : Number of digits in binary number
    
    Returns
    -------
    top_head : A dict containing the formulas for all N2_i_j ops that appear
             first in their column

    '''
    col1 = generate_col1_columns(k)
    col2 = generate_col2_columns(k)
    col3 = generate_col3_columns(k, col2)
    def full_col(r):
        return col1.get(r,[])+col2.get(r,[])+col3.get(r,[])
    top_head = {}
    # Case 1
    for r in range(4, k+1):
        i,j = k-r+1, k
        col = full_col(k-i)
        if not col: continue
        head, tail = col[0], col[1:]
        head_key = (1, head[1], head[2])
        tail_keys = [(d,i2,j2) for d,i2,j2 in tail]
        top_head[(2,i,j)] = ( head_key, tail_keys )
    # Case 2
    for r in range(k+1, 2*k-2):
        i,j = 1, 2*k-r
        col = full_col(2*k-j-1)
        if not col: continue
        head, tail = col[0], col[1:]
        head_key = (1, head[1], head[2])
        tail_keys = [(d,i2,j2) for d,i2,j2 in tail]
        top_head[(2,i,j)] = ( head_key, tail_keys )
    return top_head

def generate_recursive_column_formulas(k):
    '''

    Inputs
    ----------
    k : Number of digits in binary number

    Returns
    -------
    rec : A dict containing the formulas for all Nn_i_j ops not in top_head

    '''

    cols = generate_all_columns(k)           
    top_dict = generate_top_head_dict(k)       
    aij_dict = generate_aij_dict(k)
    rec = { **aij_dict, **top_dict }         

    def build_chunk(r_start, r_end):
        partial = {}
        for r in range(r_start, r_end):
            col = cols[r]
            # 1) find the first depth>1 element that’s in top_parsed
            head_idx = None
            head_key = None
            for idx, (d,i,j) in enumerate(col):
                key = (d,i,j)
                if d > 1 and key in top_dict:
                    head_idx = idx
                    head_key = key
                    break
            if head_idx is None:
                continue

            # 2) pull its parsed tail_keys
            _, tail_keys = top_dict[head_key]
            if not tail_keys:
                # nothing to expand
                continue

            # 3) each further entry in this column yields one new formula
            for off, (d2, i2, j2) in enumerate(col[head_idx+1:], start=1):
                child_key = (d2, i2, j2)
                if off <= len(tail_keys):
                    # use the off‑th head from tail_keys, and the rest as new tail
                    new_head = tail_keys[off-1]
                    new_tail = tail_keys[off:]
                    partial[child_key] = (new_head, new_tail)
                else:
                    # ran out of terms ⇒ literal zero
                    partial[child_key] = (child_key, [])

        return partial

    # --- split work into chunks ---
    total       = 2*k - 2
    start, end  = 4, total
    num_workers = min(32, max(1, end - start))
    chunk_size  = math.ceil((end - start)/num_workers)
    ranges = [
        (start + i*chunk_size, min(start + (i+1)*chunk_size, end))
        for i in range(num_workers)
    ]

    # --- run in parallel and merge results ---
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(build_chunk, rs, re) for rs, re in ranges]
        for fut in futures:
            rec.update(fut.result())

    return rec

# Map Generator

@lru_cache(maxsize = 1)
def generate_parsed_maps(k):
    '''

    Input
    ----------
    k : Number of digits in binary number
        
    Returns
    -------
    all_cols : A dict containing all columns and ops for digital logistic map
    aij : A python dict containing all A_i_j ops
    top_head : A dict containing the formulas for all N2_i_j ops that appear
             first in their column
    rec : A dict containing the formulas for all Nn_i_j ops not in top_head

    '''
    all_cols = generate_all_columns(k)
    aij = generate_aij_dict(k)
    top_head = generate_top_head_dict(k)
    rec = generate_recursive_column_formulas(k)
    return (
      all_cols,            
      aij,               
      top_head,        
      rec    
    )

# Helper Functions

def label_to_key(label):
    '''    

    Input
    ----------
    label : Form A_i_j or N2_i_j

    Returns
    -------
    key : Form (d, i, j)

    '''
    if isinstance(label, tuple):
        return label

    head, si, sj = label.split("_")
    if head == "A":
        return (1, int(si), int(sj))
    # head is "Nn"
    d = int(head[1:])
    return (d, int(si), int(sj))

c6 = generate_col2_columns(6)