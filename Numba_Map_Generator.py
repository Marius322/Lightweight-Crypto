# -*- coding: utf-8 -*-
"""
Generator that creates formulas via Numpy arrays and lists, suitable for Numba

@author: 22391643
"""

import numpy as np
import itertools
from functools import lru_cache

# Helper Functions

def build_full_col(cl, chunk1_array, chunk2_array, chunk3_array):
    '''

    Parameters
    ----------
    cl : index of column being created
    
    chunk1_array : List containing all (1, i, j) tuples
        
    chunk2_array : List containing all (2, i, j) tuples
        
    chunk3_array : List containing all (d, i, j) tuples with d > 2

    Returns
    -------
    full_col: List containing all ops in column 'cl'

    '''
    
    c1 = chunk1_array[cl] if cl < len(chunk1_array) else []
    c2 = chunk2_array[cl] if cl < len(chunk2_array) else []
    c3 = chunk3_array[cl] if cl < len(chunk3_array) else []
    
    full_col = list(c1) + list(c2) + list(c3)
    
    return full_col

def build_op_list(aij, Nnij, k):
    '''

    Input
    ----------
    aij : Numpy array of shape (N, 3) containing all A_i_j ops
    
    Nnij : Numpy array containing all Nn_i_j ops

    Returns
    -------
    unique_ops : List containing all ops 
        
    op_to_idx : List containing index for each op in unique_ops
    '''
    
    # Nnij[r] is a Python list of formulas for that column
    ops = []
    
    # All A_ i j entries
    for row in aij:
        ops.append((int(row[0]), int(row[1]), int(row[2])))

    # Head of every Nn_i_j formula
    C = 2*k - 2
    for cl in range(C):
        formulas = Nnij[cl]
        if not isinstance(formulas, (list, tuple)):
            continue
        for head, tails in formulas:
            d, i, j = head
            ops.append((d+1, i, j)) 

    # Deduplicate & freeze an order
    unique_ops = []
    seen = set()
    for op in ops:
        if op not in seen:
            seen.add(op)
            unique_ops.append(op)

    op_to_idx = {op: idx for idx, op in enumerate(unique_ops)}
    
    return unique_ops, op_to_idx

def build_op_keys(unique_ops):
    '''    

    Input
    ----------
    unique_ops : List containing all ops
    
    Returns
    -------
    op_keys : Numpy array containing the (d, i, j) key for every op

    '''
    
    M = len(unique_ops)
    op_keys = np.empty((M,3), dtype=np.int32)
    
    for idx, (d,i,j) in enumerate(unique_ops):
        op_keys[idx,0] = d
        op_keys[idx,1] = i
        op_keys[idx,2] = j
    
    return op_keys

def build_flattened_formulas(aij, Nnij, Nnij_ops, op_to_idx):
    '''

    Input
    ----------
    aij : Numpy array containing all aij ops
    
    Nnij : Numpy array containing the formula for all Nn_i_j ops
    
    Nnij_ops : Numpy array containing all Nn_i_j ops
    
    op_to_idx : List containing index for all ops

    Returns
    -------
    entry_op : Numpy array containing a reference for all ops - removes tuples
               from main loop in Logistic Map - same  function as op_keys but 
               keep for flexibilty
        
    head_idx : Numpy array containing the head for all ops
        
    tail_strt : Numpy array that tells you where an entries tail begins in 
                tail_idxs
        
    tail_len : Numpy array containing the lengths of all tails - combined with 
               tail_strt to identify the tail for every operator
        
    tail_idxs : Numpy array containing the index for all ops contained in a 
                tail

    '''
    
    entries = []
    
    # All A_i_j formulas
    for row in aij:
        op = (int(row[0]),int(row[1]),int(row[2]))
        entries.append((op, op, []))

    # All the Nn formulas - Problem is here
    C = len(Nnij)
    for cl in range(C):
        for Nnij_ops_keys, (head, tails) in zip( Nnij_ops[cl], Nnij[cl] ):
            entries.append((Nnij_ops_keys, head, tails))


    # Convert heads/tails into indices
    E = len(entries) # Total number of Formulas
    entry_op  = np.empty(E, dtype=np.int32) 
    head_idx  = np.empty(E, dtype=np.int32)
    tail_strt  = np.empty(E, dtype=np.int32)
    tail_len  = np.empty(E, dtype=np.int32)
    tail_list = []

    for ei, (op, head, tails) in enumerate(entries):
        entry_op[ei] = op_to_idx[op]
        head_idx[ei] = op_to_idx.get(head, -1)  
        tail_strt[ei] = len(tail_list)
        tail_len[ei] = len(tails)
        for t in tails:
            tail_list.append(op_to_idx[t])

    tail_idxs = np.array(tail_list, dtype=np.int32)
    
    return entry_op, head_idx, tail_strt, tail_len, tail_idxs

def build_flatten_columns(all_chunks, op_to_idx):
    '''

    Inputs
    ----------
    all_chunks : Numpy array containing all [n, i, j] tuples i each column
        
    op_to_idx : List containing index for all ops

    Returns
    -------
    col_strt : Numpy array containing the ops that start each column
        
    col_len : Numpy array containing the length of each column
        
    col_idxs : Numpy array containing a flattened list of all ops_idx ordered 
               by which col they appear in

    '''
    
    C = len(all_chunks)
    col_strt = np.empty(C, dtype=np.int32)
    col_len = np.empty(C, dtype=np.int32)
    ops_flat = []

    for r in range(C):
        col_strt[r] = len(ops_flat)
        col_len[r] = len(all_chunks[r])
        for (d,i,j) in all_chunks[r]:
            ops_flat.append(op_to_idx[(d,i,j)])

    col_idxs = np.array(ops_flat, dtype=np.int32)
    
    return col_strt, col_len, col_idxs

def build_inv_powers(k): 
    '''

    Input
    ----------
    k : Number of digits in binary number

    Returns
    -------
    inv_powers : Numpy array used in digital map to convert from binary to 
                 float

    '''
    
    C = 2*k - 2
    inv_powers =  2.0 ** ( - np.arange(1, C+1, dtype=np.float64) )
    
    return inv_powers

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
    '''
    
    Input
    ----------
    k : Number of digits in a binary number
    
    chunk2_array : List containing all N2_i_j operators

    Returns
    -------
    chunk3_array : List conataining all Nn_i_j operators, n>2

    '''
    
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

def generate_all_chunks(k):
    '''

    Input
    ----------
    k : Number of digits in binary number
    
    Returns
    -------
    all_chunks : Numpy array containing all Nn_i_j operators 

    '''
    
    chunk1_array = generate_chunk1_arrays(k)
    chunk2_array = generate_chunk2_arrays(k)
    chunk3_array = generate_chunk3_arrays(k, chunk2_array)
    
    C = 2*k - 2 
    all_chunks_array= [[] for _ in range(C)]
    
    for cl in range(C):
        # chunk1[r] could be any sequence
        all_chunks_array[cl] += list(chunk1_array[cl])
        all_chunks_array[cl] += list(chunk2_array[cl])
        all_chunks_array[cl] += list(chunk3_array[cl])
   
    all_chunks = np.array(all_chunks_array, dtype = object)
   
    return all_chunks

# Operator Generators

def generate_aij(k):
    '''

    Input
    ----------
    k : Number of digits in binary number

    Returns
    -------
    aij : Numpy array of shape (N, 3) containing all A_i_j ops

    '''   
    chunk1_array = generate_chunk1_arrays(k)
    
    # For computing aij
    aij = []
    for col in chunk1_array:
        for (d, i, j) in col:
            aij.append((d, i, j))
            
    # convert to a (N,3) array of int32
    aij = np.array(aij, dtype=np.int32)
    # ensure shape is (N,3)
    aij = aij.reshape(-1, 3)
    
    return aij

def generate_top_head_array(k):
    '''

    Inputs
    ----------
    k : Number of digits in binary number

    Returns
    -------
    top_head_array: List containing the formula for all N2_i_j ops that appear
                    first in their respective column

    '''
    
    chunk1_array = generate_chunk1_arrays(k)
    chunk2_array = generate_chunk2_arrays(k)
    chunk3_array = generate_chunk3_arrays(k, chunk2_array)
    
    C = 2*k - 2
    top_head_array = [None] * C 
    top_head_array[0], top_head_array[1] = 0, 0
    top_head_array[2], top_head_array[3] = 0, 0
    
    for cl in range (4, C):
        ops = build_full_col(cl - 1, chunk1_array, chunk2_array, chunk3_array)
        
        head = ops[0]      
        tails = ops[1:]    
        d, i, j = head       
        
        # Formula for Nn_i_j appears in column beforehand
        top_head_array[cl] = [ head, tails ]        
        
    return top_head_array
    
def generate_rec_array(k):
    '''

    Input
    ----------
    k : Number of digits in binary number

    Returns
    -------
    rec_array : List containing formulas for all Nn_i_j operators not defined
                in top_head_array

    '''

    chunk1_array = generate_chunk1_arrays(k)
    chunk2_array = generate_chunk2_arrays(k)
    chunk3_array = generate_chunk3_arrays(k, chunk2_array)

    C = 2*k - 2
    rec_array = [[] for _ in range(C)]
    
    for cl in range(4, C):
        ops = build_full_col(cl - 1, chunk1_array, chunk2_array, chunk3_array)
        # skip the first one (handled by top_head_array)
        recs = []
        
        if cl == 4:
            Range = len(ops)
        else:
            Range = len(ops) - 2
        
        for i in range(1, Range):  
            head  = ops[i]
            tails = ops[i+1:]
            recs.append((head, tails))
        rec_array[cl] = recs              
    
    return rec_array

def generate_Nnij(all_chunks, top_head_array, rec_array, k):
    '''

    Inputs
    ----------
    all_chunks : numpy array containing all columns and their entries
        
    top_head_array : List containing all N2_i_j ops that appear first in their
                     column
        
    rec_array : List containing all Nn_i_j ops that do not appear in 
                top_head_array

    Returns
    -------
    Nnij : Numpy array containing the formulas for all Nn_i_j ops
    
    Nnij_ops : Numpy array containing all Nn_i_j ops

    '''
    
    C = 2*k - 2
    Nnij_array = [None] * C
    Nnij_ops_array = [None] * C

    # walk column by column
    for cl in range(C):
        # get the “top” formula if present
        th = top_head_array[cl]
        if isinstance(th, (list, tuple)):
            # wrap into a single‐element list if it's not already one
            if len(th) == 2 and not isinstance(th[0], list):
                top_list = [ th ]
            else:
                top_list = list(th)
        else:
            top_list = []
            
        # get the recursive formulas
        recs = rec_array[cl] if rec_array[cl] is not None else []

        # concatenate
        Nnij_array[cl] = top_list + recs
        
        # Get operator key for Nn_i_j only
        Nnij_ops_array[cl] = [ key for key in all_chunks[cl] if key[0] != 1 ]

    # now turn into a single numpy array of objects
    Nnij = np.array(Nnij_array, dtype=object)
    Nnij_ops = np.array(Nnij_ops_array, dtype = object)
    
    return Nnij, Nnij_ops

# Map Generator

@lru_cache(maxsize = 1)
def generate_listed_map(k):
    
    all_chunks = generate_all_chunks(k)
    top_head_array = generate_top_head_array(k)
    rec_array = generate_rec_array(k)
    aij = generate_aij(k)
    Nnij, Nnij_ops = generate_Nnij(all_chunks, top_head_array, rec_array, k)
    op_list,op_to_idx = build_op_list(aij, Nnij, k)
    op_keys = build_op_keys(op_list)
    (entry_op, head_idx, 
     tail_strt, tail_len, 
     tail_idxs) = build_flattened_formulas(aij, Nnij, Nnij_ops, op_to_idx)
    col_strt, col_len, col_idxs = build_flatten_columns(all_chunks, op_to_idx)
    inv_powers = build_inv_powers(k)
    
    return (op_keys,
            entry_op,
            head_idx,
            tail_strt, tail_len, tail_idxs,
            col_strt, col_len, col_idxs,
            inv_powers)
 
         