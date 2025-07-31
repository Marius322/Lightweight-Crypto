# -*- coding: utf-8 -*-
"""
Generator that holds Numba formulas, used to further optimise digital map

@author: 22391643
"""

import Digital_Map_Generator as DMG
import numpy as np
from numba import njit, prange
from collections import defaultdict
from functools import lru_cache

def build_flat_defs(k, top_defs, rec_defs):
    """
    Build flat_defs dict mapping each operator (including A_i_j) to 
    substitution entries with 'head', 'tails', 'order', and 'chunk' index.

    Chunk 0: all A_i_j operators
    Chunk 1: all Nn_i_j ops in Col(0–7)
    Chunk 2: all Nn_i_j ops in Col(8–11)
    Chunk 3+: all Nn_i_j ops in successive blocks of 2 columns
    """
    # 1) Chunk 0: pure A_i_j
    aij_defs = DMG.generate_aij_dict(k)
    aij_keys = list(aij_defs.keys())

    # 2) Column index for every op
    all_cols  = DMG.generate_all_columns(k)   # {r: [[d,i,j], …]}
    op_to_col = {
        (d,i,j): r
        for r, entries in all_cols.items()
        for (d,i,j) in entries
    }
    total_cols = 2*k - 2  # valid r = 0 … 2*k-3

    # 3) Hard‑coded non‑A chunk sizes: [8, 4, 2, 2, ...]
    sizes = []
    # first block of size 8
    sizes.append(min(8, total_cols))
    used = sizes[0]
    # next block of size 4
    if used < total_cols:
        sizes.append(min(4, total_cols - used))
        used += sizes[-1]
    # then blocks of size 2
    while used < total_cols:
        sizes.append(min(2, total_cols - used))
        used += sizes[-1]

    # 4) Compute column ranges for chunks 1..C
    ranges = []
    start = 0
    for size in sizes:
        end = start + size - 1
        ranges.append((start, end))
        start = end + 1
    # chunk 1 → ranges[0], chunk 2 → ranges[1], etc.

    # 5) Identify non‑A operators
    is_A      = lambda op: isinstance(op, tuple) and op[0] == 1
    non_A_ops = [op for op in rec_defs if not is_A(op)]

    # 6) Assign each non‑A op to its chunk
    chunk_ops = defaultdict(set)
    for op in non_A_ops:
        r = op_to_col.get(op)
        if r is None:
            continue
        for c, (s,e) in enumerate(ranges, start=1):
            if s <= r <= e:
                chunk_ops[c].add(op)
                break

    # 7) Merge top_defs + rec_defs for lookup
    defs     = {**top_defs, **rec_defs}
    defs_get = defs.get

    flat_defs = {}

    # 8) Emit chunk 0 entries
    for op in aij_keys:
        flat_defs[op] = [{
            'head':  op,
            'tails': [],
            'order': 0,
            'chunk': 0
        }]

    # 9) Flatten each chunk in ascending order
    num_chunks = len(ranges)
    for c in range(1, num_chunks + 1):
        # build base set = all A_i_j + all ops in earlier chunks
        base = set(aij_keys)
        for cc in range(1, c):
            base |= chunk_ops[cc]

        # per‑chunk expansion cache
        expansion_cache = {}

        for root in chunk_ops[c]:
            if root in expansion_cache:
                # reuse cached entries list
                flat_defs[root] = expansion_cache[root]
            else:
                # BFS expansion with dedupe
                queue = [(root, 0)]
                seen  = {root}
                idx   = 0
                entries = []

                while idx < len(queue):
                    op, order = queue[idx]
                    idx += 1

                    head, tails = defs_get(op, (None, []))
                    d = op[0]

                    if d > 1 and not tails:
                        # auto‑simplify to zero
                        entries.append({
                            'head':  None,
                            'tails': [],
                            'order': order,
                            'chunk': c,
                            '_zero': True
                        })
                    else:
                        entries.append({
                            'head':  head,
                            'tails': list(tails),
                            'order': order,
                            'chunk': c
                        })
                        # enqueue unseen, non‑A, non‑base children
                        for child in (head, *tails):
                            if (not is_A(child)) and (child not in base) and (child not in seen):
                                seen.add(child)
                                queue.append((child, order + 1))

                # cache & assign
                expansion_cache[root] = entries
                flat_defs[root]       = entries

        # clear cache to free memory before next chunk
        expansion_cache.clear()

    return flat_defs, num_chunks

# cache once per k
_array_cache = {}

def prepare_numba_arrays(k, flat_defs, all_cols, num_chunks):
    """
    Convert flat_defs and all_cols into hand‑rolled NumPy arrays.
    Handles head=None entries by mapping to -1.
    Returns dict of arrays for Numba.
    """
    # 1) List of unique ops and map → ID
    ops = list(flat_defs.keys())
    op_to_idx = {op: idx for idx, op in enumerate(ops)}
    M = len(ops)

    # 2) op_keys: (M,3)
    op_keys = np.empty((M, 3), dtype=np.int32)
    for idx, (d, i, j) in enumerate(ops):
        op_keys[idx, 0] = d
        op_keys[idx, 1] = i
        op_keys[idx, 2] = j

    # 3) Flatten the per‑op entries into a list
    entries = []
    for op, entry_list in flat_defs.items():
        for e in entry_list:
            entries.append((op, e['head'], e['tails'], e['order'], e['chunk']))
    E = len(entries)

    # 4) Allocate arrays
    entry_op  = np.empty(E, dtype=np.int32)
    head_idx  = np.empty(E, dtype=np.int32)
    tail_ptr  = np.empty(E, dtype=np.int32)
    tail_len  = np.empty(E, dtype=np.int32)
    orders    = np.empty(E, dtype=np.int32)
    chunk_idx = np.empty(E, dtype=np.int32)

    tail_list = []
    # 5) Fill arrays, mapping head=None to -1
    for ei, (op, head, tails, order, chunk) in enumerate(entries):
        entry_op[ei]  = op_to_idx[op]
        if head is None:
            head_idx[ei] = -1
        else:
            head_idx[ei] = op_to_idx[head]
        orders[ei]    = order
        chunk_idx[ei] = chunk

        tail_ptr[ei] = len(tail_list)
        tail_len[ei] = len(tails)
        for t in tails:
            tail_list.append(op_to_idx[t])
    tail_idxs = np.array(tail_list, dtype=np.int32)

    # 6) Flatten all_cols into col_ptr, col_len, col_ops
    C = len(all_cols) 
    col_ptr = np.empty(C, dtype=np.int32)
    col_len = np.empty(C, dtype=np.int32)
    col_list = []
    for r in range(C):
        col_ptr[r] = len(col_list)
        col_len[r] = len(all_cols[r])
        for (d, i, j) in all_cols[r]:
            col_list.append(op_to_idx[(d, i, j)])
    col_ops = np.array(col_list, dtype=np.int32)

    # 7) Compute inverse powers vector
    length = 2 * k - 2
    inv_powers = 2.0 ** ( - np.arange(1, length + 1, dtype=np.float64) )
    
    # 8) Compute max order within each chunk
    chunk_max_bfs = np.zeros(num_chunks+1, dtype=np.int32)
    for e in range(E):
        c = chunk_idx[e]
        d = orders[e]
        if c > 0 and d > chunk_max_bfs[c]:
            chunk_max_bfs[c] = d
    
    return {
        'op_keys':   op_keys,
        'entry_op':  entry_op,
        'head_idx':  head_idx,
        'tail_ptr':  tail_ptr,
        'tail_len':  tail_len,
        'tail_idxs': tail_idxs,
        'orders':    orders,
        'chunk_idx': chunk_idx,
        'col_ptr':   col_ptr,
        'col_len':   col_len,
        'col_ops':   col_ops,
        'inv_powers': inv_powers,
        'num_chunks': num_chunks,
        'chunk_max_bfs': chunk_max_bfs
    }

@lru_cache(maxsize=None)
def prepare_k(k: int):
    """
    Generate & cache the NumPy arrays for this k exactly once.
    """
    # 1) Generate parsed maps
    all_cols, _, top_defs, rec_defs = DMG.generate_parsed_maps(k)
    # 2) Flatten definitions into Python dicts
    flat_defs,num_chunks = build_flat_defs(k, top_defs, rec_defs)
    # 3) Convert to hand‑rolled arrays
    return prepare_numba_arrays(k, flat_defs, all_cols, num_chunks)

# Helper Functions

@njit(parallel=True)
def compute_A_bits(a_bits: np.ndarray, op_keys: np.ndarray) -> np.ndarray:
    """
    Compute all A_i_j (depth == 1) bits in parallel.
    - a_bits: shape (k,), dtype=np.uint8 or int8, the binary expansion of x
    - op_keys: shape (M,3), dtype=np.int32, rows are (d,i,j)
    Returns:
      bit_cache: shape (M,), dtype=np.uint8
    """
    M = op_keys.shape[0]
    bit_cache = np.empty(M, dtype=np.uint8)
    for idx in prange(M):
        d = op_keys[idx, 0]
        if d == 1:
            # A_i_j operator
            i = op_keys[idx, 1] - 1
            j = op_keys[idx, 2] - 1
            bit_cache[idx] = a_bits[i] ^ a_bits[j]
        else:
            bit_cache[idx] = 0  # initialize others to zero
    return bit_cache



