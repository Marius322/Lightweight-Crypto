# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 08:57:06 2025

@author: 22391643
"""
import numpy as np
import Digital_Map_Generator as DMG
from concurrent.futures import ThreadPoolExecutor
from typing import List

def parallel_binary_digital_logistic_map(x: float, n: int, k: int) -> str:
    """
    Generate the digital logistic map and return a flat binary string.

    Inputs:
      x = real number between 0 and 1
      n = number of iterations
      k = number of bits to output per iteration

    Output:
      bits_str = string of length (n+1)*k containing the first k fractional bits
                 of x_0, x_1, ..., x_n concatenated with no separators
    """
    # Error checking
    if x <= 0 or x >= 1:
        raise ValueError('x must lie between 0 and 1')
    if k <= 0:
        raise ValueError('k must be greater or equal to 1')
    if n <= 0:
        raise ValueError('n must be greater or equal to 1')

    C = 2 * k - 2
    bits: List[str] = []
    powers = 2.0 ** -np.arange(1, C + 1)

    # Pre-generate parsed maps
    all_cols, aij_defs, top_defs, rec_defs = DMG.generate_parsed_maps(k)
    formulas = {**top_defs, **rec_defs}

    def xor_column(r: int):
        acc = 0
        for d, i, j in all_cols[r]:
            acc ^= bit_cache.get((d, i, j), 0)
        return r, acc

    with ThreadPoolExecutor() as ex:
        for iteration in range(n):
            # Stop if we've left (0,1)
            if x == 0 or x == 1:
                print(f"ERROR - {x} went outside accepted range after {iteration-1} iterations")
                return ''.join(bits)

            # Convert current x to k-bit fractional binary
            a = np.zeros(k, int)
            z = x
            for idx in range(k):
                z *= 2
                if z >= 1:
                    a[idx] = 1
                    z -= 1

            # Initialize cache for A_ij and other operations
            bit_cache: dict = {}
            for key, (head_key, _) in aij_defs.items():
                _, i, j = head_key
                bit_cache[head_key] = a[i-1] ^ a[j-1]

            # Compute remaining operations
            for r in range(4, C):
                for d, i, j in all_cols[r]:
                    if d == 1:
                        continue
                    head, tails = formulas[(d, i, j)]
                    if bit_cache.get(head, 0) == 0:
                        bit = 0
                    else:
                        xorv = 0
                        for t in tails:
                            xorv ^= bit_cache[t]
                        bit = bit_cache[head] & xorv
                    bit_cache[(d, i, j)] = bit

            # Build the column vector of bits
            XCol = np.empty(C, dtype=int)
            for r, acc in ex.map(xor_column, range(C)):
                XCol[C - 1 - r] = acc

            # Append the first k bits to our builder as chars
            bits.extend(str(int(b)) for b in XCol[:k])

            # Update x for next iteration (unless last)
            if iteration < n:
                x = np.dot(XCol, powers)

    return ''.join(bits)

x = 0.98
n = 3125
k = 32
print(parallel_binary_digital_logistic_map(x, n, k))

