# -*- coding: utf-8 -*-
"""
Binary version of numba logistic map

@author: 22391643
"""
import numpy as np
from Numba_Logistic_Map import digital_logistic_map_numba
import hashlib
from typing import Tuple

def float_to_binary_frac(x: float, k: int):
    """
    Converts float numbers to binary astrings
    """
    if not (0 < x < 1):
        raise ValueError(f"x must be >0 and <1, got {x}")
    bits = []
    for _ in range(k):
        x *= 2
        if x >= 1.0:
            bits.append('1')
            x -= 1.0
        else:
            bits.append('0')
        if x == 0.0:
            break
    return ''.join(bits)

def hash_extractor(raw_bits: str,
                   in_block_bits: int = 512,
                   out_block_bits: int = 256):
    """
    Whitening function that takes in 512 bits and outputs 256 bits - used to 
    off-set bias from converting binary numbers to float
    """
    if in_block_bits % 8 != 0 or out_block_bits % 8 != 0:
        raise ValueError("Block sizes must be multiples of 8")
    out = []
    for i in range(0, len(raw_bits) - in_block_bits + 1, in_block_bits):
        chunk = raw_bits[i : i + in_block_bits]
        data = int(chunk, 2).to_bytes(in_block_bits // 8, 'big')
        digest = hashlib.sha256(data).digest()
        bits = ''.join(f"{b:08b}" for b in digest)
        out.append(bits[:out_block_bits])
    return ''.join(out)

def array_to_whitened_bits(arr: np.ndarray,
                           k: int,
                           filename_base: str,
                           in_block_bits: int = 512,
                           out_block_bits: int = 256
                          ) -> Tuple[str, str]:
    """
    1) Build raw bit‐stream from floats.
    2) Whiten it via SHA-256 extractor.
    3) Write the final bit‐stream to filename_base.seed.

    Args:
        arr (np.ndarray): 1D array of floats in (0,1).
        k (int): Max bits per float.
        filename_base (str): Base name for output file ('.seed' will be added).
        in_block_bits (int): Input block size for hashing.
        out_block_bits (int): Output bits per hash block.

    Returns:
        Tuple[str, str]: (whitened_bit_str, output_filename)
    """
    # 1) raw stream
    raw_chunks = [float_to_binary_frac(x, k) for x in arr]
    raw_bits = ''.join(raw_chunks)

    # 2) whitening
    whitened = hash_extractor(raw_bits, in_block_bits, out_block_bits)

    # 3) write to .seed file
    fname = filename_base if filename_base.lower().endswith('.seed') else f"{filename_base}.seed"
    with open(fname, 'w') as f:
        f.write(whitened)
        
    # 4) length of bits
    bit_length = len(whitened)

    return whitened, bit_length

x = 0.0001
n = 50000
k = 64

xN = digital_logistic_map_numba(x, n, k)
_, total_bits = array_to_whitened_bits(xN, k, f"{x}")