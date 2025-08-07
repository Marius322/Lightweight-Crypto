# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 10:15:48 2025

@author: 22391643
"""

import numpy as np
import Digital_Map_Generator as MG 

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

def digital_logistic_map(x: float, n: int, k: int):
    '''
    Inputs:
      x = real number between 0 and 1
      n = number of iterations
      k = number of digits 

    Outputs:
      xN = logistic sequence of real numbers
    '''
    # --- sanity checks ---
    if x <= 0 or x >= 1:
        print('ERROR - x must lie between 0 and 1')
        return
    if k <= 0:
        print('ERROR - k must be greater or equal to 1')
        return
    if n <= 0:
        print('ERROR - n must be greater or equal to 1')
        return

    # --- Step 1: initialize output array ---
    xN = np.zeros(n+1)
    xN[0] = x

    # --- pull in our four parsed maps once per call ---
    all_cols, aij_defs, top_defs, rec_defs = MG.generate_parsed_maps(k)

    # --- iterate n steps ---
    for l in range(n):
        # if we've fallen out of (0,1), stop early
        if x == 0 or x == 1:
            print(f"ERROR - {xN[l-1]} went outside accepted range after {l} iterations")
            return xN[:l]

        # --- Step 2: convert current x to its k-bit fractional binary a[0..k-1] ---
        a = np.zeros(k, int)
        z = x
        for i in range(k):
            z *= 2
            if z >= 1:
                a[i] = 1
                z -= 1
            else:
                a[i] = 0

        # --- Step 3: seed bit_cache from all A_i_j definitions ---
        bit_cache = {}
        # aij_defs maps (1,i,j) -> (head_key, []) and head_key==(1,i,j)
        for key, (head_key, tail_keys) in aij_defs.items():
            # head_key == (1,i,j)
            _, i, j = head_key
            bit_cache[head_key] = a[i-1] ^ a[j-1]

        # --- Step 4: compute all deeper N‑operators ---
        # merge top_defs and rec_defs
        
        pending = {**top_defs, **rec_defs}

        while pending:
            progress = False

            for lbl, (head_key, tail_keys) in list(pending.items()):
                d, i, j = MG.label_to_key(lbl)

                # only proceed when *both* head_key *and* all tail_keys are in bit_cache
                if head_key not in bit_cache or not all(tk in bit_cache for tk in tail_keys):
                    continue

                # now safe
                bit = bit_cache[head_key]

                # compute this bit:
                
                # - If tail_keys = 0, bit = 0
                if not tail_keys and d > 1:
                    bit_cache[(d, i, j)] = 0
                else:    
                    
                    # - start from the head bit                
                    bit = bit_cache[head_key]
                
                    # - if there are any tail_keys, XOR them together and AND with head
                    if tail_keys:
                        xor_tail = 0
                        for tk in tail_keys:
                            xor_tail ^= bit_cache[tk]
                        bit &= xor_tail

                    bit_cache[(d, i, j)] = bit

                    # prune: if this bit==0, any deeper same-(i,j) ops must also be zero
                    if bit == 0:
                        for other, (_, other_tails) in list(pending.items()):
                            e, ii, jj = MG.label_to_key(other)
                            if (ii, jj) == (i, j) and e > d:
                                bit_cache[(e, ii, jj)] = 0
                                pending.pop(other)

                # done with this label
                pending.pop(lbl, None)
                progress = True

            if not progress:
                raise RuntimeError(
                    "Stuck on resolving recursive formulas:\n"
                    + "\n".join(str(MG.label_to_key(l)) for l in pending)
                )
                
        # --- Step 5: build the 2k‑vector XCol by XOR‑summing each column of all_cols ---
        XCol = np.zeros(2*k-2, dtype=int)
        for r, entries in all_cols.items():
            acc = 0
            for (d,i,j) in entries:
                acc ^= bit_cache.get((d,i,j), 0)
            XCol[2*k - 3 - r] = acc

        # --- Step 6: extend state by appending our a bits ---
        a_ext = np.zeros(2*k-2, dtype=int)
        a_ext[k-2:] = a

        # --- Step 7: XOR the two 2k‑vectors ---
        XCol ^= a_ext

        # --- Step 9: convert back to real and store ---
        x = float(np.dot(XCol, 2.0 ** -np.arange(1, len(XCol)+1)))
        xN[l+1] = x

    return xN
  
def logistic_map_binary(seed8: str, n: int) -> str:
    """
    seed8: 8‑char string of '0'/'1'
    n:     number of iterations
    Returns a single string of n concatenated 14‑bit blocks.
    """
    # — Input checks —
    if len(seed8) != 8 or any(c not in '01' for c in seed8):
        raise ValueError("seed8 must be exactly 8 characters of '0' or '1'")
    if n <= 0:
        raise ValueError("n must be a positive integer")
    
    result_blocks = []
    current_seed = seed8
    
    for _ in range(n):
        # unpack the 8 seed bits
        a1, a2, a3, a4, a5, a6, a7, a8 = map(int, current_seed)
        
        # 1st Order XORs
        a18 = a1 ^ a8
        a28 = a2 ^ a8
        a38 = a3 ^ a8
        a48 = a4 ^ a8
        a58 = a5^ a8
        a68 = a6 ^ a8
        a78 = a7 ^ a8
            
        a17 = a1 ^ a7
        a27 = a2 ^ a7
        a37 = a3 ^ a7
        a47 = a4 ^ a7
        a57 = a5 ^ a7
        a67 = a6 ^ a7
            
        a16 = a1 ^ a6
        a26 = a2 ^ a6
        a36 = a3 ^ a6
        a46 = a4 ^ a6
        a56 = a5 ^ a6
            
        a15 = a1 ^ a5
        a25 = a2 ^ a5
        a35 = a3 ^ a5
        a45 = a4 ^ a5
            
        a14 = a1 ^ a4
        a24 = a2 ^ a4
        a34 = a3 ^ a4
            
        a13 = a1 ^ a3
        a23 = a2 ^ a3
        a12 = a1 ^ a2
            
        # 1st/2nd Column ANDs
        b58 = a58 & a67
        b48 = a48 & (a57 ^ b58)
        b57 = a57 & b58
            
        # 3rd Column ANDs
        b38 = a38 & (a47 ^ a56 ^ b48 ^ b57)
        b47 = a47 & (a56 ^ b48 ^ b57)
        b56 = a56 & (b48 ^ b57)
            
        # 4th Column ANDs
        b28 = a27 & (a37 ^ a46 ^ b38 ^ b47 ^ b56)
        b37 = a37 & (a46 ^ b38 ^ b47 ^ b56)
        b46 = a46 & (b38 ^ b47 ^ b56)
        c38 = b38 & b56
            
        # 5th Column ANDs
        b18 = a18 & (a27 ^ a36 ^ a45 ^ b28 ^ b37 ^ b46 ^ c38)
        b27 = a27 & (a36 ^ a45 ^ b28 ^ b37 ^ b46 ^ c38)
        b36 = a36 & (a45 ^ b28 ^ b37 ^ b46 ^ c38)
        b45 = a45 & (b28 ^ b37 ^ b46 ^ c38)
        c28 = b28 & (b37 ^ b46 ^ c38)
        c37 = b37 & c38
            
        # 6th Column ANDs
        b17 = a17 & (a26 ^ a35 ^ b18 ^ b27 ^ b36 ^ b45 ^ c28 ^ c37)
        b26 = a26 & (a35 ^ b18 ^ b27 ^ b36 ^ b45 ^ c28 ^ c37)
        b35 = a35 & (b18 ^ b27 ^ b36 ^ b45 ^ c28 ^ c37)
        c18 = b18 & (b27 ^ b36 ^ b45 ^ c28 ^ c37)
        c27 = b27 & (b36 ^ b45 ^ c28 ^ c37)
        c36 = b36 & (c28 ^ c37)        
        c45 = b45 & c37
            
        # 7th Column ANDs
        b16 = a16 & (a25 ^ a34 ^ b17 ^ b26 ^ b35 ^ c18 ^ c27 ^ c36 ^ c45)
        b25 = a25 & (a34 ^ b17 ^ b26 ^ b35 ^ c18 ^ c27 ^ c36 ^ c45)  
        b34 = a34 & (b17 ^ b26 ^ b35 ^ c18 ^ c27 ^ c36 ^ c45)
        c17 = b17 & (b26 ^ b35 ^ c18 ^ c27 ^ c36 ^ c45)
        c26 = b26 & (b35 ^ c18 ^ c27 ^ c36 ^ c45)
        c35 = b35 & (c18 ^ c27 ^ c36 ^ c45)
        d18 = c18 & (c36 ^ c45)
        d27 = c27 & c45
            
        # 8th Column ANDs
        b15 = a15 & (a24 ^ b16 ^ b25 ^ b34 ^ c17 ^ c26 ^ c35 ^ d18 ^ d27)
        b24 = a24 & (b16 ^ b25 ^ b34 ^ c17 ^ c26 ^ c35 ^ d18 ^ d27)
        c16 = b16 & (b25 ^ b34 ^ c17 ^ c26 ^ c35 ^ d18 ^ d27)
        c25 = b25 & (b34 ^ c17 ^ c26 ^ c35 ^ d18 ^ d27)
        c34 = b34 & (c17 ^ c26 ^ c35 ^ d18 ^ d27)
        d17 = c17 & (c26 ^ c35 ^ d18 ^ d27)
        d26 = c26 & (d18 ^ d27)
        d35 = c35 & d27
            
        # 9th Column ANDs
        b14 = a14 & (a23 ^ b15 ^ b24 ^ c16 ^ c25 ^ c34 ^ d17 ^ d26 ^ d35)
        b23 = a23 & (b15 ^ b24 ^ c16 ^ c25 ^ c34 ^ d17 ^ d26 ^ d35)
        c15 = b15 & (b24 ^ c16 ^ c25 ^ c34 ^ d17 ^ d26 ^ d35)
        c24 = b24 & (c16 ^ c25 ^ c34 ^ d17 ^ d26 ^ d35)
        d16 = c16 & (c25 ^ c34 ^ d17 ^ d26 ^ d35)
        d25 = c25 & (c34 ^ d17 ^ d26 ^ d35)
        d34 = c34 & (d26 ^ d35)
        e17 = d17 & d35

        # 10th Column ANDs
        b13 = a13 & (b14 ^ b23 ^ c15 ^ c24 ^ d16 ^ d25 ^ d34 ^ e17)
        c14 = b14 & (b23 ^ c15 ^ c24 ^ d16 ^ d25 ^ d34 ^ e17)
        c23 = b23 & (c15 ^ c24 ^ d16 ^ d25 ^ d34 ^ e17)
        d15 = c15 & (c24 ^ d16 ^ d25 ^ d34 ^ e17)
        d24 = c24 & (d16 ^ d25 ^ d34 ^ e17)
        e16 = d16 & (d34 ^ e17)
        e25 = d25 & e17

        # Matrix construction (product of x with its' 1 compliment)
        M = np.array([
            [0  , 0  , 0  , 0  , 0  , 0  , a18, a28, a38, a48, a58, a68, a78, 0],
            [0  , 0  , 0  , 0  , 0  , a17, a27, a37, a47, a57, a67, 0  , 0  , 0],
            [0  , 0  , 0  , 0  , a16, a26, a36, a46, a56, 0  , 0  , 0  , 0  , 0],
            [0  , 0  , 0  , a15, a25, a35, a45, 0  , 0  , 0  , 0  , 0  , 0  , 0],
            [0  , 0  , a14, a24, a34, 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0],
            [0  , a13, a23, 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0],
            [a12, 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0],
            [b13, b14, b15, b16, b17, b18, 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0],
            [0  , b23, b24, b25, b26, b27, b28, 0  , 0  , 0  , 0  , 0  , 0  , 0],
            [0  , 0  , 0  , b34, b35, b36, b37, b38, 0  , 0  , 0  , 0  , 0  , 0],
            [0  , 0  , 0  , 0  , 0  , b45, b46, b47, b48, 0  , 0  , 0  , 0  , 0],
            [0  , 0  , 0  , 0  , 0  , 0  , 0  , b56, b57, b58, 0  , 0  , 0  , 0],
            [c14, c15, c16, c17, c18, 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0],
            [c23, c24, c25, c26, c27, c28, 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0],
            [0  , 0  , 0  , c35, c36, c37, c38, 0  , 0  , 0  , 0  , 0  , 0  , 0],
            [0  , 0  , 0  , 0  , c45, 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0],
            [d15, d16, d17, d18, 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0],
            [d24, d25, d26, d27, 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0],
            [0  , d34, d35, 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0],
            [e16, e17, 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0],
            [e25, 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0]
        ])

        # XOR sum of each column
        m1 = np.bitwise_xor.reduce(M, axis=0)
            
        # XOR a with m1
        a_ext = [0, 0, 0, 0, 0, 0, a1, a2, a3, a4, a5, a6, a7, a8]
        m1 ^= a_ext
        
        block14 = ''.join(str(bit) for bit in m1.tolist())

        result_blocks.append(block14)
        # feed the first 8 bits back as the next seed
        current_seed = block14[:8]

    return ''.join(result_blocks)

def digital_logistic_map_bits(init_bits: str, n: int) -> str:
    """
    Inputs:
      init_bits : string of '0'/'1', length k ≥ 1
      n         : number of iterations (≥ 0)

    Output:
      A single string of length k*(n+1), being the initial k‑bit state followed
      by each of the n subsequent k‑bit states, concatenated with no separators.
    """
    # --- sanity checks ---
    if n < 0:
        raise ValueError("n must be ≥ 0")
    k = len(init_bits)
    if k < 1:
        raise ValueError("init_bits must have length ≥ 1")
    if any(c not in '01' for c in init_bits):
        raise ValueError("init_bits must consist only of '0' and '1'")

    # convert initial bits to float x in (0,1)
    a = np.array([int(c) for c in init_bits], dtype=int)
    x = np.dot(a, 2.0 ** -np.arange(1, k+1))

    # pull in the pre‑computed maps once
    all_cols, aij_defs, top_defs, rec_defs = MG.generate_parsed_maps(k)

    # collect bit‑strings
    bit_strings = [init_bits]

    for iteration in range(n):
        if x == 0 or x == 1:
            raise RuntimeError(f"State fell to {x} at iteration {iteration}; can’t continue.")

        # --- compute all N‑operators from current bits ---
        # Step 3: seed from aij_defs
        bit_cache = {}
        for key, (head_key, tail_keys) in aij_defs.items():
            _, i, j = head_key
            bit_cache[head_key] = a[i-1] ^ a[j-1]

        # Step 4: expand via top_defs & rec_defs
        pending = {**top_defs, **rec_defs}
        while pending:
            progress = False

            for lbl, (head_key, tail_keys) in list(pending.items()):
                d, i, j = MG.label_to_key(lbl)

                # only proceed when *both* head_key *and* all tail_keys are in bit_cache
                if head_key not in bit_cache or not all(tk in bit_cache for tk in tail_keys):
                    continue

                # now safe
                bit = bit_cache[head_key]

                # compute this bit:
               
                # - If tail_keys = 0, bit = 0
                if not tail_keys and d > 1:
                    bit_cache[(d, i, j)] = 0
                else:    
                   
                    # - start from the head bit                
                    bit = bit_cache[head_key]
               
                    # - if there are any tail_keys, XOR them together and AND with head
                    if tail_keys:
                        xor_tail = 0
                        for tk in tail_keys:
                            xor_tail ^= bit_cache[tk]
                        bit &= xor_tail

                    bit_cache[(d, i, j)] = bit

                    # prune: if this bit==0, any deeper same-(i,j) ops must also be zero
                    if bit == 0:
                        for other, (_, other_tails) in list(pending.items()):
                            e, ii, jj = MG.label_to_key(other)
                            if (ii, jj) == (i, j) and e > d:
                                bit_cache[(e, ii, jj)] = 0
                                pending.pop(other)

                # done with this label
                pending.pop(lbl, None)
                progress = True

            if not progress:
                stuck = ", ".join(str(MG.label_to_key(l)) for l in pending)
                raise RuntimeError(f"Stuck resolving recursive maps: {stuck}")

        # --- Steps 5–8: build new fractional‑binary vector ---
        # Step 5: XOR‑sum each column
        XCol = np.zeros(2*k, dtype=int)
        for r, entries in all_cols.items():
            acc = 0
            for (d,i,j) in entries:
                acc ^= bit_cache.get((d,i,j), 0)
            XCol[2*k - 1 - r] = acc
 
        # Step 6: extend by appending current bits
        a_ext = np.zeros(2*k, dtype=int)
        a_ext[k:] = a

        # Step 7: XOR
        XCol ^= a_ext

        # Step 8: shift left by 2 (drop first two)
        shifted = np.concatenate([XCol[2:], [0,0]]) #adding 2 zeros creates bias

        # --- Step 9: convert back to real x ---
        x = float(np.dot(shifted, 2.0 ** -np.arange(1, len(shifted)+1)))

        # --- convert real x back into k‑bit fractional binary a[0..k-1] ---
        a = np.zeros(k, dtype=int)
        z = x
        for i in range(k):
            z *= 2
            if z >= 1:
                a[i] = 1
                z -= 1
            # else a[i] stays 0

        bits_next = "".join(str(bit) for bit in a)
        bit_strings.append(bits_next)

    # return all k‑bit states concatenated
    return "".join(bit_strings)

