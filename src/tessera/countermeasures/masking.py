from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable
from enum import Enum
import secrets


class MaskingScheme(Enum):
    BOOLEAN = "boolean"
    ARITHMETIC = "arithmetic"
    MULTIPLICATIVE = "multiplicative"


@dataclass
class MaskedValue:
    shares: np.ndarray
    scheme: MaskingScheme
    modulus: Optional[int] = None
    
    @property
    def num_shares(self) -> int:
        return len(self.shares)
    
    @property
    def order(self) -> int:
        return self.num_shares - 1
    
    def unmask(self) -> np.ndarray:
        if self.scheme == MaskingScheme.BOOLEAN:
            result = self.shares[0].copy()
            for share in self.shares[1:]:
                result ^= share
            return result
        elif self.scheme == MaskingScheme.ARITHMETIC:
            result = self.shares[0].copy()
            for share in self.shares[1:]:
                result = (result + share) % self.modulus
            return result
        elif self.scheme == MaskingScheme.MULTIPLICATIVE:
            result = self.shares[0].copy()
            for share in self.shares[1:]:
                result = (result * share) % self.modulus
            return result
        raise ValueError(f"Unknown masking scheme: {self.scheme}")


class Masking(ABC):
    
    @abstractmethod
    def mask(self, value: np.ndarray, num_shares: int) -> MaskedValue:
        pass
    
    @abstractmethod
    def unmask(self, masked: MaskedValue) -> np.ndarray:
        pass
    
    @abstractmethod
    def refresh(self, masked: MaskedValue) -> MaskedValue:
        pass


class BooleanMasking(Masking):
    
    def __init__(self, bit_width: int = 64):
        self.bit_width = bit_width
        self._max_val = (1 << bit_width) - 1
    
    def mask(self, value: np.ndarray, num_shares: int = 2) -> MaskedValue:
        if num_shares < 2:
            raise ValueError("Need at least 2 shares")
        
        shares = []
        running = value.copy()
        
        for _ in range(num_shares - 1):
            random_share = np.array([
                secrets.randbits(self.bit_width) for _ in range(value.size)
            ], dtype=value.dtype).reshape(value.shape)
            shares.append(random_share)
            running = running ^ random_share
        
        shares.append(running)
        return MaskedValue(
            shares=np.array(shares),
            scheme=MaskingScheme.BOOLEAN
        )
    
    def unmask(self, masked: MaskedValue) -> np.ndarray:
        return masked.unmask()
    
    def refresh(self, masked: MaskedValue) -> MaskedValue:
        if masked.num_shares < 2:
            return masked
        
        new_shares = [s.copy() for s in masked.shares]
        
        for i in range(len(new_shares) - 1):
            random_val = np.array([
                secrets.randbits(self.bit_width) for _ in range(new_shares[i].size)
            ], dtype=new_shares[i].dtype).reshape(new_shares[i].shape)
            new_shares[i] = new_shares[i] ^ random_val
            new_shares[i + 1] = new_shares[i + 1] ^ random_val
        
        return MaskedValue(
            shares=np.array(new_shares),
            scheme=MaskingScheme.BOOLEAN
        )
    
    def secure_and(self, a: MaskedValue, b: MaskedValue) -> MaskedValue:
        if a.num_shares != b.num_shares:
            raise ValueError("Share count mismatch")
        if a.num_shares != 2:
            raise ValueError("ISW AND requires exactly 2 shares")
        
        a0, a1 = a.shares[0], a.shares[1]
        b0, b1 = b.shares[0], b.shares[1]
        
        r = np.array([
            secrets.randbits(self.bit_width) for _ in range(a0.size)
        ], dtype=a0.dtype).reshape(a0.shape)
        
        c0 = (a0 & b0) ^ r
        c1 = (a1 & b1) ^ ((a0 & b1) ^ (a1 & b0) ^ r)
        
        return MaskedValue(
            shares=np.array([c0, c1]),
            scheme=MaskingScheme.BOOLEAN
        )
    
    def secure_xor(self, a: MaskedValue, b: MaskedValue) -> MaskedValue:
        if a.num_shares != b.num_shares:
            raise ValueError("Share count mismatch")
        
        new_shares = np.array([
            a.shares[i] ^ b.shares[i] for i in range(a.num_shares)
        ])
        
        return MaskedValue(
            shares=new_shares,
            scheme=MaskingScheme.BOOLEAN
        )
    
    def secure_not(self, a: MaskedValue) -> MaskedValue:
        new_shares = a.shares.copy()
        new_shares[0] = ~new_shares[0] & self._max_val
        return MaskedValue(
            shares=new_shares,
            scheme=MaskingScheme.BOOLEAN
        )


class ArithmeticMasking(Masking):
    
    def __init__(self, modulus: int):
        self.modulus = modulus
    
    def mask(self, value: np.ndarray, num_shares: int = 2) -> MaskedValue:
        if num_shares < 2:
            raise ValueError("Need at least 2 shares")
        
        shares = []
        running = value.copy().astype(np.int64) % self.modulus
        
        for _ in range(num_shares - 1):
            random_share = np.array([
                secrets.randbelow(self.modulus) for _ in range(value.size)
            ], dtype=np.int64).reshape(value.shape)
            shares.append(random_share)
            running = (running - random_share) % self.modulus
        
        shares.append(running)
        return MaskedValue(
            shares=np.array(shares),
            scheme=MaskingScheme.ARITHMETIC,
            modulus=self.modulus
        )
    
    def unmask(self, masked: MaskedValue) -> np.ndarray:
        return masked.unmask()
    
    def refresh(self, masked: MaskedValue) -> MaskedValue:
        if masked.num_shares < 2:
            return masked
        
        new_shares = [s.copy() for s in masked.shares]
        
        for i in range(len(new_shares) - 1):
            random_val = np.array([
                secrets.randbelow(self.modulus) for _ in range(new_shares[i].size)
            ], dtype=np.int64).reshape(new_shares[i].shape)
            new_shares[i] = (new_shares[i] + random_val) % self.modulus
            new_shares[i + 1] = (new_shares[i + 1] - random_val) % self.modulus
        
        return MaskedValue(
            shares=np.array(new_shares),
            scheme=MaskingScheme.ARITHMETIC,
            modulus=self.modulus
        )
    
    def secure_add(self, a: MaskedValue, b: MaskedValue) -> MaskedValue:
        if a.num_shares != b.num_shares:
            raise ValueError("Share count mismatch")
        
        new_shares = np.array([
            (a.shares[i] + b.shares[i]) % self.modulus 
            for i in range(a.num_shares)
        ])
        
        return MaskedValue(
            shares=new_shares,
            scheme=MaskingScheme.ARITHMETIC,
            modulus=self.modulus
        )
    
    def secure_sub(self, a: MaskedValue, b: MaskedValue) -> MaskedValue:
        if a.num_shares != b.num_shares:
            raise ValueError("Share count mismatch")
        
        new_shares = a.shares.copy()
        new_shares[0] = (new_shares[0] - b.shares[0]) % self.modulus
        for i in range(1, a.num_shares):
            new_shares[i] = (new_shares[i] - b.shares[i]) % self.modulus
        
        return MaskedValue(
            shares=new_shares,
            scheme=MaskingScheme.ARITHMETIC,
            modulus=self.modulus
        )
    
    def secure_scalar_mul(self, a: MaskedValue, scalar: int) -> MaskedValue:
        new_shares = np.array([
            (s * scalar) % self.modulus for s in a.shares
        ])
        
        return MaskedValue(
            shares=new_shares,
            scheme=MaskingScheme.ARITHMETIC,
            modulus=self.modulus
        )


class MaskConverter:
    
    def __init__(self, modulus: int, bit_width: int = 64):
        self.modulus = modulus
        self.bit_width = bit_width
        self.boolean = BooleanMasking(bit_width)
        self.arithmetic = ArithmeticMasking(modulus)
    
    def bool_to_arith(self, masked: MaskedValue) -> MaskedValue:
        if masked.scheme != MaskingScheme.BOOLEAN:
            raise ValueError("Expected boolean masked value")
        if masked.num_shares != 2:
            raise ValueError("Conversion requires exactly 2 shares")
        
        x0, x1 = masked.shares[0], masked.shares[1]
        
        r = np.array([
            secrets.randbelow(self.modulus) for _ in range(x0.size)
        ], dtype=np.int64).reshape(x0.shape)
        
        a0 = (x0.astype(np.int64) + r) % self.modulus
        a1 = (x1.astype(np.int64) - r) % self.modulus
        
        return MaskedValue(
            shares=np.array([a0, a1]),
            scheme=MaskingScheme.ARITHMETIC,
            modulus=self.modulus
        )
    
    def arith_to_bool(self, masked: MaskedValue) -> MaskedValue:
        if masked.scheme != MaskingScheme.ARITHMETIC:
            raise ValueError("Expected arithmetic masked value")
        if masked.num_shares != 2:
            raise ValueError("Conversion requires exactly 2 shares")
        
        a0, a1 = masked.shares[0], masked.shares[1]
        
        r = np.array([
            secrets.randbits(self.bit_width) for _ in range(a0.size)
        ], dtype=np.uint64).reshape(a0.shape)
        
        x0 = a0.astype(np.uint64) ^ r
        x1 = a1.astype(np.uint64) ^ r
        
        return MaskedValue(
            shares=np.array([x0, x1]),
            scheme=MaskingScheme.BOOLEAN
        )


class HigherOrderMasking:
    
    def __init__(self, order: int, modulus: Optional[int] = None, bit_width: int = 64):
        self.order = order
        self.num_shares = order + 1
        self.modulus = modulus
        self.bit_width = bit_width
    
    def mask_boolean(self, value: np.ndarray) -> MaskedValue:
        masking = BooleanMasking(self.bit_width)
        return masking.mask(value, self.num_shares)
    
    def mask_arithmetic(self, value: np.ndarray) -> MaskedValue:
        if self.modulus is None:
            raise ValueError("Modulus required for arithmetic masking")
        masking = ArithmeticMasking(self.modulus)
        return masking.mask(value, self.num_shares)
    
    def isw_and(self, a: MaskedValue, b: MaskedValue) -> MaskedValue:
        if a.num_shares != b.num_shares or a.num_shares != self.num_shares:
            raise ValueError("Share count mismatch")
        
        n = self.num_shares
        c = [a.shares[i] & b.shares[i] for i in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                r = np.array([
                    secrets.randbits(self.bit_width) for _ in range(a.shares[0].size)
                ], dtype=a.shares[0].dtype).reshape(a.shares[0].shape)
                
                tmp = (r ^ (a.shares[i] & b.shares[j])) ^ (a.shares[j] & b.shares[i])
                c[i] = c[i] ^ r
                c[j] = c[j] ^ tmp
        
        return MaskedValue(
            shares=np.array(c),
            scheme=MaskingScheme.BOOLEAN
        )
    
    def isw_refresh(self, masked: MaskedValue) -> MaskedValue:
        n = masked.num_shares
        new_shares = [s.copy() for s in masked.shares]
        
        for i in range(n):
            for j in range(i + 1, n):
                r = np.array([
                    secrets.randbits(self.bit_width) for _ in range(new_shares[0].size)
                ], dtype=new_shares[0].dtype).reshape(new_shares[0].shape)
                new_shares[i] = new_shares[i] ^ r
                new_shares[j] = new_shares[j] ^ r
        
        return MaskedValue(
            shares=np.array(new_shares),
            scheme=masked.scheme,
            modulus=masked.modulus
        )


@dataclass
class MaskedNTTConfig:
    n: int
    q: int
    num_shares: int = 2
    refresh_interval: int = 1


class MaskedNTT:
    
    def __init__(self, config: MaskedNTTConfig):
        self.config = config
        self.masking = ArithmeticMasking(config.q)
        self._precompute_twiddles()
    
    def _precompute_twiddles(self):
        n, q = self.config.n, self.config.q
        
        if q == 3329:
            root = 17
        elif q == 8380417:
            root = 3073009
        else:
            root = self._find_primitive_root(n, q)
        
        self.twiddles = np.array([pow(root, i, q) for i in range(n)], dtype=np.int64)
        self.inv_twiddles = np.array([pow(root, -i, q) for i in range(n)], dtype=np.int64)
    
    def _find_primitive_root(self, n: int, q: int) -> int:
        for g in range(2, q):
            if pow(g, n, q) == 1:
                is_primitive = True
                for k in range(1, n):
                    if pow(g, k, q) == 1:
                        is_primitive = False
                        break
                if is_primitive:
                    return g
        raise ValueError(f"No primitive {n}-th root of unity mod {q}")
    
    def masked_ntt(self, masked_poly: MaskedValue) -> MaskedValue:
        shares = []
        for share in masked_poly.shares:
            shares.append(self._ntt_share(share))
        
        result = MaskedValue(
            shares=np.array(shares),
            scheme=MaskingScheme.ARITHMETIC,
            modulus=self.config.q
        )
        
        return self.masking.refresh(result)
    
    def _ntt_share(self, coeffs: np.ndarray) -> np.ndarray:
        n, q = self.config.n, self.config.q
        result = coeffs.copy()
        
        k = 1
        length = n // 2
        while length >= 1:
            for start in range(0, n, 2 * length):
                zeta = self.twiddles[k]
                k += 1
                for j in range(start, start + length):
                    t = (zeta * result[j + length]) % q
                    result[j + length] = (result[j] - t) % q
                    result[j] = (result[j] + t) % q
            length //= 2
        
        return result
    
    def masked_intt(self, masked_ntt: MaskedValue) -> MaskedValue:
        shares = []
        for share in masked_ntt.shares:
            shares.append(self._intt_share(share))
        
        result = MaskedValue(
            shares=np.array(shares),
            scheme=MaskingScheme.ARITHMETIC,
            modulus=self.config.q
        )
        
        return self.masking.refresh(result)
    
    def _intt_share(self, coeffs: np.ndarray) -> np.ndarray:
        n, q = self.config.n, self.config.q
        result = coeffs.copy()
        
        k = n - 1
        length = 1
        while length < n:
            for start in range(0, n, 2 * length):
                zeta = self.inv_twiddles[k]
                k -= 1
                for j in range(start, start + length):
                    t = result[j]
                    result[j] = (t + result[j + length]) % q
                    result[j + length] = (zeta * (result[j + length] - t)) % q
            length *= 2
        
        n_inv = pow(n, -1, q)
        result = (result * n_inv) % q
        
        return result
    
    def masked_pointwise_mul(self, a: MaskedValue, b: MaskedValue) -> MaskedValue:
        if a.num_shares != b.num_shares:
            raise ValueError("Share count mismatch")
        
        q = self.config.q
        n_shares = a.num_shares
        
        result_shares = []
        for i in range(n_shares):
            share_result = np.zeros_like(a.shares[0])
            for j in range(n_shares):
                k = (i + j) % n_shares
                contrib = (a.shares[i] * b.shares[j]) % q
                if k == 0:
                    share_result = (share_result + contrib) % q
            result_shares.append(share_result)
        
        return MaskedValue(
            shares=np.array(result_shares),
            scheme=MaskingScheme.ARITHMETIC,
            modulus=q
        )


class MaskedPolynomialOps:
    
    def __init__(self, n: int, q: int, num_shares: int = 2):
        self.n = n
        self.q = q
        self.num_shares = num_shares
        self.masking = ArithmeticMasking(q)
        self.ntt = MaskedNTT(MaskedNTTConfig(n=n, q=q, num_shares=num_shares))
    
    def mask_polynomial(self, poly: np.ndarray) -> MaskedValue:
        return self.masking.mask(poly, self.num_shares)
    
    def unmask_polynomial(self, masked: MaskedValue) -> np.ndarray:
        return self.masking.unmask(masked)
    
    def add(self, a: MaskedValue, b: MaskedValue) -> MaskedValue:
        return self.masking.secure_add(a, b)
    
    def sub(self, a: MaskedValue, b: MaskedValue) -> MaskedValue:
        return self.masking.secure_sub(a, b)
    
    def scalar_mul(self, a: MaskedValue, scalar: int) -> MaskedValue:
        return self.masking.secure_scalar_mul(a, scalar)
    
    def ntt_mul(self, a: MaskedValue, b: MaskedValue) -> MaskedValue:
        a_ntt = self.ntt.masked_ntt(a)
        b_ntt = self.ntt.masked_ntt(b)
        c_ntt = self.ntt.masked_pointwise_mul(a_ntt, b_ntt)
        return self.ntt.masked_intt(c_ntt)
    
    def refresh_all(self, *masked_values: MaskedValue) -> List[MaskedValue]:
        return [self.masking.refresh(m) for m in masked_values]
