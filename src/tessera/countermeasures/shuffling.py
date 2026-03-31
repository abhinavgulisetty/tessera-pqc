from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable, Any, TypeVar
import secrets
import time
from enum import Enum


T = TypeVar('T')


class ShuffleStrategy(Enum):
    FISHER_YATES = "fisher_yates"
    RANDOM_PAIRS = "random_pairs"
    RANDOM_CYCLES = "random_cycles"


@dataclass
class ShuffleConfig:
    strategy: ShuffleStrategy = ShuffleStrategy.FISHER_YATES
    random_delays: bool = False
    min_delay_ns: int = 0
    max_delay_ns: int = 1000
    dummy_operations: bool = False
    num_dummy_ops: int = 0


class SecurePermutation:
    
    @staticmethod
    def fisher_yates(n: int) -> np.ndarray:
        perm = np.arange(n)
        for i in range(n - 1, 0, -1):
            j = secrets.randbelow(i + 1)
            perm[i], perm[j] = perm[j], perm[i]
        return perm
    
    @staticmethod
    def random_pairs(n: int, num_swaps: Optional[int] = None) -> np.ndarray:
        perm = np.arange(n)
        if num_swaps is None:
            num_swaps = n * 2
        
        for _ in range(num_swaps):
            i = secrets.randbelow(n)
            j = secrets.randbelow(n)
            perm[i], perm[j] = perm[j], perm[i]
        
        return perm
    
    @staticmethod
    def random_cycles(n: int, max_cycle_len: int = 4) -> np.ndarray:
        perm = np.arange(n)
        indices = list(range(n))
        secrets.SystemRandom().shuffle(indices)
        
        i = 0
        while i < n:
            cycle_len = min(secrets.randbelow(max_cycle_len) + 1, n - i)
            cycle = indices[i:i + cycle_len]
            
            if len(cycle) > 1:
                first_val = perm[cycle[0]]
                for j in range(len(cycle) - 1):
                    perm[cycle[j]] = perm[cycle[j + 1]]
                perm[cycle[-1]] = first_val
            
            i += cycle_len
        
        return perm
    
    @staticmethod
    def inverse_permutation(perm: np.ndarray) -> np.ndarray:
        inv = np.empty_like(perm)
        inv[perm] = np.arange(len(perm))
        return inv
    
    @staticmethod
    def generate(n: int, strategy: ShuffleStrategy = ShuffleStrategy.FISHER_YATES) -> np.ndarray:
        if strategy == ShuffleStrategy.FISHER_YATES:
            return SecurePermutation.fisher_yates(n)
        elif strategy == ShuffleStrategy.RANDOM_PAIRS:
            return SecurePermutation.random_pairs(n)
        elif strategy == ShuffleStrategy.RANDOM_CYCLES:
            return SecurePermutation.random_cycles(n)
        raise ValueError(f"Unknown strategy: {strategy}")


class ShuffledOperation:
    
    def __init__(self, config: ShuffleConfig = None):
        self.config = config or ShuffleConfig()
    
    def _random_delay(self):
        if self.config.random_delays:
            delay_ns = secrets.randbelow(
                self.config.max_delay_ns - self.config.min_delay_ns + 1
            ) + self.config.min_delay_ns
            time.sleep(delay_ns / 1e9)
    
    def _dummy_operation(self, size: int) -> np.ndarray:
        return np.array([secrets.randbits(64) for _ in range(size)], dtype=np.int64)
    
    def shuffle_loop(
        self, 
        n: int, 
        operation: Callable[[int], T]
    ) -> List[T]:
        perm = SecurePermutation.generate(n, self.config.strategy)
        results = [None] * n
        
        for idx in perm:
            self._random_delay()
            results[idx] = operation(idx)
        
        return results
    
    def shuffle_array_operation(
        self,
        array: np.ndarray,
        operation: Callable[[Any], Any]
    ) -> np.ndarray:
        n = len(array)
        perm = SecurePermutation.generate(n, self.config.strategy)
        result = np.empty_like(array)
        
        for idx in perm:
            self._random_delay()
            result[idx] = operation(array[idx])
        
        return result
    
    def shuffle_pairwise_operation(
        self,
        a: np.ndarray,
        b: np.ndarray,
        operation: Callable[[Any, Any], Any]
    ) -> np.ndarray:
        if len(a) != len(b):
            raise ValueError("Arrays must have same length")
        
        n = len(a)
        perm = SecurePermutation.generate(n, self.config.strategy)
        result = np.empty_like(a)
        
        if self.config.dummy_operations:
            dummy_a = self._dummy_operation(self.config.num_dummy_ops)
            dummy_b = self._dummy_operation(self.config.num_dummy_ops)
        
        for idx in perm:
            self._random_delay()
            
            if self.config.dummy_operations and secrets.randbelow(2):
                _ = operation(dummy_a[0], dummy_b[0])
            
            result[idx] = operation(a[idx], b[idx])
        
        return result


class ShuffledNTT:
    
    def __init__(self, n: int, q: int, config: ShuffleConfig = None):
        self.n = n
        self.q = q
        self.config = config or ShuffleConfig()
        self.shuffler = ShuffledOperation(config)
        self._precompute_twiddles()
    
    def _precompute_twiddles(self):
        if self.q == 3329:
            root = 17
        elif self.q == 8380417:
            root = 3073009
        else:
            root = self._find_primitive_root()
        
        self.twiddles = np.array([pow(root, i, self.q) for i in range(self.n)], dtype=np.int64)
        self.inv_twiddles = np.array([pow(root, -i, self.q) for i in range(self.n)], dtype=np.int64)
    
    def _find_primitive_root(self) -> int:
        for g in range(2, self.q):
            if pow(g, self.n, self.q) == 1:
                is_primitive = True
                for k in range(1, self.n):
                    if pow(g, k, self.q) == 1:
                        is_primitive = False
                        break
                if is_primitive:
                    return g
        raise ValueError(f"No primitive {self.n}-th root of unity mod {self.q}")
    
    def ntt_shuffled(self, coeffs: np.ndarray) -> np.ndarray:
        result = coeffs.copy()
        
        k = 1
        length = self.n // 2
        while length >= 1:
            butterflies = []
            for start in range(0, self.n, 2 * length):
                zeta = self.twiddles[k]
                k += 1
                for j in range(start, start + length):
                    butterflies.append((j, j + length, zeta))
            
            perm = SecurePermutation.generate(len(butterflies), self.config.strategy)
            
            for idx in perm:
                j, j_plus, zeta = butterflies[idx]
                t = (zeta * result[j_plus]) % self.q
                result[j_plus] = (result[j] - t) % self.q
                result[j] = (result[j] + t) % self.q
            
            length //= 2
        
        return result
    
    def intt_shuffled(self, coeffs: np.ndarray) -> np.ndarray:
        result = coeffs.copy()
        
        k = self.n - 1
        length = 1
        while length < self.n:
            butterflies = []
            for start in range(0, self.n, 2 * length):
                zeta = self.inv_twiddles[k]
                k -= 1
                for j in range(start, start + length):
                    butterflies.append((j, j + length, zeta))
            
            perm = SecurePermutation.generate(len(butterflies), self.config.strategy)
            
            for idx in perm:
                j, j_plus, zeta = butterflies[idx]
                t = result[j]
                result[j] = (t + result[j_plus]) % self.q
                result[j_plus] = (zeta * (result[j_plus] - t)) % self.q
            
            length *= 2
        
        n_inv = pow(self.n, -1, self.q)
        result = (result * n_inv) % self.q
        
        return result


class ShuffledPolynomialMul:
    
    def __init__(self, n: int, q: int, config: ShuffleConfig = None):
        self.n = n
        self.q = q
        self.config = config or ShuffleConfig()
        self.ntt = ShuffledNTT(n, q, config)
        self.shuffler = ShuffledOperation(config)
    
    def pointwise_mul_shuffled(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return self.shuffler.shuffle_pairwise_operation(
            a, b, lambda x, y: (x * y) % self.q
        )
    
    def ntt_mul_shuffled(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a_ntt = self.ntt.ntt_shuffled(a)
        b_ntt = self.ntt.ntt_shuffled(b)
        c_ntt = self.pointwise_mul_shuffled(a_ntt, b_ntt)
        return self.ntt.intt_shuffled(c_ntt)


class RandomDelayGenerator:
    
    def __init__(
        self, 
        min_delay_ns: int = 0, 
        max_delay_ns: int = 10000,
        distribution: str = "uniform"
    ):
        self.min_delay_ns = min_delay_ns
        self.max_delay_ns = max_delay_ns
        self.distribution = distribution
    
    def delay(self):
        if self.distribution == "uniform":
            delay_ns = secrets.randbelow(
                self.max_delay_ns - self.min_delay_ns + 1
            ) + self.min_delay_ns
        elif self.distribution == "exponential":
            mean = (self.max_delay_ns + self.min_delay_ns) / 2
            delay_ns = int(np.random.exponential(mean))
            delay_ns = max(self.min_delay_ns, min(self.max_delay_ns, delay_ns))
        else:
            delay_ns = secrets.randbelow(
                self.max_delay_ns - self.min_delay_ns + 1
            ) + self.min_delay_ns
        
        time.sleep(delay_ns / 1e9)
    
    def random_nop_loop(self, max_iterations: int = 100):
        n = secrets.randbelow(max_iterations)
        dummy = 0
        for _ in range(n):
            dummy ^= secrets.randbits(32)
        return dummy


class DummyOperationGenerator:
    
    def __init__(self, operation_type: str = "arithmetic"):
        self.operation_type = operation_type
    
    def generate_dummy_arithmetic(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        a = np.array([secrets.randbits(32) for _ in range(n)], dtype=np.int64)
        b = np.array([secrets.randbits(32) for _ in range(n)], dtype=np.int64)
        return a, b
    
    def execute_dummy_operations(self, count: int, size: int = 1):
        for _ in range(count):
            a, b = self.generate_dummy_arithmetic(size)
            if self.operation_type == "arithmetic":
                _ = a + b
                _ = a * b
            elif self.operation_type == "bitwise":
                _ = a ^ b
                _ = a & b
            elif self.operation_type == "memory":
                temp = a.copy()
                a[:] = b
                b[:] = temp


@dataclass
class ShufflingCountermeasure:
    shuffle_ntt_butterflies: bool = True
    shuffle_pointwise_ops: bool = True
    shuffle_coefficient_ops: bool = True
    random_delays: bool = True
    dummy_operations: bool = True
    
    def create_config(self) -> ShuffleConfig:
        return ShuffleConfig(
            strategy=ShuffleStrategy.FISHER_YATES,
            random_delays=self.random_delays,
            min_delay_ns=0,
            max_delay_ns=1000,
            dummy_operations=self.dummy_operations,
            num_dummy_ops=10
        )
