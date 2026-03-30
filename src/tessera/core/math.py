from __future__ import annotations
import numpy as np

_DEFAULT_N = 256
_DEFAULT_Q = 3329
_OMEGA = 3061

_USE_NUMBA = True

try:
    from .math_fast import (
        ntt_fast,
        intt_fast,
        pointwise_mul_fast,
        poly_add_fast,
        poly_sub_fast,
        warmup as _warmup_numba,
    )
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False


def _bit_reverse_copy(a: np.ndarray, n: int) -> np.ndarray:
    bits = n.bit_length() - 1
    result = np.empty_like(a)
    for i in range(n):
        rev = int(f"{i:0{bits}b}"[::-1], 2)
        result[rev] = a[i]
    return result


class PolynomialRing:
    def __init__(self, n: int = _DEFAULT_N, q: int = _DEFAULT_Q, use_fast: bool = True):
        if n & (n - 1):
            raise ValueError("n must be a power of 2.")
        self.n = n
        self.q = q
        self._use_fast = use_fast and _NUMBA_AVAILABLE and _USE_NUMBA
        self._omega: int = _OMEGA if (n == _DEFAULT_N and q == _DEFAULT_Q) else self._find_omega(n, q)

    @staticmethod
    def _find_omega(n: int, q: int) -> int:
        assert (q - 1) % n == 0, f"n={n} must divide q-1={q - 1}"
        exp = (q - 1) // n
        for g in range(2, q):
            omega = pow(g, exp, q)
            if pow(omega, n // 2, q) != 1:
                return omega
        raise ValueError(f"Could not find primitive {n}-th root of unity mod {q}.")

    def reduce(self, poly: np.ndarray) -> np.ndarray:
        return np.mod(poly, self.q).astype(np.int64)

    def add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if self._use_fast:
            return poly_add_fast(a.astype(np.int64), b.astype(np.int64), self.q)
        return (a + b) % self.q

    def sub(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if self._use_fast:
            return poly_sub_fast(a.astype(np.int64), b.astype(np.int64), self.q)
        return (a - b) % self.q

    def ntt(self, poly: np.ndarray) -> np.ndarray:
        if self._use_fast:
            return ntt_fast(poly.astype(np.int64), self.n, self.q, self._omega)
        return self._ntt_pure(poly)

    def _ntt_pure(self, poly: np.ndarray) -> np.ndarray:
        n, q = self.n, self.q
        a = _bit_reverse_copy(poly.astype(np.int64), n)
        length = 2
        while length <= n:
            half = length // 2
            w_len = pow(self._omega, n // length, q)
            for start in range(0, n, length):
                wj = 1
                for j in range(half):
                    u = int(a[start + j])
                    v = int(a[start + j + half]) * wj % q
                    a[start + j] = (u + v) % q
                    a[start + j + half] = (u - v) % q
                    wj = wj * w_len % q
            length <<= 1
        return a

    def inv_ntt(self, poly_ntt: np.ndarray) -> np.ndarray:
        if self._use_fast:
            return intt_fast(poly_ntt.astype(np.int64), self.n, self.q, self._omega)
        return self._inv_ntt_pure(poly_ntt)

    def _inv_ntt_pure(self, poly_ntt: np.ndarray) -> np.ndarray:
        n, q = self.n, self.q
        a = poly_ntt.copy().astype(np.int64)
        omega_inv = pow(self._omega, q - 2, q)
        length = n
        while length >= 2:
            half = length // 2
            w_len = pow(omega_inv, n // length, q)
            for start in range(0, n, length):
                wj = 1
                for j in range(half):
                    u = int(a[start + j])
                    v = int(a[start + j + half])
                    a[start + j] = (u + v) % q
                    a[start + j + half] = (u - v) * wj % q
                    wj = wj * w_len % q
            length >>= 1
        a = _bit_reverse_copy(a, n)
        n_inv = pow(n, q - 2, q)
        a = a * n_inv % q
        return a

    def point_mul(self, a_ntt: np.ndarray, b_ntt: np.ndarray) -> np.ndarray:
        if self._use_fast:
            return pointwise_mul_fast(a_ntt.astype(np.int64), b_ntt.astype(np.int64), self.q)
        return (a_ntt.astype(np.int64) * b_ntt.astype(np.int64)) % self.q

    def poly_mul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return self.inv_ntt(self.point_mul(self.ntt(a), self.ntt(b)))

    def verify_round_trip(self, poly: np.ndarray) -> bool:
        recovered = self.inv_ntt(self.ntt(poly))
        return bool(np.all(recovered == poly % self.q))


def set_use_numba(enabled: bool) -> None:
    global _USE_NUMBA
    _USE_NUMBA = enabled


def is_numba_available() -> bool:
    return _NUMBA_AVAILABLE


def warmup() -> None:
    if _NUMBA_AVAILABLE:
        _warmup_numba()
