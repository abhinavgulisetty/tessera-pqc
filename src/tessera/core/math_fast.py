from __future__ import annotations
import numpy as np
from numba import njit, prange

KYBER_Q = 3329
KYBER_N = 256
KYBER_OMEGA = 3061

DILITHIUM_Q = 8380417
DILITHIUM_N = 256
DILITHIUM_OMEGA = 1753


@njit(cache=True)
def _mod_exp(base: int, exp: int, mod: int) -> int:
    result = 1
    base = base % mod
    while exp > 0:
        if exp & 1:
            result = (result * base) % mod
        exp >>= 1
        base = (base * base) % mod
    return result


@njit(cache=True)
def _bit_reverse_index(i: int, bits: int) -> int:
    rev = 0
    for _ in range(bits):
        rev = (rev << 1) | (i & 1)
        i >>= 1
    return rev


@njit(cache=True)
def _bit_reverse_copy(a: np.ndarray, n: int) -> np.ndarray:
    bits = 0
    temp = n - 1
    while temp > 0:
        bits += 1
        temp >>= 1
    result = np.empty(n, dtype=np.int64)
    for i in range(n):
        result[_bit_reverse_index(i, bits)] = a[i]
    return result


@njit(cache=True)
def _compute_twiddles(n: int, q: int, omega: int) -> np.ndarray:
    twiddles = np.empty(n, dtype=np.int64)
    w = 1
    for i in range(n):
        twiddles[i] = w
        w = (w * omega) % q
    return twiddles


@njit(cache=True)
def ntt_fast(poly: np.ndarray, n: int, q: int, omega: int) -> np.ndarray:
    a = _bit_reverse_copy(poly.astype(np.int64), n)
    length = 2
    while length <= n:
        half = length >> 1
        w_len = _mod_exp(omega, n // length, q)
        for start in range(0, n, length):
            wj = 1
            for j in range(half):
                u = a[start + j]
                v = (a[start + j + half] * wj) % q
                a[start + j] = (u + v) % q
                a[start + j + half] = (u - v + q) % q
                wj = (wj * w_len) % q
        length <<= 1
    return a


@njit(cache=True)
def intt_fast(poly_ntt: np.ndarray, n: int, q: int, omega: int) -> np.ndarray:
    a = poly_ntt.copy().astype(np.int64)
    omega_inv = _mod_exp(omega, q - 2, q)
    length = n
    while length >= 2:
        half = length >> 1
        w_len = _mod_exp(omega_inv, n // length, q)
        for start in range(0, n, length):
            wj = 1
            for j in range(half):
                u = a[start + j]
                v = a[start + j + half]
                a[start + j] = (u + v) % q
                a[start + j + half] = ((u - v + q) * wj) % q
                wj = (wj * w_len) % q
        length >>= 1
    a = _bit_reverse_copy(a, n)
    n_inv = _mod_exp(n, q - 2, q)
    for i in range(n):
        a[i] = (a[i] * n_inv) % q
    return a


@njit(cache=True, parallel=True)
def pointwise_mul_fast(a: np.ndarray, b: np.ndarray, q: int) -> np.ndarray:
    n = len(a)
    result = np.empty(n, dtype=np.int64)
    for i in prange(n):
        result[i] = (a[i] * b[i]) % q
    return result


@njit(cache=True, parallel=True)
def poly_add_fast(a: np.ndarray, b: np.ndarray, q: int) -> np.ndarray:
    n = len(a)
    result = np.empty(n, dtype=np.int64)
    for i in prange(n):
        result[i] = (a[i] + b[i]) % q
    return result


@njit(cache=True, parallel=True)
def poly_sub_fast(a: np.ndarray, b: np.ndarray, q: int) -> np.ndarray:
    n = len(a)
    result = np.empty(n, dtype=np.int64)
    for i in prange(n):
        result[i] = (a[i] - b[i] + q) % q
    return result


@njit(cache=True)
def poly_reduce_fast(a: np.ndarray, q: int) -> np.ndarray:
    n = len(a)
    result = np.empty(n, dtype=np.int64)
    for i in range(n):
        result[i] = a[i] % q
    return result


def ntt_kyber(poly: np.ndarray) -> np.ndarray:
    return ntt_fast(poly.astype(np.int64), KYBER_N, KYBER_Q, KYBER_OMEGA)


def intt_kyber(poly_ntt: np.ndarray) -> np.ndarray:
    return intt_fast(poly_ntt.astype(np.int64), KYBER_N, KYBER_Q, KYBER_OMEGA)


def ntt_dilithium(poly: np.ndarray) -> np.ndarray:
    return ntt_fast(poly.astype(np.int64), DILITHIUM_N, DILITHIUM_Q, DILITHIUM_OMEGA)


def intt_dilithium(poly_ntt: np.ndarray) -> np.ndarray:
    return intt_fast(poly_ntt.astype(np.int64), DILITHIUM_N, DILITHIUM_Q, DILITHIUM_OMEGA)


def warmup():
    test_poly = np.random.randint(0, KYBER_Q, KYBER_N, dtype=np.int64)
    _ = ntt_kyber(test_poly)
    _ = intt_kyber(test_poly)
    test_poly2 = np.random.randint(0, DILITHIUM_Q, DILITHIUM_N, dtype=np.int64)
    _ = ntt_dilithium(test_poly2)
    _ = intt_dilithium(test_poly2)
