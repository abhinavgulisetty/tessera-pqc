from __future__ import annotations
import os
import hashlib
import numpy as np

KYBER_Q = 3329
KYBER_N = 256


def cbd(eta: int, num_bytes: int | None = None) -> np.ndarray:
    if num_bytes is None:
        num_bytes = (2 * eta * KYBER_N + 7) // 8
    raw = np.frombuffer(os.urandom(num_bytes), dtype=np.uint8)
    bits = np.unpackbits(raw)[:2 * eta * KYBER_N].reshape(KYBER_N, 2 * eta)
    a = bits[:, :eta].sum(axis=1).astype(np.int64)
    b = bits[:, eta:].sum(axis=1).astype(np.int64)
    return (a - b) % KYBER_Q


def cbd_from_bytes(data: bytes, eta: int, n: int = KYBER_N, q: int = KYBER_Q) -> np.ndarray:
    raw = np.frombuffer(data, dtype=np.uint8)
    bits = np.unpackbits(raw)[:2 * eta * n].reshape(n, 2 * eta)
    a = bits[:, :eta].sum(axis=1).astype(np.int64)
    b = bits[:, eta:].sum(axis=1).astype(np.int64)
    return (a - b) % q


def rejection_sample_uniform(seed: bytes, q: int, n: int, nonce: int = 0) -> np.ndarray:
    shake = hashlib.shake_128(seed + bytes([nonce]))
    coeffs = []
    while len(coeffs) < n:
        block = shake.digest((n - len(coeffs)) * 3)
        for pos in range(0, len(block) - 2, 3):
            d1 = block[pos] | ((block[pos + 1] & 0x0F) << 8)
            d2 = (block[pos + 1] >> 4) | (block[pos + 2] << 4)
            if d1 < q and len(coeffs) < n:
                coeffs.append(d1)
            if d2 < q and len(coeffs) < n:
                coeffs.append(d2)
        shake = hashlib.shake_128(seed + bytes([nonce]) + len(coeffs).to_bytes(2, 'little'))
    return np.array(coeffs[:n], dtype=np.int64)


def expand_a(seed: bytes, k: int, q: int = KYBER_Q, n: int = KYBER_N) -> list[list[np.ndarray]]:
    A = [[None] * k for _ in range(k)]
    for i in range(k):
        for j in range(k):
            nonce = i * k + j
            A[i][j] = rejection_sample_uniform(seed + bytes([i, j]), q, n, nonce)
    return A


def prf(seed: bytes, nonce: int, length: int) -> bytes:
    shake = hashlib.shake_256(seed + bytes([nonce]))
    return shake.digest(length)


def hash_h(data: bytes) -> bytes:
    return hashlib.sha3_256(data).digest()


def hash_g(data: bytes) -> tuple[bytes, bytes]:
    h = hashlib.sha3_512(data).digest()
    return h[:32], h[32:]


def kdf(data: bytes, length: int = 32) -> bytes:
    shake = hashlib.shake_256(data)
    return shake.digest(length)


def xof(seed: bytes, i: int, j: int) -> hashlib._hashlib.HASH:
    return hashlib.shake_128(seed + bytes([i, j]))
