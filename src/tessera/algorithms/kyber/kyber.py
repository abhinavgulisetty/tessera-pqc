from __future__ import annotations
import os
import hashlib
from typing import Tuple
import numpy as np

from ..base import KEM
from .params import KyberParams, KYBER_1024
from ...core.math import PolynomialRing
from ...core.math_fast import ntt_fast, intt_fast, pointwise_mul_fast, KYBER_Q, KYBER_N, KYBER_OMEGA


class Kyber(KEM):
    def __init__(self, params: KyberParams = KYBER_1024):
        self.params = params
        self.ring = PolynomialRing(n=params.n, q=params.q)
        self._use_fast = True

    @property
    def name(self) -> str:
        return self.params.name

    @property
    def public_key_size(self) -> int:
        return self.params.public_key_bytes

    @property
    def secret_key_size(self) -> int:
        return self.params.secret_key_bytes

    @property
    def ciphertext_size(self) -> int:
        return self.params.ciphertext_bytes

    @property
    def shared_secret_size(self) -> int:
        return self.params.shared_secret_bytes

    def _ntt(self, poly: np.ndarray) -> np.ndarray:
        if self._use_fast:
            return ntt_fast(poly.astype(np.int64), self.params.n, self.params.q, KYBER_OMEGA)
        return self.ring.ntt(poly)

    def _intt(self, poly: np.ndarray) -> np.ndarray:
        if self._use_fast:
            return intt_fast(poly.astype(np.int64), self.params.n, self.params.q, KYBER_OMEGA)
        return self.ring.inv_ntt(poly)

    def _pointwise_mul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if self._use_fast:
            return pointwise_mul_fast(a.astype(np.int64), b.astype(np.int64), self.params.q)
        return self.ring.point_mul(a, b)

    def _prf(self, seed: bytes, nonce: int, length: int) -> bytes:
        return hashlib.shake_256(seed + bytes([nonce])).digest(length)

    def _xof(self, seed: bytes, i: int, j: int):
        return hashlib.shake_128(seed + bytes([j, i]))

    def _hash_h(self, data: bytes) -> bytes:
        return hashlib.sha3_256(data).digest()

    def _hash_g(self, data: bytes) -> Tuple[bytes, bytes]:
        h = hashlib.sha3_512(data).digest()
        return h[:32], h[32:]

    def _kdf(self, data: bytes) -> bytes:
        return hashlib.shake_256(data).digest(32)

    def _cbd(self, data: bytes, eta: int) -> np.ndarray:
        bits_needed = 2 * eta * self.params.n
        raw = np.frombuffer(data[:bits_needed // 4], dtype=np.uint8)
        bits = np.unpackbits(raw)[:bits_needed].reshape(self.params.n, 2 * eta)
        a = bits[:, :eta].sum(axis=1).astype(np.int64)
        b = bits[:, eta:].sum(axis=1).astype(np.int64)
        return (a - b) % self.params.q

    def _parse(self, xof_stream, n: int, q: int) -> np.ndarray:
        coeffs = []
        while len(coeffs) < n:
            buf = xof_stream.digest(3 * (n - len(coeffs)))
            for i in range(0, len(buf) - 2, 3):
                d1 = buf[i] | ((buf[i + 1] & 0x0F) << 8)
                d2 = (buf[i + 1] >> 4) | (buf[i + 2] << 4)
                if d1 < q and len(coeffs) < n:
                    coeffs.append(d1)
                if d2 < q and len(coeffs) < n:
                    coeffs.append(d2)
        return np.array(coeffs[:n], dtype=np.int64)

    def _expand_a(self, rho: bytes) -> list[list[np.ndarray]]:
        k = self.params.k
        A = [[None] * k for _ in range(k)]
        for i in range(k):
            for j in range(k):
                xof = self._xof(rho, i, j)
                A[i][j] = self._parse(xof, self.params.n, self.params.q)
        return A

    def _polyvec_ntt(self, v: list[np.ndarray]) -> list[np.ndarray]:
        return [self._ntt(p) for p in v]

    def _polyvec_intt(self, v: list[np.ndarray]) -> list[np.ndarray]:
        return [self._intt(p) for p in v]

    def _polyvec_add(self, a: list[np.ndarray], b: list[np.ndarray]) -> list[np.ndarray]:
        return [(ai + bi) % self.params.q for ai, bi in zip(a, b)]

    def _polyvec_pointwise_acc(self, a: list[np.ndarray], b: list[np.ndarray]) -> np.ndarray:
        acc = np.zeros(self.params.n, dtype=np.int64)
        for ai, bi in zip(a, b):
            acc = (acc + self._pointwise_mul(ai, bi)) % self.params.q
        return acc

    def _compress(self, x: np.ndarray, d: int) -> np.ndarray:
        return np.round(x * (1 << d) / self.params.q).astype(np.int64) % (1 << d)

    def _decompress(self, x: np.ndarray, d: int) -> np.ndarray:
        return np.round(x * self.params.q / (1 << d)).astype(np.int64) % self.params.q

    def _encode(self, poly: np.ndarray, bits: int) -> bytes:
        out = bytearray()
        buf, filled = 0, 0
        mask = (1 << bits) - 1
        for val in poly:
            buf |= (int(val) & mask) << filled
            filled += bits
            while filled >= 8:
                out.append(buf & 0xFF)
                buf >>= 8
                filled -= 8
        if filled > 0:
            out.append(buf & 0xFF)
        return bytes(out)

    def _decode(self, data: bytes, bits: int, n: int) -> np.ndarray:
        mask = (1 << bits) - 1
        result = []
        buf, filled = 0, 0
        idx = 0
        while len(result) < n and idx < len(data):
            buf |= data[idx] << filled
            filled += 8
            idx += 1
            while filled >= bits and len(result) < n:
                result.append(buf & mask)
                buf >>= bits
                filled -= bits
        return np.array(result, dtype=np.int64)

    def _encode_polyvec(self, v: list[np.ndarray], bits: int) -> bytes:
        return b"".join(self._encode(p, bits) for p in v)

    def _decode_polyvec(self, data: bytes, bits: int, k: int, n: int) -> list[np.ndarray]:
        chunk_size = n * bits // 8
        return [self._decode(data[i * chunk_size:(i + 1) * chunk_size], bits, n) for i in range(k)]

    def _cpapke_keygen(self) -> Tuple[bytes, bytes]:
        d = os.urandom(32)
        rho, sigma = self._hash_g(d)
        A = self._expand_a(rho)
        N = 0
        s = []
        for i in range(self.params.k):
            prf_out = self._prf(sigma, N, 64 * self.params.eta1)
            s.append(self._cbd(prf_out, self.params.eta1))
            N += 1
        e = []
        for i in range(self.params.k):
            prf_out = self._prf(sigma, N, 64 * self.params.eta1)
            e.append(self._cbd(prf_out, self.params.eta1))
            N += 1
        s_hat = self._polyvec_ntt(s)
        e_hat = self._polyvec_ntt(e)
        t_hat = []
        for i in range(self.params.k):
            acc = np.zeros(self.params.n, dtype=np.int64)
            for j in range(self.params.k):
                acc = (acc + self._pointwise_mul(self._ntt(A[i][j]), s_hat[j])) % self.params.q
            t_hat.append((self._intt(acc) + e[i]) % self.params.q)
        t_hat = self._polyvec_ntt(t_hat)
        pk = rho + self._encode_polyvec(t_hat, 12)
        sk = self._encode_polyvec(s_hat, 12)
        return pk, sk

    def _cpapke_encrypt(self, pk: bytes, m: bytes, coins: bytes) -> bytes:
        rho = pk[:32]
        t_hat = self._decode_polyvec(pk[32:], 12, self.params.k, self.params.n)
        A = self._expand_a(rho)
        N = 0
        r = []
        for i in range(self.params.k):
            prf_out = self._prf(coins, N, 64 * self.params.eta1)
            r.append(self._cbd(prf_out, self.params.eta1))
            N += 1
        e1 = []
        for i in range(self.params.k):
            prf_out = self._prf(coins, N, 64 * self.params.eta2)
            e1.append(self._cbd(prf_out, self.params.eta2))
            N += 1
        prf_out = self._prf(coins, N, 64 * self.params.eta2)
        e2 = self._cbd(prf_out, self.params.eta2)
        r_hat = self._polyvec_ntt(r)
        u = []
        for i in range(self.params.k):
            acc = np.zeros(self.params.n, dtype=np.int64)
            for j in range(self.params.k):
                acc = (acc + self._pointwise_mul(self._ntt(A[j][i]), r_hat[j])) % self.params.q
            u.append((self._intt(acc) + e1[i]) % self.params.q)
        m_poly = self._decode(m, 1, self.params.n)
        m_poly = self._decompress(m_poly, 1)
        v = self._polyvec_pointwise_acc(t_hat, r_hat)
        v = (self._intt(v) + e2 + m_poly) % self.params.q
        u_compressed = [self._compress(ui, self.params.du) for ui in u]
        v_compressed = self._compress(v, self.params.dv)
        c1 = self._encode_polyvec(u_compressed, self.params.du)
        c2 = self._encode(v_compressed, self.params.dv)
        return c1 + c2

    def _cpapke_decrypt(self, sk: bytes, ct: bytes) -> bytes:
        c1_len = self.params.k * self.params.n * self.params.du // 8
        c1, c2 = ct[:c1_len], ct[c1_len:]
        u = self._decode_polyvec(c1, self.params.du, self.params.k, self.params.n)
        v = self._decode(c2, self.params.dv, self.params.n)
        u = [self._decompress(ui, self.params.du) for ui in u]
        v = self._decompress(v, self.params.dv)
        s_hat = self._decode_polyvec(sk, 12, self.params.k, self.params.n)
        u_hat = self._polyvec_ntt(u)
        m = self._polyvec_pointwise_acc(s_hat, u_hat)
        m = (v - self._intt(m)) % self.params.q
        m = self._compress(m, 1)
        return self._encode(m, 1)

    def keygen(self) -> Tuple[bytes, bytes]:
        z = os.urandom(32)
        pk, sk_pke = self._cpapke_keygen()
        h_pk = self._hash_h(pk)
        sk = sk_pke + pk + h_pk + z
        return pk, sk

    def encaps(self, pk: bytes) -> Tuple[bytes, bytes]:
        m = os.urandom(32)
        m_h = self._hash_h(m)
        K_bar, r = self._hash_g(m_h + self._hash_h(pk))
        ct = self._cpapke_encrypt(pk, m_h, r)
        K = self._kdf(K_bar + self._hash_h(ct))
        return ct, K

    def decaps(self, sk: bytes, ct: bytes) -> bytes:
        sk_pke_len = self.params.k * self.params.n * 12 // 8
        pk_len = 32 + self.params.k * self.params.n * 12 // 8
        sk_pke = sk[:sk_pke_len]
        pk = sk[sk_pke_len:sk_pke_len + pk_len]
        h_pk = sk[sk_pke_len + pk_len:sk_pke_len + pk_len + 32]
        z = sk[sk_pke_len + pk_len + 32:]
        m_prime = self._cpapke_decrypt(sk_pke, ct)
        K_bar_prime, r_prime = self._hash_g(m_prime + h_pk)
        ct_prime = self._cpapke_encrypt(pk, m_prime, r_prime)
        h_ct = self._hash_h(ct)
        if ct == ct_prime:
            return self._kdf(K_bar_prime + h_ct)
        else:
            return self._kdf(z + h_ct)


def kyber512() -> Kyber:
    from .params import KYBER_512
    return Kyber(KYBER_512)


def kyber768() -> Kyber:
    from .params import KYBER_768
    return Kyber(KYBER_768)


def kyber1024() -> Kyber:
    return Kyber(KYBER_1024)
