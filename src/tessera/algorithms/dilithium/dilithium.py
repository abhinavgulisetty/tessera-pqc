from __future__ import annotations
import os
import hashlib
from typing import Tuple
import numpy as np

from ..base import Signature
from .params import DilithiumParams, DILITHIUM5
from ...core.math_fast import ntt_fast, intt_fast, pointwise_mul_fast, _mod_exp

DILITHIUM_Q = 8380417
DILITHIUM_N = 256
DILITHIUM_OMEGA_512 = 1753
DILITHIUM_OMEGA_256 = pow(1753, 2, 8380417)


class Dilithium(Signature):
    def __init__(self, params: DilithiumParams = DILITHIUM5):
        self.params = params
        self._zetas = self._compute_zetas()

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
    def signature_size(self) -> int:
        return self.params.signature_bytes

    def _compute_zetas(self) -> np.ndarray:
        zetas = np.zeros(256, dtype=np.int64)
        zetas[0] = 1
        for i in range(1, 256):
            zetas[i] = (zetas[i-1] * DILITHIUM_OMEGA_256) % DILITHIUM_Q
        return zetas

    def _ntt(self, a: np.ndarray) -> np.ndarray:
        return ntt_fast(a.astype(np.int64), DILITHIUM_N, DILITHIUM_Q, DILITHIUM_OMEGA_256)

    def _intt(self, a: np.ndarray) -> np.ndarray:
        return intt_fast(a.astype(np.int64), DILITHIUM_N, DILITHIUM_Q, DILITHIUM_OMEGA_256)

    def _pointwise_mul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return pointwise_mul_fast(a.astype(np.int64), b.astype(np.int64), DILITHIUM_Q)

    def _reduce32(self, a: int) -> int:
        t = (a + (1 << 22)) >> 23
        t = a - t * DILITHIUM_Q
        return t

    def _caddq(self, a: int) -> int:
        a += (a >> 31) & DILITHIUM_Q
        return a

    def _freeze(self, a: int) -> int:
        a = self._reduce32(a)
        a = self._caddq(a)
        return a

    def _poly_reduce(self, a: np.ndarray) -> np.ndarray:
        return np.array([self._freeze(int(x)) for x in a], dtype=np.int64)

    def _poly_add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (a + b) % DILITHIUM_Q

    def _poly_sub(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (a - b + DILITHIUM_Q) % DILITHIUM_Q

    def _to_centered(self, val: int) -> int:
        if val > DILITHIUM_Q // 2:
            return val - DILITHIUM_Q
        return val

    def _polyvec_ntt(self, v: list[np.ndarray]) -> list[np.ndarray]:
        return [self._ntt(p) for p in v]

    def _polyvec_intt(self, v: list[np.ndarray]) -> list[np.ndarray]:
        return [self._intt(p) for p in v]

    def _polyvec_pointwise_acc(self, a: list[np.ndarray], b: list[np.ndarray]) -> np.ndarray:
        acc = np.zeros(DILITHIUM_N, dtype=np.int64)
        for ai, bi in zip(a, b):
            acc = (acc + self._pointwise_mul(ai, bi)) % DILITHIUM_Q
        return acc

    def _power2round(self, a: int) -> Tuple[int, int]:
        a = a % DILITHIUM_Q
        a1 = (a + (1 << (self.params.d - 1)) - 1) >> self.params.d
        a0 = a - (a1 << self.params.d)
        return a1, a0

    def _decompose(self, a: int) -> Tuple[int, int]:
        a = a % DILITHIUM_Q
        a1 = (a + 127) >> 7
        if self.params.gamma2 == (DILITHIUM_Q - 1) // 32:
            a1 = (a1 * 1025 + (1 << 21)) >> 22
            a1 &= 15
        else:
            a1 = (a1 * 11275 + (1 << 23)) >> 24
            a1 ^= ((43 - a1) >> 31) & a1
        a0 = a - a1 * 2 * self.params.gamma2
        a0 -= (((DILITHIUM_Q - 1) // 2 - a0) >> 31) & DILITHIUM_Q
        return a1, a0

    def _highbits(self, a: int) -> int:
        return self._decompose(a)[0]

    def _lowbits(self, a: int) -> int:
        return self._decompose(a)[1]

    def _make_hint(self, z: int, r: int) -> int:
        r = r % DILITHIUM_Q
        z = z % DILITHIUM_Q
        r1 = self._highbits(r)
        v1 = self._highbits((r + z) % DILITHIUM_Q)
        if r1 != v1:
            return 1
        return 0

    def _use_hint(self, h: int, r: int) -> int:
        r = r % DILITHIUM_Q
        r1, r0 = self._decompose(r)
        if h == 0:
            return r1
        if r0 > 0:
            if self.params.gamma2 == (DILITHIUM_Q - 1) // 32:
                return (r1 + 1) & 15
            else:
                return (r1 + 1) % 44 if r1 < 43 else 0
        else:
            if self.params.gamma2 == (DILITHIUM_Q - 1) // 32:
                return (r1 - 1) & 15
            else:
                return (r1 - 1) if r1 > 0 else 43

    def _rej_uniform(self, buf: bytes, n: int) -> np.ndarray:
        coeffs = []
        idx = 0
        while len(coeffs) < n and idx + 3 <= len(buf):
            t = buf[idx] | (buf[idx + 1] << 8) | (buf[idx + 2] << 16)
            t &= 0x7FFFFF
            idx += 3
            if t < DILITHIUM_Q:
                coeffs.append(t)
        while len(coeffs) < n:
            coeffs.append(0)
        return np.array(coeffs[:n], dtype=np.int64)

    def _rej_eta(self, buf: bytes, eta: int, n: int) -> np.ndarray:
        coeffs = []
        idx = 0
        if eta == 2:
            while len(coeffs) < n and idx < len(buf):
                t0 = buf[idx] & 0x0F
                t1 = buf[idx] >> 4
                idx += 1
                if t0 < 15:
                    coeffs.append(2 - (t0 % 5))
                if t1 < 15 and len(coeffs) < n:
                    coeffs.append(2 - (t1 % 5))
        else:
            while len(coeffs) < n and idx < len(buf):
                t0 = buf[idx] & 0x0F
                t1 = buf[idx] >> 4
                idx += 1
                if t0 < 9:
                    coeffs.append(4 - t0)
                if t1 < 9 and len(coeffs) < n:
                    coeffs.append(4 - t1)
        while len(coeffs) < n:
            coeffs.append(0)
        return np.array(coeffs[:n], dtype=np.int64) % DILITHIUM_Q

    def _sample_in_ball(self, seed: bytes) -> np.ndarray:
        c = np.zeros(DILITHIUM_N, dtype=np.int64)
        shake = hashlib.shake_256(seed)
        buf = shake.digest(136)
        signs = int.from_bytes(buf[:8], 'little')
        k = 8
        for i in range(256 - self.params.tau, 256):
            while True:
                if k >= len(buf):
                    buf = shake.digest(136)
                    k = 0
                j = buf[k]
                k += 1
                if j <= i:
                    break
            c[i] = c[j]
            c[j] = 1 - 2 * (signs & 1)
            signs >>= 1
        return c

    def _expand_a(self, rho: bytes) -> list[list[np.ndarray]]:
        A = []
        for i in range(self.params.k):
            row = []
            for j in range(self.params.l):
                shake = hashlib.shake_128(rho + bytes([j, i]))
                buf = shake.digest(672)
                poly = self._rej_uniform(buf, DILITHIUM_N)
                row.append(poly)
            A.append(row)
        return A

    def _expand_s(self, rho_prime: bytes, eta: int, start: int, count: int) -> list[np.ndarray]:
        polys = []
        for i in range(count):
            shake = hashlib.shake_256(rho_prime + (start + i).to_bytes(2, 'little'))
            buf = shake.digest(136 * 2)
            poly = self._rej_eta(buf, eta, DILITHIUM_N)
            polys.append(poly)
        return polys

    def _expand_mask(self, rho_prime: bytes, kappa: int) -> list[np.ndarray]:
        polys = []
        gamma1_bits = 18 if self.params.gamma1 == (1 << 17) else 20
        poly_bytes = DILITHIUM_N * gamma1_bits // 8
        for i in range(self.params.l):
            shake = hashlib.shake_256(rho_prime + (kappa + i).to_bytes(2, 'little'))
            buf = shake.digest(poly_bytes)
            poly = self._unpack_gamma1(buf, gamma1_bits)
            polys.append(poly)
        return polys

    def _unpack_gamma1(self, buf: bytes, bits: int) -> np.ndarray:
        coeffs = []
        gamma1 = self.params.gamma1
        if bits == 18:
            mask = 0x3FFFF
            for i in range(0, len(buf) - 8, 9):
                coeffs.append((buf[i] | (buf[i+1] << 8) | (buf[i+2] << 16)) & mask)
                coeffs.append(((buf[i+2] >> 2) | (buf[i+3] << 6) | (buf[i+4] << 14)) & mask)
                coeffs.append(((buf[i+4] >> 4) | (buf[i+5] << 4) | (buf[i+6] << 12)) & mask)
                coeffs.append(((buf[i+6] >> 6) | (buf[i+7] << 2) | (buf[i+8] << 10)) & mask)
        else:
            mask = 0xFFFFF
            for i in range(0, len(buf) - 4, 5):
                coeffs.append((buf[i] | (buf[i+1] << 8) | ((buf[i+2] & 0x0F) << 16)) & mask)
                coeffs.append(((buf[i+2] >> 4) | (buf[i+3] << 4) | (buf[i+4] << 12)) & mask)
        result = []
        for c in coeffs[:DILITHIUM_N]:
            val = gamma1 - c
            if val < 0:
                val += DILITHIUM_Q
            result.append(val)
        return np.array(result, dtype=np.int64)

    def _pack_pk(self, rho: bytes, t1: list[np.ndarray]) -> bytes:
        pk = rho
        for poly in t1:
            for i in range(0, DILITHIUM_N, 4):
                pk += bytes([
                    poly[i] & 0xFF,
                    ((poly[i] >> 8) | (poly[i+1] << 2)) & 0xFF,
                    ((poly[i+1] >> 6) | (poly[i+2] << 4)) & 0xFF,
                    ((poly[i+2] >> 4) | (poly[i+3] << 6)) & 0xFF,
                    (poly[i+3] >> 2) & 0xFF,
                ])
        return pk

    def _unpack_pk(self, pk: bytes) -> Tuple[bytes, list[np.ndarray]]:
        rho = pk[:32]
        t1 = []
        idx = 32
        for _ in range(self.params.k):
            coeffs = []
            for i in range(0, DILITHIUM_N, 4):
                coeffs.append(pk[idx] | ((pk[idx+1] & 0x03) << 8))
                coeffs.append((pk[idx+1] >> 2) | ((pk[idx+2] & 0x0F) << 6))
                coeffs.append((pk[idx+2] >> 4) | ((pk[idx+3] & 0x3F) << 4))
                coeffs.append((pk[idx+3] >> 6) | (pk[idx+4] << 2))
                idx += 5
            t1.append(np.array(coeffs, dtype=np.int64))
        return rho, t1

    def _poly_chknorm(self, poly: np.ndarray, bound: int) -> bool:
        for c in poly:
            c = int(c)
            if c > (DILITHIUM_Q - 1) // 2:
                c = DILITHIUM_Q - c
            if c >= bound:
                return False
        return True

    def _polyvec_chknorm(self, v: list[np.ndarray], bound: int) -> bool:
        return all(self._poly_chknorm(p, bound) for p in v)

    def keygen(self) -> Tuple[bytes, bytes]:
        zeta = os.urandom(32)
        shake = hashlib.shake_256(zeta)
        buf = shake.digest(128)
        rho, rho_prime, K = buf[:32], buf[32:96], buf[96:128]
        A = self._expand_a(rho)
        s1 = self._expand_s(rho_prime, self.params.eta, 0, self.params.l)
        s2 = self._expand_s(rho_prime, self.params.eta, self.params.l, self.params.k)
        s1_hat = self._polyvec_ntt(s1)
        t = []
        for i in range(self.params.k):
            acc = np.zeros(DILITHIUM_N, dtype=np.int64)
            for j in range(self.params.l):
                acc = (acc + self._pointwise_mul(self._ntt(A[i][j]), s1_hat[j])) % DILITHIUM_Q
            t.append((self._intt(acc) + s2[i]) % DILITHIUM_Q)
        t1, t0 = [], []
        for poly in t:
            t1_poly, t0_poly = [], []
            for c in poly:
                hi, lo = self._power2round(int(c))
                t1_poly.append(hi)
                t0_poly.append(lo)
            t1.append(np.array(t1_poly, dtype=np.int64))
            t0.append(np.array(t0_poly, dtype=np.int64))
        pk = self._pack_pk(rho, t1)
        tr = hashlib.shake_256(pk).digest(64)
        sk = rho + K + tr
        for poly in s1:
            sk += self._pack_eta(poly)
        for poly in s2:
            sk += self._pack_eta(poly)
        for poly in t0:
            sk += self._pack_t0(poly)
        return pk, sk

    def _pack_eta(self, poly: np.ndarray) -> bytes:
        eta = self.params.eta
        buf = bytearray()
        if eta == 2:
            for i in range(0, DILITHIUM_N, 8):
                t = []
                for j in range(8):
                    c = int(poly[i+j])
                    if c > DILITHIUM_Q // 2:
                        c -= DILITHIUM_Q
                    t.append(eta - c)
                buf.append((t[0] | (t[1] << 3) | (t[2] << 6)) & 0xFF)
                buf.append(((t[2] >> 2) | (t[3] << 1) | (t[4] << 4) | (t[5] << 7)) & 0xFF)
                buf.append(((t[5] >> 1) | (t[6] << 2) | (t[7] << 5)) & 0xFF)
        else:
            for i in range(0, DILITHIUM_N, 2):
                c0 = int(poly[i])
                c1 = int(poly[i+1])
                if c0 > DILITHIUM_Q // 2:
                    c0 -= DILITHIUM_Q
                if c1 > DILITHIUM_Q // 2:
                    c1 -= DILITHIUM_Q
                t0 = eta - c0
                t1 = eta - c1
                buf.append((t0 | (t1 << 4)) & 0xFF)
        return bytes(buf)

    def _pack_t0(self, poly: np.ndarray) -> bytes:
        buf = bytearray()
        d = self.params.d
        for i in range(0, DILITHIUM_N, 8):
            t = []
            for j in range(8):
                c = int(poly[i+j])
                if c > DILITHIUM_Q // 2:
                    c -= DILITHIUM_Q
                t.append((1 << (d - 1)) - c)
            buf.append(t[0] & 0xFF)
            buf.append(((t[0] >> 8) | (t[1] << 5)) & 0xFF)
            buf.append((t[1] >> 3) & 0xFF)
            buf.append(((t[1] >> 11) | (t[2] << 2)) & 0xFF)
            buf.append(((t[2] >> 6) | (t[3] << 7)) & 0xFF)
            buf.append((t[3] >> 1) & 0xFF)
            buf.append(((t[3] >> 9) | (t[4] << 4)) & 0xFF)
            buf.append((t[4] >> 4) & 0xFF)
            buf.append(((t[4] >> 12) | (t[5] << 1)) & 0xFF)
            buf.append(((t[5] >> 7) | (t[6] << 6)) & 0xFF)
            buf.append((t[6] >> 2) & 0xFF)
            buf.append(((t[6] >> 10) | (t[7] << 3)) & 0xFF)
            buf.append((t[7] >> 5) & 0xFF)
        return bytes(buf)

    def _unpack_sk(self, sk: bytes) -> Tuple[bytes, bytes, bytes, list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        rho = sk[:32]
        K = sk[32:64]
        tr = sk[64:128]
        idx = 128
        eta_bytes = DILITHIUM_N * (3 if self.params.eta == 2 else 4) // 8
        s1 = []
        for _ in range(self.params.l):
            s1.append(self._unpack_eta(sk[idx:idx+eta_bytes]))
            idx += eta_bytes
        s2 = []
        for _ in range(self.params.k):
            s2.append(self._unpack_eta(sk[idx:idx+eta_bytes]))
            idx += eta_bytes
        t0_bytes = DILITHIUM_N * 13 // 8
        t0 = []
        for _ in range(self.params.k):
            t0.append(self._unpack_t0(sk[idx:idx+t0_bytes]))
            idx += t0_bytes
        return rho, K, tr, s1, s2, t0

    def _unpack_eta(self, buf: bytes) -> np.ndarray:
        eta = self.params.eta
        coeffs = []
        if eta == 2:
            for i in range(0, len(buf), 3):
                coeffs.append(buf[i] & 0x07)
                coeffs.append((buf[i] >> 3) & 0x07)
                coeffs.append(((buf[i] >> 6) | (buf[i+1] << 2)) & 0x07)
                coeffs.append((buf[i+1] >> 1) & 0x07)
                coeffs.append((buf[i+1] >> 4) & 0x07)
                coeffs.append(((buf[i+1] >> 7) | (buf[i+2] << 1)) & 0x07)
                coeffs.append((buf[i+2] >> 2) & 0x07)
                coeffs.append((buf[i+2] >> 5) & 0x07)
        else:
            for b in buf:
                coeffs.append(b & 0x0F)
                coeffs.append(b >> 4)
        return np.array([eta - c for c in coeffs[:DILITHIUM_N]], dtype=np.int64) % DILITHIUM_Q

    def _unpack_t0(self, buf: bytes) -> np.ndarray:
        coeffs = []
        d = self.params.d
        for i in range(0, len(buf), 13):
            coeffs.append(buf[i] | ((buf[i+1] & 0x1F) << 8))
            coeffs.append((buf[i+1] >> 5) | (buf[i+2] << 3) | ((buf[i+3] & 0x03) << 11))
            coeffs.append((buf[i+3] >> 2) | ((buf[i+4] & 0x7F) << 6))
            coeffs.append((buf[i+4] >> 7) | (buf[i+5] << 1) | ((buf[i+6] & 0x0F) << 9))
            coeffs.append((buf[i+6] >> 4) | (buf[i+7] << 4) | ((buf[i+8] & 0x01) << 12))
            coeffs.append((buf[i+8] >> 1) | ((buf[i+9] & 0x3F) << 7))
            coeffs.append((buf[i+9] >> 6) | (buf[i+10] << 2) | ((buf[i+11] & 0x07) << 10))
            coeffs.append((buf[i+11] >> 3) | (buf[i+12] << 5))
        return np.array([(1 << (d-1)) - c for c in coeffs[:DILITHIUM_N]], dtype=np.int64) % DILITHIUM_Q

    def sign(self, sk: bytes, message: bytes) -> bytes:
        rho, K, tr, s1, s2, t0 = self._unpack_sk(sk)
        A = self._expand_a(rho)
        mu = hashlib.shake_256(tr + message).digest(64)
        rho_prime = hashlib.shake_256(K + mu).digest(64)
        s1_hat = self._polyvec_ntt(s1)
        s2_hat = self._polyvec_ntt(s2)
        t0_hat = self._polyvec_ntt(t0)
        kappa = 0
        while True:
            y = self._expand_mask(rho_prime, kappa)
            kappa += self.params.l
            y_hat = self._polyvec_ntt(y)
            w = []
            for i in range(self.params.k):
                acc = np.zeros(DILITHIUM_N, dtype=np.int64)
                for j in range(self.params.l):
                    acc = (acc + self._pointwise_mul(self._ntt(A[i][j]), y_hat[j])) % DILITHIUM_Q
                w.append(self._intt(acc))
            w1 = []
            for poly in w:
                w1.append(np.array([self._highbits(int(c)) for c in poly], dtype=np.int64))
            w1_packed = b""
            if self.params.gamma2 == (DILITHIUM_Q - 1) // 32:
                for poly in w1:
                    for i in range(0, DILITHIUM_N, 2):
                        w1_packed += bytes([(int(poly[i]) | (int(poly[i+1]) << 4)) & 0xFF])
            else:
                for poly in w1:
                    for i in range(0, DILITHIUM_N, 4):
                        w1_packed += bytes([
                            (int(poly[i]) | (int(poly[i+1]) << 6)) & 0xFF,
                            ((int(poly[i+1]) >> 2) | (int(poly[i+2]) << 4)) & 0xFF,
                            ((int(poly[i+2]) >> 4) | (int(poly[i+3]) << 2)) & 0xFF,
                        ])
            c_tilde = hashlib.shake_256(mu + w1_packed).digest(32)
            c = self._sample_in_ball(c_tilde)
            c_hat = self._ntt(c)
            z = []
            for j in range(self.params.l):
                cs1 = self._intt(self._pointwise_mul(c_hat, s1_hat[j]))
                z.append((y[j] + cs1) % DILITHIUM_Q)
            if not self._polyvec_chknorm(z, self.params.gamma1 - self.params.beta):
                continue
            w0 = []
            for i in range(self.params.k):
                cs2 = self._intt(self._pointwise_mul(c_hat, s2_hat[i]))
                w0_poly = (w[i] - cs2) % DILITHIUM_Q
                w0.append(np.array([self._lowbits(int(c)) for c in w0_poly], dtype=np.int64))
            if not self._polyvec_chknorm(w0, self.params.gamma2 - self.params.beta):
                continue
            h = []
            total_hints = 0
            for i in range(self.params.k):
                ct0 = self._intt(self._pointwise_mul(c_hat, t0_hat[i]))
                neg_ct0 = self._poly_sub(np.zeros(DILITHIUM_N, dtype=np.int64), ct0)
                cs2 = self._intt(self._pointwise_mul(c_hat, s2_hat[i]))
                r = (w[i] - cs2 + ct0) % DILITHIUM_Q
                h_poly = np.array([self._make_hint(int(neg_ct0[j]), int(r[j])) for j in range(DILITHIUM_N)], dtype=np.int64)
                total_hints += int(np.sum(h_poly))
                h.append(h_poly)
            if total_hints > self.params.omega:
                continue
            return self._pack_sig(c_tilde, z, h)

    def _pack_sig(self, c_tilde: bytes, z: list[np.ndarray], h: list[np.ndarray]) -> bytes:
        sig = c_tilde
        gamma1_bits = 18 if self.params.gamma1 == (1 << 17) else 20
        for poly in z:
            if gamma1_bits == 18:
                for i in range(0, DILITHIUM_N, 4):
                    t = [(self.params.gamma1 - int(poly[i+j])) % DILITHIUM_Q for j in range(4)]
                    sig += bytes([
                        t[0] & 0xFF,
                        (t[0] >> 8) & 0xFF,
                        ((t[0] >> 16) | (t[1] << 2)) & 0xFF,
                        (t[1] >> 6) & 0xFF,
                        ((t[1] >> 14) | (t[2] << 4)) & 0xFF,
                        (t[2] >> 4) & 0xFF,
                        ((t[2] >> 12) | (t[3] << 6)) & 0xFF,
                        (t[3] >> 2) & 0xFF,
                        (t[3] >> 10) & 0xFF,
                    ])
            else:
                for i in range(0, DILITHIUM_N, 4):
                    t = [(self.params.gamma1 - int(poly[i+j])) % DILITHIUM_Q for j in range(4)]
                    sig += bytes([
                        t[0] & 0xFF,
                        (t[0] >> 8) & 0xFF,
                        ((t[0] >> 16) | (t[1] << 4)) & 0xFF,
                        (t[1] >> 4) & 0xFF,
                        ((t[1] >> 12) | (t[2] << 8)) & 0xFF,
                        (t[2] >> 0) & 0xFF,
                        (t[2] >> 8) & 0xFF,
                        ((t[2] >> 16) | (t[3] << 4)) & 0xFF,
                        (t[3] >> 4) & 0xFF,
                        (t[3] >> 12) & 0xFF,
                    ])
        h_packed = bytearray(self.params.omega + self.params.k)
        k_idx = 0
        for i, h_poly in enumerate(h):
            for j in range(DILITHIUM_N):
                if h_poly[j]:
                    h_packed[k_idx] = j
                    k_idx += 1
            h_packed[self.params.omega + i] = k_idx
        sig += bytes(h_packed)
        return sig

    def _unpack_sig(self, sig: bytes) -> Tuple[bytes, list[np.ndarray], list[np.ndarray]]:
        c_tilde = sig[:32]
        idx = 32
        gamma1_bits = 18 if self.params.gamma1 == (1 << 17) else 20
        z_bytes = DILITHIUM_N * gamma1_bits // 8
        z = []
        for _ in range(self.params.l):
            z.append(self._unpack_z(sig[idx:idx+z_bytes], gamma1_bits))
            idx += z_bytes
        h_packed = sig[idx:]
        h = []
        k = 0
        for i in range(self.params.k):
            h_poly = np.zeros(DILITHIUM_N, dtype=np.int64)
            limit = h_packed[self.params.omega + i]
            while k < limit:
                h_poly[h_packed[k]] = 1
                k += 1
            h.append(h_poly)
        return c_tilde, z, h

    def _unpack_z(self, buf: bytes, bits: int) -> np.ndarray:
        coeffs = []
        if bits == 18:
            for i in range(0, len(buf), 9):
                coeffs.append(buf[i] | (buf[i+1] << 8) | ((buf[i+2] & 0x03) << 16))
                coeffs.append((buf[i+2] >> 2) | (buf[i+3] << 6) | ((buf[i+4] & 0x0F) << 14))
                coeffs.append((buf[i+4] >> 4) | (buf[i+5] << 4) | ((buf[i+6] & 0x3F) << 12))
                coeffs.append((buf[i+6] >> 6) | (buf[i+7] << 2) | (buf[i+8] << 10))
        else:
            for i in range(0, len(buf), 5):
                coeffs.append(buf[i] | (buf[i+1] << 8) | ((buf[i+2] & 0x0F) << 16))
                coeffs.append((buf[i+2] >> 4) | (buf[i+3] << 4) | (buf[i+4] << 12))
        return np.array([(self.params.gamma1 - c) % DILITHIUM_Q for c in coeffs[:DILITHIUM_N]], dtype=np.int64)

    def verify(self, pk: bytes, message: bytes, signature: bytes) -> bool:
        rho, t1 = self._unpack_pk(pk)
        c_tilde, z, h = self._unpack_sig(signature)
        if not self._polyvec_chknorm(z, self.params.gamma1 - self.params.beta):
            return False
        A = self._expand_a(rho)
        tr = hashlib.shake_256(pk).digest(64)
        mu = hashlib.shake_256(tr + message).digest(64)
        c = self._sample_in_ball(c_tilde)
        c_hat = self._ntt(c)
        z_hat = self._polyvec_ntt(z)
        w1_prime = []
        for i in range(self.params.k):
            acc = np.zeros(DILITHIUM_N, dtype=np.int64)
            for j in range(self.params.l):
                acc = (acc + self._pointwise_mul(self._ntt(A[i][j]), z_hat[j])) % DILITHIUM_Q
            t1_2d = (t1[i] * (1 << self.params.d)) % DILITHIUM_Q
            ct1 = self._intt(self._pointwise_mul(c_hat, self._ntt(t1_2d)))
            w_approx = (self._intt(acc) - ct1) % DILITHIUM_Q
            w1_prime_poly = np.array([self._use_hint(int(h[i][j]), int(w_approx[j])) for j in range(DILITHIUM_N)], dtype=np.int64)
            w1_prime.append(w1_prime_poly)
        w1_packed = b""
        if self.params.gamma2 == (DILITHIUM_Q - 1) // 32:
            for poly in w1_prime:
                for i in range(0, DILITHIUM_N, 2):
                    w1_packed += bytes([(int(poly[i]) | (int(poly[i+1]) << 4)) & 0xFF])
        else:
            for poly in w1_prime:
                for i in range(0, DILITHIUM_N, 4):
                    w1_packed += bytes([
                        (int(poly[i]) | (int(poly[i+1]) << 6)) & 0xFF,
                        ((int(poly[i+1]) >> 2) | (int(poly[i+2]) << 4)) & 0xFF,
                        ((int(poly[i+2]) >> 4) | (int(poly[i+3]) << 2)) & 0xFF,
                    ])
        c_tilde_prime = hashlib.shake_256(mu + w1_packed).digest(32)
        return c_tilde == c_tilde_prime


def dilithium2() -> Dilithium:
    from .params import DILITHIUM2
    return Dilithium(DILITHIUM2)


def dilithium3() -> Dilithium:
    from .params import DILITHIUM3
    return Dilithium(DILITHIUM3)


def dilithium5() -> Dilithium:
    return Dilithium(DILITHIUM5)
