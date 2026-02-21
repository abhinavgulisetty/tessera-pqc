import numpy as np


_DEFAULT_N = 256
_DEFAULT_Q = 3329
_OMEGA     = 3061


def _bit_reverse_copy(a: np.ndarray, n: int) -> np.ndarray:
    """Return a bit-reversal permuted copy of array *a* (length must be n=2^k)."""
    bits = n.bit_length() - 1
    result = np.empty_like(a)
    for i in range(n):
        rev = int(f"{i:0{bits}b}"[::-1], 2)
        result[rev] = a[i]
    return result


class PolynomialRing:
    """
    Handles polynomial arithmetic over ring R_q = Z_q[X] / (X^n + 1).
    Used for Lattice-based cryptography (e.g., Kyber).
    """

    def __init__(self, n: int = _DEFAULT_N, q: int = _DEFAULT_Q):
        """
        Args:
            n (int): Polynomial degree (must be a power of 2).
            q (int): Modulus (prime, with n | q-1).
        """
        if n & (n - 1):
            raise ValueError("n must be a power of 2.")
        self.n = n
        self.q = q
        self._omega: int = _OMEGA if (n == _DEFAULT_N and q == _DEFAULT_Q) else self._find_omega(n, q)


    @staticmethod
    def _find_omega(n: int, q: int) -> int:
        """Find a primitive n-th root of unity mod q via brute-force (small n).

        Requires n | (q-1).
        """
        assert (q - 1) % n == 0, f"n={n} must divide q-1={q - 1}"
        exp = (q - 1) // n
        for g in range(2, q):
            omega = pow(g, exp, q)
            if pow(omega, n // 2, q) != 1:
                return omega
        raise ValueError(f"Could not find primitive {n}-th root of unity mod {q}.")


    def reduce(self, poly: np.ndarray) -> np.ndarray:
        """Reduce all coefficients modulo q into [0, q)."""
        return np.mod(poly, self.q).astype(np.int64)

    def __add__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Element-wise polynomial addition mod q."""
        return (a + b) % self.q

    def add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (a + b) % self.q

    def __sub__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Element-wise polynomial subtraction mod q."""
        return (a - b) % self.q

    def sub(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (a - b) % self.q


    def ntt(self, poly: np.ndarray) -> np.ndarray:
        """
        Forward Number Theoretic Transform (Cooley-Tukey DIT).

        Implements the iterative butterfly algorithm:
            NTT[k] = Σ_{j=0}^{n-1}  a[j] · ω^{jk}  (mod q)

        Steps:
          1. Bit-reversal permutation of the input.
          2. log₂(n) butterfly stages; each stage doubles the DFT size.

        Time complexity: O(n log n).

        Args:
            poly: Array of n integer coefficients from Z_q.
        Returns:
            Point-value representation (NTT domain), same length.
        """
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
                    a[start + j]        = (u + v) % q
                    a[start + j + half] = (u - v) % q
                    wj = wj * w_len % q
            length <<= 1
        return a


    def inv_ntt(self, poly_ntt: np.ndarray) -> np.ndarray:
        """
        Inverse Number Theoretic Transform (Gentleman-Sande DIF).

        The inverse uses ω⁻¹ (modular inverse of ω) and scales the output
        by n⁻¹ mod q so that ``inv_ntt(ntt(x)) == x``.

        Steps:
          1. log₂(n) inverse butterfly stages.
          2. Bit-reversal permutation.
          3. Scale by n⁻¹ mod q.

        Args:
            poly_ntt: Point-value array in NTT domain.
        Returns:
            Coefficient representation.
        """
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
                    a[start + j]        = (u + v) % q
                    a[start + j + half] = (u - v) * wj % q
                    wj = wj * w_len % q
            length >>= 1

        a = _bit_reverse_copy(a, n)
        n_inv = pow(n, q - 2, q)
        a = a * n_inv % q
        return a


    def point_mul(self, a_ntt: np.ndarray, b_ntt: np.ndarray) -> np.ndarray:
        """
        Point-wise (element-wise) multiplication in the NTT domain.

        Because we are in the NTT domain, polynomial multiplication becomes
        simple element-wise multiplication:
            (a · b)[k] = a[k] · b[k]  (mod q)

        Args:
            a_ntt: First operand in NTT domain.
            b_ntt: Second operand in NTT domain.
        Returns:
            Product in NTT domain.
        """
        return (a_ntt.astype(np.int64) * b_ntt.astype(np.int64)) % self.q


    def poly_mul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Multiply two polynomials in R_q by going through the NTT domain.

        Equivalent to schoolbook multiplication followed by reduction
        modulo (X^n + 1) and q, but runs in O(n log n).
        """
        return self.inv_ntt(self.point_mul(self.ntt(a), self.ntt(b)))


    def verify_round_trip(self, poly: np.ndarray) -> bool:
        """Assert inv_ntt(ntt(poly)) ≡ poly (mod q)."""
        recovered = self.inv_ntt(self.ntt(poly))
        return bool(np.all(recovered == poly % self.q))
