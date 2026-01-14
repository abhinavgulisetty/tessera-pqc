import numpy as np

class PolynomialRing:
    """
    Handles polynomial arithmetic over ring R_q = Z_q[X] / (X^n + 1).
    Used for Lattice-based cryptography (e.g., Kyber).
    """

    def __init__(self, n: int = 256, q: int = 3329):
        """
        Args:
            n (int): The polynomial degree.
            q (int): The modulus.
        """
        self.n = n
        self.q = q

    def ntt(self, poly: np.ndarray) -> np.ndarray:
        """
        Forward Number Theoretic Transform.
        
        TODO: Implement the iterative NTT algorithm (Cooley-Tukey).
        Reference: "Speeding up the Number Theoretic Transform for R-LWE"
        
        Args:
            poly: Coefficient representation
        Returns:
            np.ndarray: Point-value representation
        """
        raise NotImplementedError("Implement NTT logic here.")

    def inv_ntt(self, poly_ntt: np.ndarray) -> np.ndarray:
        """
        Inverse Number Theoretic Transform.
        
        TODO: Implement inverse NTT (Gentleman-Sande).
        """
        raise NotImplementedError("Implement Inverse NTT logic here.")

    def point_mul(self, a_ntt: np.ndarray, b_ntt: np.ndarray) -> np.ndarray:
        """
        Point-wise multiplication in the NTT domain.
        
        TODO: Implement modular multiplication for the vector elements.
        """
        pass
