from typing import Tuple, Optional
import numpy as np
from .math import PolynomialRing

class LatticeKEM:
    """
    Scaffold for a Key Encapsulation Mechanism (e.g., simplified Kyber).
    """

    def __init__(self, ring: PolynomialRing):
        self.ring = ring

    def keygen(self) -> Tuple[bytes, bytes]:
        """
        Generates public and private keys.
        
        TODO:
        1. Generate random seed.
        2. Expand seed into matrix A (using SHAKE-128).
        3. Sample error vectors s and e.
        4. Compute b = A*s + e.
        5. Return (pk, sk).
        """
        raise NotImplementedError("Implement Lattice KeyGen.")

    def encaps(self, pk: bytes) -> Tuple[bytes, bytes]:
        """
        Generates a shared secret and a ciphertext encapsulated under pk.
        
        TODO:
        1. Decode pk.
        2. Generate random coins.
        3. Encrypt internal message to get ciphertext c.
        4. Hash message to get shared secret ss.
        """
        raise NotImplementedError("Implement Encapsulation.")

    def decaps(self, sk: bytes, c: bytes) -> bytes:
        """
        Decapsulates ciphertext c using secret key sk to recover shared secret.
        """
        raise NotImplementedError("Implement Decapsulation.")
