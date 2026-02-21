"""
Tests for core/primitives.py — Baby-Kyber LatticeKEM.
"""
import pytest
from tessera.core.math import PolynomialRing
from tessera.core.primitives import LatticeKEM


@pytest.fixture
def kem():
    return LatticeKEM()


class TestKeygen:
    def test_returns_two_bytes(self, kem):
        pk, sk = kem.keygen()
        assert isinstance(pk, bytes)
        assert isinstance(sk, bytes)

    def test_pk_length(self, kem):
        pk, _ = kem.keygen()
        assert len(pk) > 32

    def test_sk_length(self, kem):
        _, sk = kem.keygen()
        assert len(sk) > 0

    def test_keypairs_are_unique(self, kem):
        pk1, sk1 = kem.keygen()
        pk2, sk2 = kem.keygen()
        assert pk1 != pk2
        assert sk1 != sk2


class TestEncaps:
    def test_returns_ct_and_ss(self, kem):
        pk, _ = kem.keygen()
        ct, ss = kem.encaps(pk)
        assert isinstance(ct, bytes)
        assert isinstance(ss, bytes)

    def test_ss_is_32_bytes(self, kem):
        pk, _ = kem.keygen()
        _, ss = kem.encaps(pk)
        assert len(ss) == 32

    def test_ciphertexts_are_unique(self, kem):
        pk, _ = kem.keygen()
        ct1, ss1 = kem.encaps(pk)
        ct2, ss2 = kem.encaps(pk)
        assert ct1 != ct2
        assert ss1 != ss2


class TestDecaps:
    def test_shared_secret_matches(self, kem):
        """Core correctness: encapper and decapper derive the same secret."""
        pk, sk = kem.keygen()
        ct, ss_enc = kem.encaps(pk)
        ss_dec = kem.decaps(sk, ct)
        assert ss_enc == ss_dec

    def test_shared_secret_is_32_bytes(self, kem):
        pk, sk = kem.keygen()
        ct, _ = kem.encaps(pk)
        ss = kem.decaps(sk, ct)
        assert len(ss) == 32

    def test_wrong_sk_gives_different_secret(self, kem):
        pk,  sk  = kem.keygen()
        pk2, sk2 = kem.keygen()
        ct, ss_enc = kem.encaps(pk)
        ss_wrong   = kem.decaps(sk2, ct)
        assert ss_wrong != ss_enc

    @pytest.mark.parametrize("trial", range(10))
    def test_round_trip_repeated(self, kem, trial):
        """Run the full KEM 10 times; every round must succeed."""
        pk, sk = kem.keygen()
        ct, ss_enc = kem.encaps(pk)
        ss_dec = kem.decaps(sk, ct)
        assert ss_enc == ss_dec, f"Trial {trial} failed"
