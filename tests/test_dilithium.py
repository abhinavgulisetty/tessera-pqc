import pytest
from tessera.algorithms.dilithium import (
    Dilithium, dilithium2, dilithium3, dilithium5,
    DILITHIUM2, DILITHIUM3, DILITHIUM5
)


class TestDilithiumParams:
    def test_dilithium2_params(self):
        assert DILITHIUM2.k == 4
        assert DILITHIUM2.l == 4
        assert DILITHIUM2.eta == 2
        assert DILITHIUM2.tau == 39
        assert DILITHIUM2.gamma1 == 2**17
        assert DILITHIUM2.gamma2 == (8380417 - 1) // 88
        assert DILITHIUM2.beta == 78
        assert DILITHIUM2.omega == 80

    def test_dilithium3_params(self):
        assert DILITHIUM3.k == 6
        assert DILITHIUM3.l == 5
        assert DILITHIUM3.eta == 4
        assert DILITHIUM3.tau == 49
        assert DILITHIUM3.gamma1 == 2**19
        assert DILITHIUM3.gamma2 == (8380417 - 1) // 32
        assert DILITHIUM3.beta == 196
        assert DILITHIUM3.omega == 55

    def test_dilithium5_params(self):
        assert DILITHIUM5.k == 8
        assert DILITHIUM5.l == 7
        assert DILITHIUM5.eta == 2
        assert DILITHIUM5.tau == 60
        assert DILITHIUM5.gamma1 == 2**19
        assert DILITHIUM5.gamma2 == (8380417 - 1) // 32
        assert DILITHIUM5.beta == 120
        assert DILITHIUM5.omega == 75


class TestDilithiumKeygen:
    @pytest.mark.parametrize("sig_fn", [dilithium2, dilithium3, dilithium5])
    def test_keygen_returns_bytes(self, sig_fn):
        sig = sig_fn()
        pk, sk = sig.keygen()
        assert isinstance(pk, bytes)
        assert isinstance(sk, bytes)

    @pytest.mark.parametrize("sig_fn", [dilithium2, dilithium3, dilithium5])
    def test_keygen_unique(self, sig_fn):
        sig = sig_fn()
        pk1, sk1 = sig.keygen()
        pk2, sk2 = sig.keygen()
        assert pk1 != pk2
        assert sk1 != sk2


class TestDilithiumSign:
    @pytest.mark.parametrize("sig_fn", [dilithium2, dilithium3, dilithium5])
    def test_sign_returns_bytes(self, sig_fn):
        sig = sig_fn()
        pk, sk = sig.keygen()
        signature = sig.sign(sk, b"test message")
        assert isinstance(signature, bytes)

    @pytest.mark.parametrize("sig_fn", [dilithium2, dilithium3, dilithium5])
    def test_sign_different_for_different_messages(self, sig_fn):
        sig = sig_fn()
        pk, sk = sig.keygen()
        sig1 = sig.sign(sk, b"message 1")
        sig2 = sig.sign(sk, b"message 2")
        assert sig1 != sig2

    @pytest.mark.parametrize("sig_fn", [dilithium2, dilithium3, dilithium5])
    def test_sign_deterministic(self, sig_fn):
        sig = sig_fn()
        pk, sk = sig.keygen()
        msg = b"test message"
        sig1 = sig.sign(sk, msg)
        sig2 = sig.sign(sk, msg)
        assert sig1 == sig2


class TestDilithiumVerify:
    @pytest.mark.parametrize("sig_fn", [dilithium2, dilithium3, dilithium5])
    def test_verify_valid_signature(self, sig_fn):
        sig = sig_fn()
        pk, sk = sig.keygen()
        msg = b"test message"
        signature = sig.sign(sk, msg)
        assert sig.verify(pk, msg, signature) is True

    @pytest.mark.parametrize("sig_fn", [dilithium2, dilithium3, dilithium5])
    def test_verify_wrong_message(self, sig_fn):
        sig = sig_fn()
        pk, sk = sig.keygen()
        msg = b"test message"
        signature = sig.sign(sk, msg)
        assert sig.verify(pk, b"wrong message", signature) is False

    @pytest.mark.parametrize("sig_fn", [dilithium2, dilithium3, dilithium5])
    def test_verify_wrong_public_key(self, sig_fn):
        sig = sig_fn()
        pk1, sk1 = sig.keygen()
        pk2, sk2 = sig.keygen()
        msg = b"test message"
        signature = sig.sign(sk1, msg)
        assert sig.verify(pk2, msg, signature) is False

    @pytest.mark.parametrize("sig_fn", [dilithium2, dilithium3, dilithium5])
    def test_verify_corrupted_signature(self, sig_fn):
        sig = sig_fn()
        pk, sk = sig.keygen()
        msg = b"test message"
        signature = sig.sign(sk, msg)
        corrupted = bytearray(signature)
        corrupted[50] ^= 0xFF
        assert sig.verify(pk, msg, bytes(corrupted)) is False


class TestDilithiumFullCycle:
    @pytest.mark.parametrize("sig_fn", [dilithium2, dilithium3, dilithium5])
    def test_multiple_sign_verify(self, sig_fn):
        sig = sig_fn()
        pk, sk = sig.keygen()
        for i in range(5):
            msg = f"message {i}".encode()
            signature = sig.sign(sk, msg)
            assert sig.verify(pk, msg, signature) is True

    @pytest.mark.parametrize("sig_fn", [dilithium2, dilithium3, dilithium5])
    def test_empty_message(self, sig_fn):
        sig = sig_fn()
        pk, sk = sig.keygen()
        signature = sig.sign(sk, b"")
        assert sig.verify(pk, b"", signature) is True

    @pytest.mark.parametrize("sig_fn", [dilithium2, dilithium3, dilithium5])
    def test_long_message(self, sig_fn):
        sig = sig_fn()
        pk, sk = sig.keygen()
        msg = b"A" * 10000
        signature = sig.sign(sk, msg)
        assert sig.verify(pk, msg, signature) is True


class TestDilithiumProperties:
    def test_name_property(self):
        assert dilithium2().name == "Dilithium2"
        assert dilithium3().name == "Dilithium3"
        assert dilithium5().name == "Dilithium5"

    @pytest.mark.parametrize("sig_fn,expected_pk,expected_sk,expected_sig", [
        (dilithium2, 1312, None, None),
        (dilithium3, 1952, None, None),
        (dilithium5, 2592, None, None),
    ])
    def test_key_sizes(self, sig_fn, expected_pk, expected_sk, expected_sig):
        sig = sig_fn()
        assert sig.public_key_size == expected_pk


class TestDilithiumBaseClass:
    def test_inherits_from_signature(self):
        from tessera.algorithms.base import Signature
        sig = dilithium2()
        assert isinstance(sig, Signature)

    def test_abstract_methods_implemented(self):
        sig = dilithium2()
        assert hasattr(sig, 'keygen')
        assert hasattr(sig, 'sign')
        assert hasattr(sig, 'verify')
        assert callable(sig.keygen)
        assert callable(sig.sign)
        assert callable(sig.verify)
