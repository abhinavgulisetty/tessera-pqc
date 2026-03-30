import pytest
from tessera.algorithms.kyber import (
    Kyber, kyber512, kyber768, kyber1024,
    KYBER_512, KYBER_768, KYBER_1024
)


class TestKyberParams:
    def test_kyber512_params(self):
        assert KYBER_512.k == 2
        assert KYBER_512.eta1 == 3
        assert KYBER_512.eta2 == 2
        assert KYBER_512.du == 10
        assert KYBER_512.dv == 4

    def test_kyber768_params(self):
        assert KYBER_768.k == 3
        assert KYBER_768.eta1 == 2
        assert KYBER_768.eta2 == 2
        assert KYBER_768.du == 10
        assert KYBER_768.dv == 4

    def test_kyber1024_params(self):
        assert KYBER_1024.k == 4
        assert KYBER_1024.eta1 == 2
        assert KYBER_1024.eta2 == 2
        assert KYBER_1024.du == 11
        assert KYBER_1024.dv == 5


class TestKyberKeygen:
    @pytest.mark.parametrize("kem_fn", [kyber512, kyber768, kyber1024])
    def test_keygen_returns_bytes(self, kem_fn):
        kem = kem_fn()
        pk, sk = kem.keygen()
        assert isinstance(pk, bytes)
        assert isinstance(sk, bytes)

    @pytest.mark.parametrize("kem_fn", [kyber512, kyber768, kyber1024])
    def test_keygen_correct_sizes(self, kem_fn):
        kem = kem_fn()
        pk, sk = kem.keygen()
        assert len(pk) == kem.public_key_size
        assert len(sk) == kem.secret_key_size

    @pytest.mark.parametrize("kem_fn", [kyber512, kyber768, kyber1024])
    def test_keygen_unique(self, kem_fn):
        kem = kem_fn()
        pk1, sk1 = kem.keygen()
        pk2, sk2 = kem.keygen()
        assert pk1 != pk2
        assert sk1 != sk2


class TestKyberEncaps:
    @pytest.mark.parametrize("kem_fn", [kyber512, kyber768, kyber1024])
    def test_encaps_returns_bytes(self, kem_fn):
        kem = kem_fn()
        pk, _ = kem.keygen()
        ct, ss = kem.encaps(pk)
        assert isinstance(ct, bytes)
        assert isinstance(ss, bytes)

    @pytest.mark.parametrize("kem_fn", [kyber512, kyber768, kyber1024])
    def test_encaps_correct_sizes(self, kem_fn):
        kem = kem_fn()
        pk, _ = kem.keygen()
        ct, ss = kem.encaps(pk)
        assert len(ct) == kem.ciphertext_size
        assert len(ss) == kem.shared_secret_size

    @pytest.mark.parametrize("kem_fn", [kyber512, kyber768, kyber1024])
    def test_encaps_unique(self, kem_fn):
        kem = kem_fn()
        pk, _ = kem.keygen()
        ct1, ss1 = kem.encaps(pk)
        ct2, ss2 = kem.encaps(pk)
        assert ct1 != ct2
        assert ss1 != ss2


class TestKyberDecaps:
    @pytest.mark.parametrize("kem_fn", [kyber512, kyber768, kyber1024])
    def test_decaps_matches_encaps(self, kem_fn):
        kem = kem_fn()
        pk, sk = kem.keygen()
        ct, ss_enc = kem.encaps(pk)
        ss_dec = kem.decaps(sk, ct)
        assert ss_enc == ss_dec

    @pytest.mark.parametrize("kem_fn", [kyber512, kyber768, kyber1024])
    def test_decaps_correct_size(self, kem_fn):
        kem = kem_fn()
        pk, sk = kem.keygen()
        ct, _ = kem.encaps(pk)
        ss = kem.decaps(sk, ct)
        assert len(ss) == kem.shared_secret_size

    @pytest.mark.parametrize("kem_fn", [kyber512, kyber768, kyber1024])
    def test_wrong_sk_implicit_reject(self, kem_fn):
        kem = kem_fn()
        pk1, sk1 = kem.keygen()
        _, sk2 = kem.keygen()
        ct, ss_enc = kem.encaps(pk1)
        ss_dec = kem.decaps(sk2, ct)
        assert ss_enc != ss_dec

    @pytest.mark.parametrize("kem_fn,trials", [
        (kyber512, 5),
        (kyber768, 5),
        (kyber1024, 5),
    ])
    def test_round_trip_multiple(self, kem_fn, trials):
        kem = kem_fn()
        for _ in range(trials):
            pk, sk = kem.keygen()
            ct, ss_enc = kem.encaps(pk)
            ss_dec = kem.decaps(sk, ct)
            assert ss_enc == ss_dec


class TestKyberHelpers:
    def test_keygen_keypair(self):
        kem = kyber1024()
        kp = kem.keygen_keypair()
        assert hasattr(kp, 'public_key')
        assert hasattr(kp, 'secret_key')
        assert len(kp.public_key) == kem.public_key_size
        assert len(kp.secret_key) == kem.secret_key_size

    def test_encaps_result(self):
        kem = kyber1024()
        pk, _ = kem.keygen()
        result = kem.encaps_result(pk)
        assert hasattr(result, 'ciphertext')
        assert hasattr(result, 'shared_secret')
        assert len(result.ciphertext) == kem.ciphertext_size
        assert len(result.shared_secret) == kem.shared_secret_size
