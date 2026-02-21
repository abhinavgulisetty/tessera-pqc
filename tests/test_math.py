"""
Tests for core/math.py — PolynomialRing and NTT routines.
"""
import numpy as np
import pytest
from tessera.core.math import PolynomialRing, _bit_reverse_copy


@pytest.fixture
def ring():
    return PolynomialRing()


@pytest.fixture
def random_poly(ring):
    rng = np.random.default_rng(seed=42)
    return rng.integers(0, ring.q, ring.n, dtype=np.int64)


class TestBitReverseCopy:
    def test_length_preserved(self):
        a = np.arange(8, dtype=np.int64)
        assert len(_bit_reverse_copy(a, 8)) == 8

    def test_known_order(self):
        a = np.arange(8, dtype=np.int64)
        r = _bit_reverse_copy(a, 8)
        assert list(r) == [0, 4, 2, 6, 1, 5, 3, 7]

    def test_double_application_is_identity(self):
        a = np.arange(16, dtype=np.int64)
        assert np.all(_bit_reverse_copy(_bit_reverse_copy(a, 16), 16) == a)


class TestNTT:
    def test_output_length(self, ring, random_poly):
        assert len(ring.ntt(random_poly)) == ring.n

    def test_output_in_range(self, ring, random_poly):
        result = ring.ntt(random_poly)
        assert np.all(result >= 0)
        assert np.all(result < ring.q)

    def test_zero_input(self, ring):
        z = np.zeros(ring.n, dtype=np.int64)
        assert np.all(ring.ntt(z) == 0)

    def test_deterministic(self, ring, random_poly):
        assert np.all(ring.ntt(random_poly) == ring.ntt(random_poly))


class TestInvNTT:
    def test_round_trip(self, ring, random_poly):
        """inv_ntt(ntt(x)) must equal x mod q for any polynomial."""
        recovered = ring.inv_ntt(ring.ntt(random_poly))
        expected  = random_poly % ring.q
        assert np.all(recovered == expected)

    def test_round_trip_many(self, ring):
        """Pass 20 random polynomials through the round-trip."""
        rng = np.random.default_rng(seed=0)
        for _ in range(20):
            x   = rng.integers(0, ring.q, ring.n, dtype=np.int64)
            rec = ring.inv_ntt(ring.ntt(x))
            assert np.all(rec == x % ring.q), "Round-trip failed"

    def test_verify_round_trip_helper(self, ring, random_poly):
        assert ring.verify_round_trip(random_poly) is True


class TestPointMul:
    def test_output_shape(self, ring, random_poly):
        a = ring.ntt(random_poly)
        b = ring.ntt(random_poly)
        assert ring.point_mul(a, b).shape == (ring.n,)

    def test_output_in_range(self, ring, random_poly):
        a = ring.ntt(random_poly)
        b = ring.ntt(random_poly)
        r = ring.point_mul(a, b)
        assert np.all(r >= 0) and np.all(r < ring.q)

    def test_commutativity(self, ring, random_poly):
        rng = np.random.default_rng(1)
        a = ring.ntt(random_poly)
        b = ring.ntt(rng.integers(0, ring.q, ring.n, dtype=np.int64))
        assert np.all(ring.point_mul(a, b) == ring.point_mul(b, a))

    def test_multiply_by_one(self, ring, random_poly):
        """Multiplying by the NTT of [1, 0, 0, ...] should return the original."""
        one = np.zeros(ring.n, dtype=np.int64)
        one[0] = 1
        a_ntt   = ring.ntt(random_poly)
        one_ntt = ring.ntt(one)
        prod    = ring.point_mul(a_ntt, one_ntt)
        expected = ring.poly_mul(random_poly, one)
        recovered_direct   = ring.inv_ntt(prod)
        assert np.all(recovered_direct == expected)


class TestPolyMul:
    def test_multiply_by_zero(self, ring, random_poly):
        z = np.zeros(ring.n, dtype=np.int64)
        assert np.all(ring.poly_mul(random_poly, z) == 0)

    def test_commutativity(self, ring):
        rng = np.random.default_rng(2)
        a = rng.integers(0, ring.q, ring.n, dtype=np.int64)
        b = rng.integers(0, ring.q, ring.n, dtype=np.int64)
        assert np.all(ring.poly_mul(a, b) == ring.poly_mul(b, a))

    def test_associativity(self, ring):
        rng = np.random.default_rng(3)
        a = rng.integers(0, ring.q, ring.n, dtype=np.int64)
        b = rng.integers(0, ring.q, ring.n, dtype=np.int64)
        c = rng.integers(0, ring.q, ring.n, dtype=np.int64)
        ab_c = ring.poly_mul(ring.poly_mul(a, b), c)
        a_bc = ring.poly_mul(a, ring.poly_mul(b, c))
        assert np.all(ab_c == a_bc)


class TestAddSub:
    def test_add_is_inverse_of_sub(self, ring, random_poly):
        rng = np.random.default_rng(4)
        b = rng.integers(0, ring.q, ring.n, dtype=np.int64)
        result = ring.add(ring.sub(random_poly, b), b) % ring.q
        assert np.all(result == random_poly % ring.q)

    def test_add_zero(self, ring, random_poly):
        z = np.zeros(ring.n, dtype=np.int64)
        assert np.all(ring.add(random_poly, z) == random_poly % ring.q)
