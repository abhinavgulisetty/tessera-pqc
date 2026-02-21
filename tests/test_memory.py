"""
Tests for hardware/memory.py — NonVolatileMemory and leakage model.
"""
import numpy as np
import pytest
from tessera.hardware.memory import NonVolatileMemory, _hamming_weight_array


class TestHammingWeight:
    def test_zero_array(self):
        a = np.zeros(10, dtype=np.int64)
        assert _hamming_weight_array(a) == 0

    def test_all_ones_in_16bit(self):
        a = np.full(4, 0xFFFF, dtype=np.uint16).astype(np.int64)
        assert _hamming_weight_array(a) == 4 * 16

    def test_single_bit(self):
        a = np.array([1], dtype=np.int64)
        assert _hamming_weight_array(a) == 1

    def test_known_value(self):
        a = np.array([3329], dtype=np.int64)
        hw = _hamming_weight_array(a)
        assert hw == bin(3329 & 0xFFFF).count('1')


class TestNonVolatileMemory:
    @pytest.fixture
    def nvm(self):
        return NonVolatileMemory()

    @pytest.fixture
    def sample_poly(self):
        rng = np.random.default_rng(99)
        return rng.integers(0, 3329, 256, dtype=np.int64)


    def test_write_then_read(self, nvm, sample_poly):
        nvm.write_checkpoint(0, sample_poly, 0.0)
        recovered = nvm.read_checkpoint(0)
        assert np.all(recovered == sample_poly)

    def test_read_missing_returns_none(self, nvm):
        assert nvm.read_checkpoint(999) is None

    def test_overwrite(self, nvm, sample_poly):
        nvm.write_checkpoint(0, sample_poly, 0.0)
        new_data = np.zeros(256, dtype=np.int64)
        nvm.write_checkpoint(0, new_data, 1.0)
        assert np.all(nvm.read_checkpoint(0) == 0)

    def test_data_is_copied(self, nvm, sample_poly):
        nvm.write_checkpoint(0, sample_poly, 0.0)
        sample_poly[0] = 9999
        assert nvm.read_checkpoint(0)[0] != 9999


    def test_leakage_recorded_per_write(self, nvm, sample_poly):
        nvm.write_checkpoint(0, sample_poly, 10.0)
        nvm.write_checkpoint(1, sample_poly, 20.0)
        assert len(nvm.leakage_trace) == 2

    def test_leakage_values_positive(self, nvm, sample_poly):
        nvm.write_checkpoint(0, sample_poly, 5.0)
        _, hw = nvm.leakage_trace[0]
        assert hw > 0

    def test_zero_data_leakage_is_zero(self, nvm):
        z = np.zeros(256, dtype=np.int64)
        nvm.write_checkpoint(0, z, 0.0)
        _, hw = nvm.leakage_trace[0]
        assert hw == 0

    def test_timestamps_recorded(self, nvm, sample_poly):
        nvm.write_checkpoint(0, sample_poly, 42.5)
        t, _ = nvm.leakage_trace[0]
        assert t == pytest.approx(42.5)


    def test_times_and_power_values(self, nvm, sample_poly):
        nvm.write_checkpoint(0, sample_poly, 1.0)
        nvm.write_checkpoint(1, sample_poly, 2.0)
        assert nvm.times() == [1.0, 2.0]
        assert len(nvm.power_values()) == 2

    def test_summary_no_data(self, nvm):
        assert "No leakage" in nvm.summary()

    def test_summary_with_data(self, nvm, sample_poly):
        nvm.write_checkpoint(0, sample_poly, 0.0)
        s = nvm.summary()
        assert "samples" in s
        assert "HW-bits" in s
