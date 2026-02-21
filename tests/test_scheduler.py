"""
Tests for scheduler.py — AtomicTaskScheduler.
"""
import numpy as np
import simpy
import pytest

from tessera.core.math import PolynomialRing
from tessera.hardware.power import PowerSource
from tessera.hardware.memory import NonVolatileMemory
from tessera.scheduler import AtomicTaskScheduler


def _run(duration=2000, on_avg=500, off_avg=10, seed=None):
    """Helper: run a simulation and return (scheduler, nvm)."""
    if seed is not None:
        import random
        random.seed(seed)
    env   = simpy.Environment()
    ring  = PolynomialRing()
    power = PowerSource(env, on_time_avg=on_avg, off_time_avg=off_avg)
    nvm   = NonVolatileMemory()
    sched = AtomicTaskScheduler(env, power, nvm, ring)
    poly  = np.random.randint(0, ring.q, ring.n, dtype=np.int64)
    env.process(sched.run_atomic_ntt(poly_data=poly))
    env.run(until=duration)
    return sched, nvm


class TestSchedulerBasic:
    def test_completes_all_layers_no_failures(self):
        """With effectively infinite power-on time, all 8 layers must finish."""
        sched, _ = _run(duration=5000, on_avg=9999, off_avg=1, seed=7)
        assert sched.completed_layers == 8

    def test_checkpoints_written_to_nvm(self):
        sched, nvm = _run(duration=5000, on_avg=9999, off_avg=1, seed=8)
        assert len(nvm.leakage_trace) == 16

    def test_leakage_trace_non_empty_after_run(self):
        _, nvm = _run(duration=5000, on_avg=9999, off_avg=1, seed=9)
        assert len(nvm.leakage_trace) > 0

    def test_nvm_contains_final_layer(self):
        sched, nvm = _run(duration=5000, on_avg=9999, off_avg=1, seed=10)
        last = nvm.read_checkpoint(sched._DATA_BASE + 7)
        assert last is not None
        assert len(last) == 256


class TestSchedulerWithFailures:
    def test_survives_power_failures(self):
        """Even with frequent power failures the NTT must eventually finish."""
        sched, _ = _run(duration=10_000, on_avg=200, off_avg=100, seed=1)
        assert sched.completed_layers == 8

    def test_failures_and_restores_counted(self):
        sched, _ = _run(duration=10_000, on_avg=50, off_avg=80, seed=2)
        assert sched.power_failures >= 0
        assert sched.restores <= sched.power_failures + 1

    def test_all_layers_correct_with_interruptions(self):
        """With failures enabled the final NTT output must still be valid (in Z_q)."""
        import random
        random.seed(3)
        env   = simpy.Environment()
        ring  = PolynomialRing()
        power = PowerSource(env, on_time_avg=30, off_time_avg=20)
        nvm   = NonVolatileMemory()
        sched = AtomicTaskScheduler(env, power, nvm, ring)
        poly  = np.random.randint(0, ring.q, ring.n, dtype=np.int64)
        env.process(sched.run_atomic_ntt(poly_data=poly))
        env.run(until=50_000)

        assert sched.completed_layers == 8
        final = nvm.read_checkpoint(sched._DATA_BASE + 7)
        assert np.all(final >= 0) and np.all(final < ring.q)


class TestSchedulerRestore:
    def test_restore_from_existing_checkpoint(self):
        """Pre-populate NVM at layer 4 and verify scheduler resumes from there."""
        env   = simpy.Environment()
        ring  = PolynomialRing()
        power = PowerSource(env, on_time_avg=9999, off_time_avg=1)
        nvm   = NonVolatileMemory()
        sched = AtomicTaskScheduler(env, power, nvm, ring)

        fake_state = np.random.randint(0, ring.q, ring.n, dtype=np.int64)
        nvm.write_checkpoint(sched._DATA_BASE + 3, fake_state, 0.0)
        nvm.write_checkpoint(sched._STATE_ADDR, np.array([4], dtype=np.int64), 0.0)

        poly = np.random.randint(0, ring.q, ring.n, dtype=np.int64)
        env.process(sched.run_atomic_ntt(poly_data=poly))
        env.run(until=5000)

        assert sched.restores >= 1
        assert sched.completed_layers == 4
