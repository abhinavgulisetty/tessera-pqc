"""
Atomic Task Scheduler  –  the heart of Tessera.

Concept
-------
Intermittent devices (battery-free IoT) lose power randomly and repeatedly.
A naïve NTT would restart from scratch after every failure.  Tessera instead
divides the NTT computation into *Tesserae* (small tiles = one butterfly layer)
and checkpoints after each tile to Non-Volatile Memory (NVM).  On reboot the
device restores the last good checkpoint and continues — achieving **forward
progress** in spite of frequent power failures.

Atomicity guarantee
-------------------
- One tessera (= one NTT layer) is the *atomic unit* of computation.
- Before starting a tessera the device checks power and blocks if needed.
- After completing a tessera it writes to NVM *before* advancing the counter.
- Worst case: the last tessera is re-executed (idempotent butterfly stages → OK).
"""
from __future__ import annotations

import numpy as np
import simpy

from .core.math import PolynomialRing
from .hardware.power import PowerSource
from .hardware.memory import NonVolatileMemory

_COMPUTE_COST_PER_LAYER   = 10
_CHECKPOINT_COST_PER_LAYER = 5


class AtomicTaskScheduler:
    """
    Orchestrates NTT computation as a sequence of interruptible atomic tiles.

    Each NTT layer is treated as one *tessera*:
      1.  Wait for power (if off).
      2.  Restore state from NVM (if waking from a failure).
      3.  Compute butterfly stage  (simulate cost via SimPy timeout).
      4.  Write checkpoint to NVM  (records leakage trace).
      5.  Advance layer counter.

    Statistics collected:
      - ``completed_layers``  : number of layers finished successfully.
      - ``power_failures``    : number of interruptions encountered.
      - ``restores``          : number of checkpoint restores performed.
    """

    _STATE_ADDR   = 0xFFFF
    _DATA_BASE    = 0x0000

    def __init__(self,
                 env:   simpy.Environment,
                 power: PowerSource,
                 nvm:   NonVolatileMemory,
                 ring:  PolynomialRing | None = None):
        self.env   = env
        self.power = power
        self.nvm   = nvm
        self.ring  = ring or PolynomialRing()

        self.current_step   = 0
        self.completed_layers: int = 0
        self.power_failures:   int = 0
        self.restores:         int = 0


    def run_atomic_ntt(self, poly_data: np.ndarray | None = None):
        """
        SimPy generator: compute a full 8-layer NTT atomically.

        The method is a Python generator (contains ``yield`` statements) so
        SimPy can pause it at any ``yield env.timeout(...)`` or
        ``yield power.power_restored`` and resume later.

        Args:
            poly_data: Initial polynomial coefficients (length 256, mod q).
                       If *None* a random polynomial is generated.
        """
        ring = self.ring
        n    = ring.n
        total_layers = n.bit_length() - 1

        if poly_data is None:
            poly_data = np.random.randint(0, ring.q, n, dtype=np.int64)

        saved_step = self.nvm.read_checkpoint(self._STATE_ADDR)
        if saved_step is not None:
            self.current_step = int(saved_step[0])
            saved_data = self.nvm.read_checkpoint(
                self._DATA_BASE + self.current_step - 1
            )
            if saved_data is not None:
                poly_data = saved_data
                self.restores += 1
                print(f"[Scheduler] Restored from NVM at layer {self.current_step} "
                      f"(t={self.env.now:.1f})")

        print(f"[Scheduler] Starting Atomic NTT at t={self.env.now:.1f} "
              f"| Layers: {total_layers} | Starting from layer {self.current_step}")

        from .core.math import _bit_reverse_copy
        working = _bit_reverse_copy(poly_data.astype(np.int64), n)

        q      = ring.q
        omega  = ring._omega
        layers = []
        length = 2
        while length <= n:
            half  = length // 2
            w_len = pow(omega, n // length, q)
            layers.append((length, half, w_len))
            length <<= 1

        for layer_idx, (length, half, w_len) in enumerate(layers):

            if layer_idx < self.current_step:
                continue

            if not self.power.is_powered:
                self.power_failures += 1
                print(f"[Scheduler] Power FAILURE before layer {layer_idx} "
                      f"(t={self.env.now:.1f}) — waiting...")
                yield self.power.power_restored

                if layer_idx > 0:
                    saved = self.nvm.read_checkpoint(self._DATA_BASE + layer_idx - 1)
                    if saved is not None:
                        working = saved
                        self.restores += 1
                print(f"[Scheduler] Resuming at layer {layer_idx} "
                      f"(t={self.env.now:.1f})")

            yield self.env.timeout(_COMPUTE_COST_PER_LAYER)

            for start in range(0, n, length):
                wj = 1
                for j in range(half):
                    u = int(working[start + j])
                    v = int(working[start + j + half]) * wj % q
                    working[start + j]        = (u + v) % q
                    working[start + j + half] = (u - v) % q
                    wj = wj * w_len % q

            yield self.env.timeout(_CHECKPOINT_COST_PER_LAYER)
            self.nvm.write_checkpoint(
                self._DATA_BASE + layer_idx, working, self.env.now
            )
            self.nvm.write_checkpoint(
                self._STATE_ADDR,
                np.array([layer_idx + 1], dtype=np.int64),
                self.env.now
            )

            self.current_step = layer_idx + 1
            self.completed_layers += 1
            print(f"[Scheduler] Layer {layer_idx + 1}/{total_layers} complete "
                  f"(t={self.env.now:.1f})")

        print(f"[Scheduler] NTT COMPLETE at t={self.env.now:.1f} | "
              f"Failures: {self.power_failures} | Restores: {self.restores}")
        return working
