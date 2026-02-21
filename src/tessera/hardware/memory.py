"""
Non-Volatile Memory (NVM) simulator with power side-channel instrumentation.

Hardware background
-------------------
FRAM / MRAM cells consume current proportional to the number of bit transitions
(switching 0→1 or 1→0).  A widely-used approximation is the **Hamming Weight**
(HW) model: power ≈ ΣᵢHW(word_i) where HW counts the '1' bits in a word.

- Writing 0xFF (HW=8) draws more current than 0x00 (HW=0).
- Attackers correlate HW(checkpoint data) with secret-key hypotheses → DPA.
"""
from __future__ import annotations
import numpy as np


def _hamming_weight_array(arr: np.ndarray) -> int:
    """
    Compute total Hamming Weight (number of 1-bits) across all elements.

    Each coefficient is treated as a 16-bit unsigned integer for the model
    (q=3329 fits in 12 bits, so 16-bit is a reasonable hardware word size).

    Hamming Weight power model:
        P(write) ≈ Σ_{i=0}^{N-1} HW(coeff_i)
    """
    words = arr.astype(np.uint16).flatten()
    total = 0
    for w in words:
        total += bin(int(w)).count('1')
    return total


class NonVolatileMemory:
    """
    Simulates FRAM/MRAM.
    Tracks 'leakage' (Hamming Weight) whenever a checkpoint is written,
    building a power side-channel trace that can be analysed for DPA.
    """

    def __init__(self):
        self.storage: dict = {}
        self.leakage_trace: list[tuple[float, int]] = []


    def write_checkpoint(self, address: int, data: np.ndarray, time: float) -> None:
        """
        Write *data* to NVM at logical *address* and record leakage.

        The Hamming Weight of every 16-bit word written is summed and appended
        to ``self.leakage_trace`` as ``(simulation_time, power_proxy)``.

        Args:
            address: Logical ID for the checkpoint (e.g., NTT layer index).
            data   : NumPy array holding intermediate cryptographic state.
            time   : Current simulation clock (SimPy environment time).
        """
        self.storage[address] = data.copy()

        hw = _hamming_weight_array(data)
        self.leakage_trace.append((time, hw))


    def read_checkpoint(self, address: int) -> np.ndarray | None:
        """
        Retrieve previously checkpointed data.  Returns ``None`` if not found.
        """
        return self.storage.get(address, None)


    def times(self) -> list[float]:
        """Return list of timestamps from the leakage trace."""
        return [t for t, _ in self.leakage_trace]

    def power_values(self) -> list[int]:
        """Return list of Hamming-Weight power proxy values."""
        return [p for _, p in self.leakage_trace]

    def summary(self) -> str:
        """One-line summary of leakage statistics."""
        vals = self.power_values()
        if not vals:
            return "No leakage recorded."
        return (
            f"Leakage trace: {len(vals)} samples | "
            f"min={min(vals)} | max={max(vals)} | "
            f"mean={sum(vals)/len(vals):.1f} HW-bits"
        )
