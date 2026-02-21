# Tessera-PQC (Tessera)

## Abstract
**Tessera** is a research simulation framework for exploring **Post-Quantum Cryptography (PQC)** on **Intermittent Computing** devices (battery-free IoT). It models "Atomic" cryptography—breaking large Lattice-based math operations (like NTT) into small tiles ("tesserae") that can be checkpointed to Non-Volatile Memory (NVM) to survive frequent power failures.

## Installation

### Prerequisites
- Python 3.10 (Strictly required for dependency compatibility)

### Quick Setup (Windows)
Run the automated setup script:
```cmd
setup_env.bat
```

### Manual Setup
1. Create a virtual environment using Python 3.10:
   ```bash
   py -3.10 -m venv venv
   # Or depending on your system: python3.10 -m venv venv
   ```
2. Activate the environment:
   - **Windows:** `venv\Scripts\activate`
   - **Linux/Mac:** `source venv/bin/activate`
3. Install the package in editable mode:
   ```bash
   pip install -e .
   ```
4. Verify installation:
   ```bash
   tessera run --help
   ```

## Curriculum & Roadmap
To complete the `TODO`s in the code, research these topics:
1. **Number Theoretic Transform (NTT):** Efficient polynomial multiplication in rings.
2. **Intermittent Computing:** Checkpointing, forward progress, and idempotency.
3. **Side-Channel Analysis:** Leakage modeling (Hamming Weight) during NVM writes.

**Development Phases:**
1. **Crypto Core:** Implement `math.py` and `primitives.py`.
2. **Power Simulation:** Refine `power.py` events.
3. **Atomic Scheduler:** Implement logic in `scheduler.py` to handle interruptions.
4. **Leakage Analysis:** Instrument `memory.py` to trace power side-channels.
