# Tessera-PQC (Tessera)

## Abstract
**Tessera** is a research simulation framework designed to explore the intersection of **Post-Quantum Cryptography (PQC)** and **Intermittent Computing**. It simulates a battery-free IoT device attempting to perform heavy Lattice-based cryptographic operations (like Kyber/ML-KEM) while powered by unstable energy harvesting sources. The project focuses on "Atomic" cryptography breaking large mathematical operations (NTT, matrix multiplication) into small "tesserae" (tiles) that can be checkpointed to Non-Volatile Memory (NVM) before power failure occurs.

## Installation

### Prerequisites
- Python 3.10+

### Setup
1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```
2. Install the package in editable mode:
   ```bash
   pip install -e .
   ```
3. Verify installation:
   ```bash
   tessera --help
   ```

## Development Plan
1.  **Phase 1: Basic Crypto Core**: Implement `core/math.py` and `core/primitives.py`. Get a basic KeyGen/Encap working without power interruptions.
2.  **Phase 2: Chaos Monkey**: Implement `hardware/power.py`. Create a simulation where power dies randomly.
3.  **Phase 3: Atomic Scheduler**: Implement `scheduler.py`. Break the crypto math into chunks that save state to `hardware/memory.py` to survive Phase 2.
4.  **Phase 4: Leakage**: Instrument `memory.py` to log "power consumption" during writes, simulating side-channel leakage.
