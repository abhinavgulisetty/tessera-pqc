# Tessera-PQC (Tessera)

## Abstract
**Tessera** is a research simulation framework designed to explore the intersection of **Post-Quantum Cryptography (PQC)** and **Intermittent Computing**. It simulates a battery-free IoT device attempting to perform heavy Lattice-based cryptographic operations (like Kyber/ML-KEM) while powered by unstable energy harvesting sources. The project focuses on "Atomic" cryptography—breaking large mathematical operations (NTT, matrix multiplication) into small "tesserae" (tiles) that can be checkpointed to Non-Volatile Memory (NVM) before power failure occurs.

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

## The "Self-Taught" Curriculum
To implement the TODOs in this codebase, you will need to research:
1.  **Number Theoretic Transform (NTT)**: Fast polynomial multiplication over finite fields. *Key to Lattice crypto speed.*
2.  **Intermittent Computing**: Concepts like forward progress, idempotency, and state checkpointing.
3.  **Side-Channel Power Analysis**: How writing to memory (NVM) leaks Hamming Weight/Distance information that adversaries can measure.

## Development Plan
1.  **Phase 1: Basic Crypto Core**: Implement `core/math.py` and `core/primitives.py`. Get a basic KeyGen/Encap working without power interruptions.
2.  **Phase 2: Chaos Monkey**: Implement `hardware/power.py`. Create a simulation where power dies randomly.
3.  **Phase 3: Atomic Scheduler**: Implement `scheduler.py`. Break the crypto math into chunks that save state to `hardware/memory.py` to survive Phase 2.
4.  **Phase 4: Leakage**: Instrument `memory.py` to log "power consumption" during writes, simulating side-channel leakage.
