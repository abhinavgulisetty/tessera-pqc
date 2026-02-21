# Tessera-PQC

[![CI](https://github.com/your-org/tessera-pqc/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/tessera-pqc/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/tessera-pqc)](https://pypi.org/project/tessera-pqc/)
[![Python](https://img.shields.io/pypi/pyversions/tessera-pqc)](https://pypi.org/project/tessera-pqc/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Tessera** is a research simulation framework for **Post-Quantum Cryptography (PQC)** on **intermittent-power** (battery-free IoT) devices.

It models *Atomic Cryptography* — breaking lattice-based operations (NTT, Kyber KEM) into small checkpointed tiles that survive arbitrary power failures by persisting state to Non-Volatile Memory (NVM) after every layer. Side-channel power leakage is modelled using the Hamming Weight of each NVM write.

---

## Features

- **Baby-Kyber KEM** — full Module-LWE key generation, encapsulation, and decapsulation (k=2, q=3329, n=256, η=2)
- **NTT engine** — Cooley-Tukey DIT forward transform + Gentleman-Sande DIF inverse over ℤ_3329\[X\]/(X²⁵⁶+1)
- **Atomic scheduler** — SimPy discrete-event simulation with exponential on/off power model; checkpoints every NTT layer to NVM
- **Hamming Weight leakage model** — records side-channel power trace on every NVM write
- **Rich terminal demo** — live animated panels showing hardware state, NTT progress, event log, and leakage trace
- **62 tests** across math, KEM, memory, and scheduler

---

## Installation

```bash
pip install tessera-pqc
```

Requires Python ≥ 3.10.

### Development install

```bash
git clone https://github.com/your-org/tessera-pqc.git
cd tessera-pqc
pip install -e ".[dev]"
pytest
```

---

## CLI Usage

```bash
tessera verify          # NTT round-trip correctness (5 tests)
tessera kem             # Baby-Kyber key exchange demo
tessera run             # Atomic NTT simulation with SimPy
tessera demo            # Full animated Rich terminal demo
```

### Example — KEM

```
============================================================
 Tessera — Baby-Kyber KEM Demo
============================================================
[KEM] Generating key pair...
      pk length = 672 bytes
      sk length = 768 bytes
[KEM] Encapsulating...
      ciphertext length = 768 bytes
      shared secret (enc) = 1292eb5807fd564239ffa78ab484e840...
[KEM] Decapsulating...
      shared secret (dec) = 1292eb5807fd564239ffa78ab484e840...
[KEM] SUCCESS — shared secrets match! ✓
```

---

## Architecture

```
tessera-pqc/
├── src/tessera/
│   ├── core/
│   │   ├── math.py          # NTT / inverse-NTT / polynomial ring
│   │   └── primitives.py    # Baby-Kyber KEM (keygen / encaps / decaps)
│   ├── hardware/
│   │   ├── memory.py        # NVM simulator + Hamming Weight leakage model
│   │   └── power.py         # SimPy intermittent-power chaos source
│   ├── scheduler.py         # Atomic tile scheduler with NVM checkpointing
│   ├── cli.py               # CLI entry point
│   └── demo.py              # Rich animated terminal demo
└── tests/                   # 62 pytest tests
```

### Key parameters

| Symbol | Value | Meaning |
|--------|-------|---------|
| n | 256 | Polynomial degree |
| q | 3329 | NTT prime |
| ω | 3061 | Primitive 256th root of unity (mod q) |
| k | 2 | Module rank (Baby-Kyber) |
| η | 2 | CBD noise parameter |
| D_U | 10 bits | Ciphertext u compression |
| D_V | 4 bits | Ciphertext v compression |

---

## Publishing workflow

Releases are published to PyPI automatically via [GitHub Actions OIDC Trusted Publisher](https://docs.pypi.org/trusted-publishers/).  
No API tokens are stored — publishing is triggered by creating a GitHub Release.

---

## License

MIT — see [LICENSE](LICENSE).
