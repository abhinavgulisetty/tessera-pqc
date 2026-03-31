# Tessera-PQC

[![CI](https://github.com/abhinavgulisetty/tessera-pqc/actions/workflows/ci.yml/badge.svg)](https://github.com/abhinavgulisetty/tessera-pqc/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/tessera-pqc)](https://pypi.org/project/tessera-pqc/)
[![Python](https://img.shields.io/pypi/pyversions/tessera-pqc)](https://pypi.org/project/tessera-pqc/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Tessera-PQC** is a research-grade simulation framework for **Post-Quantum Cryptography (PQC)** with comprehensive **side-channel analysis** tools.

It provides complete implementations of NIST PQC standards (Kyber, Dilithium), side-channel attack simulations (CPA, DPA, TVLA, Template), countermeasures (masking, shuffling, hiding), and analysis/export tools for cryptographic research.

---

## Features

### Cryptographic Algorithms
- **Kyber KEM** (512/768/1024) — NIST PQC standard for key encapsulation
- **Dilithium Signatures** (2/3/5) — NIST PQC standard for digital signatures
- **Numba-accelerated NTT** — ~18μs for Kyber, ~27μs for Dilithium

### Side-Channel Analysis
- **Power Analysis Attacks** — CPA, DPA, Higher-Order attacks
- **Leakage Assessment** — TVLA (Fixed vs Random, Semi-Fixed)
- **Template Attacks** — Profiled attacks with POI selection
- **Lattice-Specific Attacks** — NTT coefficient and butterfly attacks

### Countermeasures
- **Masking** — Boolean and Arithmetic masking (arbitrary order)
- **Shuffling** — Secure permutation, Shuffled NTT
- **Hiding** — Noise injection, Dual-rail logic, Jitter

### Analysis & Export
- **Metrics** — SNR, Guessing Entropy, Success Rate, Correlation
- **Visualization** — Matplotlib plots, Plotly interactive charts
- **Export** — CSV, JSON, LaTeX tables, NPZ archives

### Additional
- **401 tests** with comprehensive coverage
- **CLI tools** for quick experiments
- **High-level API** for easy integration
- **Tutorial notebooks** for learning

---

## Installation

```bash
pip install tessera-pqc
```

Requires Python ≥ 3.10.

### Development install

```bash
git clone https://github.com/abhinavgulisetty/tessera-pqc.git
cd tessera-pqc
pip install -e ".[dev,notebook]"
pytest
```

---

## Quick Start

### High-Level API

```python
import tessera as t

# Kyber Key Exchange
pk, sk = t.kyber_keygen("768")
ct, shared_secret1 = t.kyber_encaps(pk, "768")
shared_secret2 = t.kyber_decaps(sk, ct, "768")
assert shared_secret1 == shared_secret2

# Dilithium Signatures
pk, sk = t.dilithium_keygen("3")
signature = t.dilithium_sign(sk, b"Hello, PQC!", "3")
valid = t.dilithium_verify(pk, b"Hello, PQC!", signature, "3")
assert valid

# Side-Channel Attack Simulation
data = t.generate_traces(n_traces=1000, trace_length=32, key_bytes=16, noise_level=0.5)
result = t.run_cpa(data["traces"], data["plaintexts"])
print(f"Recovered: {result['recovered_key']}")
print(f"True key:  {data['key']}")

# TVLA Leakage Assessment
tvla = t.run_tvla(traces_fixed, traces_random)
print(f"Leakage detected: {tvla['leakage_detected']}")
```

### CLI Usage

```bash
# Cryptographic operations
tessera kyber --variant 768              # Kyber-768 KEM demo
tessera dilithium --variant 3            # Dilithium3 signature demo

# Side-channel attacks
tessera attack --type cpa --traces 1000  # CPA attack simulation
tessera attack --type tvla --traces 500  # TVLA leakage assessment

# Benchmarks
tessera benchmark --all                  # Full benchmark suite
tessera benchmark --ntt -n 1000          # NTT performance

# Legacy commands
tessera verify                           # NTT round-trip test
tessera demo                             # Animated terminal demo
```

---

## Architecture

```
tessera-pqc/
├── src/tessera/
│   ├── __init__.py              # High-level API exports
│   ├── api.py                   # Facade API functions
│   ├── cli.py                   # CLI entry point
│   ├── config.py                # YAML configuration
│   │
│   ├── core/
│   │   ├── math.py              # Polynomial ring operations
│   │   ├── math_fast.py         # Numba-accelerated NTT
│   │   └── sampling.py          # Cryptographic sampling
│   │
│   ├── algorithms/
│   │   ├── kyber/               # Kyber-512/768/1024
│   │   └── dilithium/           # Dilithium2/3/5
│   │
│   ├── leakage/
│   │   ├── models.py            # HW, HD, Identity, Custom models
│   │   ├── trace.py             # Trace generation & storage
│   │   ├── noise.py             # Noise models (electronic, environmental)
│   │   └── preprocessing.py     # Alignment, filtering, compression
│   │
│   ├── attacks/
│   │   ├── cpa.py               # Correlation Power Analysis
│   │   ├── dpa.py               # Differential Power Analysis
│   │   ├── ttest.py             # TVLA (Welch's t-test)
│   │   ├── template.py          # Template attacks
│   │   └── lattice_sca.py       # Lattice-specific attacks
│   │
│   ├── countermeasures/
│   │   ├── masking.py           # Boolean/Arithmetic masking
│   │   ├── shuffling.py         # Secure permutation, Shuffled NTT
│   │   └── hiding.py            # Noise injection, Dual-rail logic
│   │
│   ├── analysis/
│   │   ├── metrics.py           # SNR, TVLA, Guessing Entropy
│   │   ├── plotting.py          # Matplotlib visualizations
│   │   ├── interactive.py       # Plotly interactive charts
│   │   └── export.py            # CSV, JSON, LaTeX, NPZ
│   │
│   └── hardware/                # Intermittent computing simulation
│       ├── memory.py            # NVM with leakage model
│       └── power.py             # Power failure simulation
│
├── notebooks/
│   ├── 01_pqc_basics.ipynb           # Kyber & Dilithium tutorial
│   ├── 02_side_channel_analysis.ipynb # CPA, DPA, TVLA tutorial
│   └── 03_countermeasures.ipynb       # Masking, shuffling tutorial
│
├── tests/                       # 401 pytest tests
└── docs/                        # Interactive visualization website
```

---

## Algorithm Parameters

### Kyber (ML-KEM)

| Variant | NIST Level | Public Key | Secret Key | Ciphertext |
|---------|------------|------------|------------|------------|
| Kyber-512 | 1 (~AES-128) | 800 bytes | 1,632 bytes | 768 bytes |
| Kyber-768 | 3 (~AES-192) | 1,184 bytes | 2,400 bytes | 1,088 bytes |
| Kyber-1024 | 5 (~AES-256) | 1,568 bytes | 3,168 bytes | 1,568 bytes |

### Dilithium (ML-DSA)

| Variant | NIST Level | Public Key | Secret Key | Signature |
|---------|------------|------------|------------|-----------|
| Dilithium2 | 2 (~AES-128) | 1,312 bytes | 2,560 bytes | 2,420 bytes |
| Dilithium3 | 3 (~AES-192) | 1,952 bytes | 4,032 bytes | 3,293 bytes |
| Dilithium5 | 5 (~AES-256) | 2,592 bytes | 4,896 bytes | 4,595 bytes |

### NTT Parameters

| Parameter | Kyber | Dilithium |
|-----------|-------|-----------|
| n | 256 | 256 |
| q | 3,329 | 8,380,417 |
| Root of unity | 17 | 1,753 |

---

## Tutorials

### Jupyter Notebooks

1. **[PQC Basics](notebooks/01_pqc_basics.ipynb)** — Kyber KEM and Dilithium signatures
2. **[Side-Channel Analysis](notebooks/02_side_channel_analysis.ipynb)** — CPA, DPA, TVLA, SNR
3. **[Countermeasures](notebooks/03_countermeasures.ipynb)** — Masking, shuffling, hiding

### Run notebooks

```bash
pip install tessera-pqc[notebook]
jupyter notebook notebooks/
```

---

## Performance

Benchmarks on typical hardware (results vary):

| Operation | Time |
|-----------|------|
| NTT (Kyber) | ~18 μs |
| NTT (Dilithium) | ~27 μs |
| Kyber-768 KeyGen | ~3 ms |
| Kyber-768 Encaps | ~4 ms |
| Kyber-768 Decaps | ~5 ms |
| Dilithium3 KeyGen | ~15 ms |
| Dilithium3 Sign | ~50 ms |
| Dilithium3 Verify | ~15 ms |

Run your own benchmarks:

```bash
tessera benchmark --all -n 100
```

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass: `pytest`
5. Submit a pull request

---

## Citation

If you use Tessera-PQC in your research, please cite:

```bibtex
@software{tessera_pqc,
  title = {Tessera-PQC: Post-Quantum Cryptography Research Framework},
  author = {Gulisetty, Abhinav},
  year = {2024},
  url = {https://github.com/abhinavgulisetty/tessera-pqc}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
