from __future__ import annotations

from typing import Tuple, Optional, Dict, Any, List, Union
from pathlib import Path
import numpy as np


def kyber_keygen(variant: str = "768") -> Tuple[bytes, bytes]:
    from tessera.algorithms.kyber import kyber512, kyber768, kyber1024
    variants = {"512": kyber512, "768": kyber768, "1024": kyber1024}
    if variant not in variants:
        raise ValueError(f"Unknown variant: {variant}. Choose from: 512, 768, 1024")
    kem = variants[variant]()
    return kem.keygen()


def kyber_encaps(public_key: bytes, variant: str = "768") -> Tuple[bytes, bytes]:
    from tessera.algorithms.kyber import kyber512, kyber768, kyber1024
    variants = {"512": kyber512, "768": kyber768, "1024": kyber1024}
    if variant not in variants:
        raise ValueError(f"Unknown variant: {variant}. Choose from: 512, 768, 1024")
    kem = variants[variant]()
    return kem.encaps(public_key)


def kyber_decaps(secret_key: bytes, ciphertext: bytes, variant: str = "768") -> bytes:
    from tessera.algorithms.kyber import kyber512, kyber768, kyber1024
    variants = {"512": kyber512, "768": kyber768, "1024": kyber1024}
    if variant not in variants:
        raise ValueError(f"Unknown variant: {variant}. Choose from: 512, 768, 1024")
    kem = variants[variant]()
    return kem.decaps(secret_key, ciphertext)


def dilithium_keygen(variant: str = "3") -> Tuple[bytes, bytes]:
    from tessera.algorithms.dilithium import dilithium2, dilithium3, dilithium5
    variants = {"2": dilithium2, "3": dilithium3, "5": dilithium5}
    if variant not in variants:
        raise ValueError(f"Unknown variant: {variant}. Choose from: 2, 3, 5")
    sig = variants[variant]()
    return sig.keygen()


def dilithium_sign(secret_key: bytes, message: bytes, variant: str = "3") -> bytes:
    from tessera.algorithms.dilithium import dilithium2, dilithium3, dilithium5
    variants = {"2": dilithium2, "3": dilithium3, "5": dilithium5}
    if variant not in variants:
        raise ValueError(f"Unknown variant: {variant}. Choose from: 2, 3, 5")
    sig = variants[variant]()
    return sig.sign(secret_key, message)


def dilithium_verify(public_key: bytes, message: bytes, signature: bytes, variant: str = "3") -> bool:
    from tessera.algorithms.dilithium import dilithium2, dilithium3, dilithium5
    variants = {"2": dilithium2, "3": dilithium3, "5": dilithium5}
    if variant not in variants:
        raise ValueError(f"Unknown variant: {variant}. Choose from: 2, 3, 5")
    sig = variants[variant]()
    return sig.verify(public_key, message, signature)


def generate_traces(
    n_traces: int,
    trace_length: int,
    key: Optional[np.ndarray] = None,
    key_bytes: int = 16,
    noise_level: float = 1.0,
    leakage_model: str = "hw",
    seed: Optional[int] = None
) -> Dict[str, np.ndarray]:
    from tessera.leakage import HammingWeightModel, HammingDistanceModel
    
    rng = np.random.default_rng(seed)
    
    if key is None:
        key = rng.integers(0, 256, key_bytes, dtype=np.uint8)
    else:
        key = np.asarray(key, dtype=np.uint8)
        key_bytes = len(key)
    
    if leakage_model == "hw":
        model = HammingWeightModel(8)
    elif leakage_model == "hd":
        model = HammingDistanceModel(8)
    else:
        model = HammingWeightModel(8)
    
    traces = []
    plaintexts = []
    
    for _ in range(n_traces):
        pt = rng.integers(0, 256, key_bytes, dtype=np.uint8)
        plaintexts.append(pt)
        
        intermediate = pt ^ key
        leakage = np.array([model(int(v)) for v in intermediate], dtype=np.float64)
        
        trace = np.zeros(trace_length, dtype=np.float64)
        trace[:min(key_bytes, trace_length)] = leakage[:min(key_bytes, trace_length)]
        trace += rng.standard_normal(trace_length) * noise_level
        traces.append(trace)
    
    return {
        "traces": np.array(traces),
        "plaintexts": np.array(plaintexts),
        "key": key,
    }


def run_cpa(
    traces: np.ndarray,
    plaintexts: np.ndarray,
    key_bytes: Optional[List[int]] = None,
    intermediate_func: Optional[Any] = None
) -> Dict[str, Any]:
    from tessera.attacks import CPA
    
    if key_bytes is None:
        key_bytes = list(range(plaintexts.shape[1]))
    
    if intermediate_func is None:
        intermediate_func = lambda pt, kg: pt ^ kg
    
    attack = CPA(intermediate_func=intermediate_func)
    
    recovered_key = []
    confidences = []
    
    for byte_idx in key_bytes:
        best_key, confidence, _ = attack.attack_byte(traces, plaintexts, byte_idx)
        recovered_key.append(best_key)
        confidences.append(confidence)
    
    return {
        "recovered_key": np.array(recovered_key, dtype=np.uint8),
        "confidences": np.array(confidences),
        "correlation_matrix": attack.get_correlation_matrix(),
    }


def run_dpa(
    traces: np.ndarray,
    plaintexts: np.ndarray,
    key_bytes: Optional[List[int]] = None,
    intermediate_func: Optional[Any] = None
) -> Dict[str, Any]:
    from tessera.attacks import DPA
    
    if key_bytes is None:
        key_bytes = list(range(plaintexts.shape[1]))
    
    if intermediate_func is None:
        intermediate_func = lambda pt, kg: pt ^ kg
    
    attack = DPA(intermediate_func=intermediate_func)
    
    recovered_key = []
    confidences = []
    
    for byte_idx in key_bytes:
        best_key, confidence, _ = attack.attack_byte(traces, plaintexts, byte_idx)
        recovered_key.append(best_key)
        confidences.append(confidence)
    
    return {
        "recovered_key": np.array(recovered_key, dtype=np.uint8),
        "confidences": np.array(confidences),
    }


def run_tvla(
    traces_fixed: np.ndarray,
    traces_random: np.ndarray,
    order: int = 1,
    confidence_level: float = 0.99999
) -> Dict[str, Any]:
    from tessera.attacks import FixedVsRandomTVLA
    
    tvla = FixedVsRandomTVLA(confidence_level=confidence_level, order=order)
    result = tvla.assess(traces_fixed, traces_random)
    
    return {
        "leakage_detected": result.leakage_detected,
        "max_t_value": result.max_statistic,
        "threshold": result.threshold,
        "t_values": result.statistic_per_sample,
        "poi_indices": result.poi_indices,
    }


def compute_snr(traces: np.ndarray, labels: np.ndarray) -> np.ndarray:
    from tessera.analysis import SNRCalculator
    calc = SNRCalculator()
    return calc.compute_snr(traces, labels)


def compute_guessing_entropy(
    correct_key: int,
    key_scores: np.ndarray,
    n_traces_list: Optional[List[int]] = None
) -> Dict[str, Any]:
    from tessera.analysis import GuessingEntropy
    ge = GuessingEntropy()
    
    if n_traces_list is None:
        result = ge.compute_rank(correct_key, key_scores)
        return {"rank": result, "guessing_entropy": np.log2(result + 1)}
    
    ranks = []
    for n in n_traces_list:
        rank = ge.compute_rank(correct_key, key_scores[:n] if len(key_scores.shape) > 1 else key_scores)
        ranks.append(rank)
    
    return {
        "n_traces": n_traces_list,
        "ranks": ranks,
        "guessing_entropy": [np.log2(r + 1) for r in ranks],
    }


def apply_masking(
    data: np.ndarray,
    num_shares: int = 2,
    mask_type: str = "boolean",
    modulus: int = 3329
) -> Dict[str, Any]:
    from tessera.countermeasures import BooleanMasking, ArithmeticMasking
    
    if mask_type == "boolean":
        masking = BooleanMasking()
    elif mask_type == "arithmetic":
        masking = ArithmeticMasking(modulus=modulus)
    else:
        raise ValueError(f"Unknown mask_type: {mask_type}")
    
    masked = masking.mask(data, num_shares=num_shares)
    
    return {
        "shares": masked.shares,
        "original": data,
        "masked_value": masked,
    }


def apply_shuffling(
    operations: List[Any],
    seed: Optional[int] = None
) -> List[Any]:
    from tessera.countermeasures import SecurePermutation
    perm = SecurePermutation(len(operations), seed=seed)
    indices = perm.generate()
    return [operations[i] for i in indices]


def export_traces_csv(traces: np.ndarray, path: Union[str, Path]):
    from tessera.analysis import CSVExporter
    exporter = CSVExporter()
    exporter.export_traces(traces, path)


def export_results_json(results: Dict[str, Any], path: Union[str, Path]):
    from tessera.analysis import JSONExporter
    exporter = JSONExporter()
    exporter.export_stats(results, path)


def export_traces_npz(
    path: Union[str, Path],
    traces: np.ndarray,
    plaintexts: Optional[np.ndarray] = None,
    key: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    **kwargs
):
    data = {"traces": traces}
    if plaintexts is not None:
        data["plaintexts"] = plaintexts
    if key is not None:
        data["key"] = key
    if labels is not None:
        data["labels"] = labels
    data.update(kwargs)
    np.savez(path, **data)


def load_traces_npz(path: Union[str, Path]) -> Dict[str, np.ndarray]:
    data = np.load(path)
    return {key: data[key] for key in data.files}


def benchmark_kyber(variant: str = "768", iterations: int = 10) -> Dict[str, float]:
    import time
    from tessera.algorithms.kyber import kyber512, kyber768, kyber1024
    
    variants = {"512": kyber512, "768": kyber768, "1024": kyber1024}
    kem = variants[variant]()
    
    times_kg, times_enc, times_dec = [], [], []
    
    for _ in range(iterations):
        t0 = time.perf_counter()
        pk, sk = kem.keygen()
        times_kg.append(time.perf_counter() - t0)
        
        t0 = time.perf_counter()
        ct, ss = kem.encaps(pk)
        times_enc.append(time.perf_counter() - t0)
        
        t0 = time.perf_counter()
        kem.decaps(sk, ct)
        times_dec.append(time.perf_counter() - t0)
    
    return {
        "keygen_ms": np.mean(times_kg) * 1000,
        "keygen_std_ms": np.std(times_kg) * 1000,
        "encaps_ms": np.mean(times_enc) * 1000,
        "encaps_std_ms": np.std(times_enc) * 1000,
        "decaps_ms": np.mean(times_dec) * 1000,
        "decaps_std_ms": np.std(times_dec) * 1000,
    }


def benchmark_dilithium(variant: str = "3", iterations: int = 10, message: bytes = b"test") -> Dict[str, float]:
    import time
    from tessera.algorithms.dilithium import dilithium2, dilithium3, dilithium5
    
    variants = {"2": dilithium2, "3": dilithium3, "5": dilithium5}
    sig = variants[variant]()
    
    times_kg, times_sign, times_ver = [], [], []
    
    for _ in range(iterations):
        t0 = time.perf_counter()
        pk, sk = sig.keygen()
        times_kg.append(time.perf_counter() - t0)
        
        t0 = time.perf_counter()
        signature = sig.sign(sk, message)
        times_sign.append(time.perf_counter() - t0)
        
        t0 = time.perf_counter()
        sig.verify(pk, message, signature)
        times_ver.append(time.perf_counter() - t0)
    
    return {
        "keygen_ms": np.mean(times_kg) * 1000,
        "keygen_std_ms": np.std(times_kg) * 1000,
        "sign_ms": np.mean(times_sign) * 1000,
        "sign_std_ms": np.std(times_sign) * 1000,
        "verify_ms": np.mean(times_ver) * 1000,
        "verify_std_ms": np.std(times_ver) * 1000,
    }


__all__ = [
    "kyber_keygen",
    "kyber_encaps",
    "kyber_decaps",
    "dilithium_keygen",
    "dilithium_sign",
    "dilithium_verify",
    "generate_traces",
    "run_cpa",
    "run_dpa",
    "run_tvla",
    "compute_snr",
    "compute_guessing_entropy",
    "apply_masking",
    "apply_shuffling",
    "export_traces_csv",
    "export_results_json",
    "export_traces_npz",
    "load_traces_npz",
    "benchmark_kyber",
    "benchmark_dilithium",
]
