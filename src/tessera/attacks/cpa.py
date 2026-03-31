from typing import Optional, Tuple, List, Callable
import numpy as np

from .base import (
    KeyRecoveryAttack,
    AttackType,
    AttackResult,
    IncrementalAttack,
    pearson_correlation,
)
from ..leakage import LeakageModel, HammingWeightModel, TraceSet


SBOX = np.array([
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
], dtype=np.uint8)


class CPA(KeyRecoveryAttack):
    
    def __init__(self, leakage_model: Optional[LeakageModel] = None,
                 key_byte_indices: Optional[List[int]] = None,
                 num_key_bytes: int = 16,
                 intermediate_func: Optional[Callable] = None):
        if leakage_model is None:
            leakage_model = HammingWeightModel(8)
        super().__init__(leakage_model, key_byte_indices, num_key_bytes)
        self._intermediate_func = intermediate_func or self._default_intermediate
    
    @staticmethod
    def _default_intermediate(plaintext_byte: int, key_guess: int) -> int:
        return SBOX[plaintext_byte ^ key_guess]
    
    @property
    def intermediate_func(self) -> Callable:
        return self._intermediate_func
    
    @intermediate_func.setter
    def intermediate_func(self, func: Callable):
        self._intermediate_func = func
    
    @property
    def attack_type(self) -> AttackType:
        return AttackType.CPA
    
    def attack_byte(self, traces: np.ndarray, 
                    plaintexts: np.ndarray,
                    byte_index: int) -> Tuple[int, float, np.ndarray]:
        num_traces = traces.shape[0]
        num_samples = traces.shape[1]
        
        correlations = np.zeros((256, num_samples))
        
        for key_guess in range(256):
            hypothetical_leakage = np.zeros(num_traces, dtype=np.float64)
            
            for i in range(num_traces):
                intermediate = self._intermediate_func(plaintexts[i, byte_index], key_guess)
                hypothetical_leakage[i] = self._leakage_model(intermediate)
            
            corr = pearson_correlation(hypothetical_leakage, traces)
            correlations[key_guess] = corr.flatten()
        
        self._correlation_matrix = correlations
        
        max_correlations = np.max(np.abs(correlations), axis=1)
        best_key = int(np.argmax(max_correlations))
        best_confidence = float(max_correlations[best_key])
        
        second_best = np.partition(max_correlations, -2)[-2]
        confidence_margin = best_confidence - second_best
        
        return best_key, confidence_margin, correlations[best_key]
    
    def get_correlation_matrix(self) -> Optional[np.ndarray]:
        return self._correlation_matrix


class IncrementalCPA(IncrementalAttack):
    
    def __init__(self, leakage_model: Optional[LeakageModel] = None,
                 batch_size: int = 100,
                 key_byte_indices: Optional[List[int]] = None,
                 num_key_bytes: int = 16,
                 intermediate_func: Optional[Callable] = None):
        if leakage_model is None:
            leakage_model = HammingWeightModel(8)
        super().__init__(leakage_model, batch_size)
        self._key_byte_indices = key_byte_indices or list(range(num_key_bytes))
        self._num_key_bytes = num_key_bytes
        self._intermediate_func = intermediate_func or CPA._default_intermediate
        
        self._sum_traces: Optional[np.ndarray] = None
        self._sum_traces_sq: Optional[np.ndarray] = None
        self._sum_hyp: Optional[np.ndarray] = None
        self._sum_hyp_sq: Optional[np.ndarray] = None
        self._sum_product: Optional[np.ndarray] = None
    
    @property
    def attack_type(self) -> AttackType:
        return AttackType.CPA
    
    def reset(self):
        super().reset()
        self._sum_traces = None
        self._sum_traces_sq = None
        self._sum_hyp = None
        self._sum_hyp_sq = None
        self._sum_product = None
    
    def update(self, traces: np.ndarray,
               plaintexts: Optional[np.ndarray] = None,
               **kwargs) -> Optional[AttackResult]:
        if plaintexts is None:
            raise ValueError("plaintexts must be provided for CPA")
        
        num_traces = traces.shape[0]
        num_samples = traces.shape[1]
        num_bytes = len(self._key_byte_indices)
        
        if self._sum_traces is None:
            self._sum_traces = np.zeros(num_samples, dtype=np.float64)
            self._sum_traces_sq = np.zeros(num_samples, dtype=np.float64)
            self._sum_hyp = np.zeros((num_bytes, 256), dtype=np.float64)
            self._sum_hyp_sq = np.zeros((num_bytes, 256), dtype=np.float64)
            self._sum_product = np.zeros((num_bytes, 256, num_samples), dtype=np.float64)
        
        self._sum_traces += np.sum(traces, axis=0)
        self._sum_traces_sq += np.sum(traces ** 2, axis=0)
        
        for byte_idx_pos, byte_idx in enumerate(self._key_byte_indices):
            for key_guess in range(256):
                hyp_leakage = np.zeros(num_traces, dtype=np.float64)
                for i in range(num_traces):
                    intermediate = self._intermediate_func(plaintexts[i, byte_idx], key_guess)
                    hyp_leakage[i] = self._leakage_model(intermediate)
                
                self._sum_hyp[byte_idx_pos, key_guess] += np.sum(hyp_leakage)
                self._sum_hyp_sq[byte_idx_pos, key_guess] += np.sum(hyp_leakage ** 2)
                self._sum_product[byte_idx_pos, key_guess] += np.dot(hyp_leakage, traces)
        
        self._trace_count += num_traces
        
        return None
    
    def finalize(self) -> AttackResult:
        if self._trace_count < 2:
            raise ValueError("Need at least 2 traces")
        
        n = self._trace_count
        
        mean_traces = self._sum_traces / n
        var_traces = self._sum_traces_sq / n - mean_traces ** 2
        std_traces = np.sqrt(np.maximum(var_traces, 1e-10))
        
        recovered_bytes = {}
        confidences = []
        
        for byte_idx_pos, byte_idx in enumerate(self._key_byte_indices):
            correlations = np.zeros((256, len(self._sum_traces)))
            
            for key_guess in range(256):
                mean_hyp = self._sum_hyp[byte_idx_pos, key_guess] / n
                var_hyp = self._sum_hyp_sq[byte_idx_pos, key_guess] / n - mean_hyp ** 2
                std_hyp = np.sqrt(max(var_hyp, 1e-10))
                
                cov = self._sum_product[byte_idx_pos, key_guess] / n - mean_hyp * mean_traces
                
                correlations[key_guess] = cov / (std_hyp * std_traces + 1e-10)
            
            max_corrs = np.max(np.abs(correlations), axis=1)
            best_key = int(np.argmax(max_corrs))
            
            second_best = np.partition(max_corrs, -2)[-2]
            confidence = max_corrs[best_key] - second_best
            
            recovered_bytes[byte_idx] = best_key
            confidences.append(confidence)
        
        recovered_key = np.zeros(self._num_key_bytes, dtype=np.uint8)
        for idx, key_byte in recovered_bytes.items():
            if idx < self._num_key_bytes:
                recovered_key[idx] = key_byte
        
        self._result = AttackResult(
            attack_type=AttackType.CPA,
            success=True,
            recovered_key=recovered_key,
            recovered_bytes=recovered_bytes,
            confidence=float(np.mean(confidences)),
            num_traces_used=self._trace_count,
            metrics={
                "per_byte_confidence": {i: c for i, c in zip(self._key_byte_indices, confidences)},
            }
        )
        
        return self._result
    
    def run(self, traces: TraceSet, **kwargs) -> AttackResult:
        samples = traces.get_samples_matrix()
        plaintexts = traces.get_plaintexts()
        
        if plaintexts is None:
            raise ValueError("TraceSet must contain plaintexts for CPA")
        
        self.update(samples, plaintexts)
        return self.finalize()


class HigherOrderCPA(KeyRecoveryAttack):
    
    def __init__(self, leakage_model: Optional[LeakageModel] = None,
                 key_byte_indices: Optional[List[int]] = None,
                 num_key_bytes: int = 16,
                 order: int = 2,
                 combining_func: Optional[Callable] = None):
        if leakage_model is None:
            leakage_model = HammingWeightModel(8)
        super().__init__(leakage_model, key_byte_indices, num_key_bytes)
        self._order = order
        self._combining_func = combining_func or self._default_combining
    
    @staticmethod
    def _default_combining(samples: np.ndarray, indices: List[int]) -> np.ndarray:
        result = samples[:, indices[0]]
        for i in indices[1:]:
            result = result * samples[:, i]
        return result
    
    @property
    def order(self) -> int:
        return self._order
    
    @order.setter
    def order(self, value: int):
        if value < 2:
            raise ValueError("order must be >= 2 for higher-order CPA")
        self._order = value
    
    @property
    def attack_type(self) -> AttackType:
        return AttackType.CPA
    
    def preprocess_traces(self, traces: np.ndarray, 
                          poi_indices: List[int]) -> np.ndarray:
        mean = np.mean(traces, axis=0)
        centered = traces - mean
        
        num_traces = traces.shape[0]
        num_combinations = len(poi_indices) ** self._order
        
        processed = np.zeros((num_traces, num_combinations))
        
        idx = 0
        for i, poi in enumerate(poi_indices):
            processed[:, idx] = centered[:, poi] ** self._order
            idx += 1
        
        return processed
    
    def attack_byte(self, traces: np.ndarray,
                    plaintexts: np.ndarray,
                    byte_index: int) -> Tuple[int, float, np.ndarray]:
        cpa = CPA(self._leakage_model)
        return cpa.attack_byte(traces, plaintexts, byte_index)


def run_cpa(traces: TraceSet,
            leakage_model: Optional[LeakageModel] = None,
            key_byte_indices: Optional[List[int]] = None,
            intermediate_func: Optional[Callable] = None) -> AttackResult:
    cpa = CPA(
        leakage_model=leakage_model,
        key_byte_indices=key_byte_indices,
        intermediate_func=intermediate_func,
    )
    return cpa.run(traces)
