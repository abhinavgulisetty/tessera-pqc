from typing import Optional, Tuple, List, Callable
import numpy as np

from .base import (
    KeyRecoveryAttack,
    AttackType,
    AttackResult,
    IncrementalAttack,
    differential_means,
)
from .cpa import SBOX
from ..leakage import LeakageModel, HammingWeightModel, TraceSet


class DPA(KeyRecoveryAttack):
    
    def __init__(self, leakage_model: Optional[LeakageModel] = None,
                 key_byte_indices: Optional[List[int]] = None,
                 num_key_bytes: int = 16,
                 target_bit: int = 0,
                 intermediate_func: Optional[Callable] = None):
        if leakage_model is None:
            leakage_model = HammingWeightModel(8)
        super().__init__(leakage_model, key_byte_indices, num_key_bytes)
        self._target_bit = target_bit
        self._intermediate_func = intermediate_func or self._default_intermediate
    
    @staticmethod
    def _default_intermediate(plaintext_byte: int, key_guess: int) -> int:
        return SBOX[plaintext_byte ^ key_guess]
    
    @property
    def target_bit(self) -> int:
        return self._target_bit
    
    @target_bit.setter
    def target_bit(self, value: int):
        if not 0 <= value <= 7:
            raise ValueError("target_bit must be between 0 and 7")
        self._target_bit = value
    
    @property
    def intermediate_func(self) -> Callable:
        return self._intermediate_func
    
    @intermediate_func.setter
    def intermediate_func(self, func: Callable):
        self._intermediate_func = func
    
    @property
    def attack_type(self) -> AttackType:
        return AttackType.DPA
    
    def attack_byte(self, traces: np.ndarray,
                    plaintexts: np.ndarray,
                    byte_index: int) -> Tuple[int, float, np.ndarray]:
        num_traces = traces.shape[0]
        
        differentials = np.zeros((256, traces.shape[1]))
        
        for key_guess in range(256):
            group0_mask = np.zeros(num_traces, dtype=bool)
            group1_mask = np.zeros(num_traces, dtype=bool)
            
            for i in range(num_traces):
                intermediate = self._intermediate_func(plaintexts[i, byte_index], key_guess)
                bit_value = (intermediate >> self._target_bit) & 1
                if bit_value == 0:
                    group0_mask[i] = True
                else:
                    group1_mask[i] = True
            
            if np.sum(group0_mask) > 0 and np.sum(group1_mask) > 0:
                diff = differential_means(traces[group1_mask], traces[group0_mask])
                differentials[key_guess] = diff
        
        self._correlation_matrix = differentials
        
        max_differentials = np.max(np.abs(differentials), axis=1)
        best_key = int(np.argmax(max_differentials))
        best_magnitude = float(max_differentials[best_key])
        
        second_best = np.partition(max_differentials, -2)[-2]
        confidence_margin = best_magnitude - second_best
        
        return best_key, confidence_margin, differentials[best_key]


class MultiBitDPA(KeyRecoveryAttack):
    
    def __init__(self, leakage_model: Optional[LeakageModel] = None,
                 key_byte_indices: Optional[List[int]] = None,
                 num_key_bytes: int = 16,
                 target_bits: Optional[List[int]] = None,
                 intermediate_func: Optional[Callable] = None):
        if leakage_model is None:
            leakage_model = HammingWeightModel(8)
        super().__init__(leakage_model, key_byte_indices, num_key_bytes)
        self._target_bits = target_bits or list(range(8))
        self._intermediate_func = intermediate_func or DPA._default_intermediate
    
    @property
    def target_bits(self) -> List[int]:
        return self._target_bits
    
    @target_bits.setter
    def target_bits(self, bits: List[int]):
        for b in bits:
            if not 0 <= b <= 7:
                raise ValueError(f"target_bit {b} must be between 0 and 7")
        self._target_bits = bits
    
    @property
    def attack_type(self) -> AttackType:
        return AttackType.DPA
    
    def attack_byte(self, traces: np.ndarray,
                    plaintexts: np.ndarray,
                    byte_index: int) -> Tuple[int, float, np.ndarray]:
        num_traces = traces.shape[0]
        num_samples = traces.shape[1]
        
        combined_scores = np.zeros(256)
        best_differentials = np.zeros(num_samples)
        best_max = 0.0
        best_key = 0
        
        for key_guess in range(256):
            bit_differentials = np.zeros((len(self._target_bits), num_samples))
            
            for bit_idx, target_bit in enumerate(self._target_bits):
                group0_mask = np.zeros(num_traces, dtype=bool)
                group1_mask = np.zeros(num_traces, dtype=bool)
                
                for i in range(num_traces):
                    intermediate = self._intermediate_func(plaintexts[i, byte_index], key_guess)
                    bit_value = (intermediate >> target_bit) & 1
                    if bit_value == 0:
                        group0_mask[i] = True
                    else:
                        group1_mask[i] = True
                
                if np.sum(group0_mask) > 0 and np.sum(group1_mask) > 0:
                    diff = differential_means(traces[group1_mask], traces[group0_mask])
                    bit_differentials[bit_idx] = diff
            
            combined = np.sum(np.abs(bit_differentials), axis=0)
            max_combined = np.max(combined)
            combined_scores[key_guess] = max_combined
            
            if max_combined > best_max:
                best_max = max_combined
                best_key = key_guess
                best_differentials = combined
        
        self._correlation_matrix = None
        
        second_best = np.partition(combined_scores, -2)[-2]
        confidence_margin = best_max - second_best
        
        return best_key, confidence_margin, best_differentials


class IncrementalDPA(IncrementalAttack):
    
    def __init__(self, leakage_model: Optional[LeakageModel] = None,
                 batch_size: int = 100,
                 key_byte_indices: Optional[List[int]] = None,
                 num_key_bytes: int = 16,
                 target_bit: int = 0,
                 intermediate_func: Optional[Callable] = None):
        if leakage_model is None:
            leakage_model = HammingWeightModel(8)
        super().__init__(leakage_model, batch_size)
        self._key_byte_indices = key_byte_indices or list(range(num_key_bytes))
        self._num_key_bytes = num_key_bytes
        self._target_bit = target_bit
        self._intermediate_func = intermediate_func or DPA._default_intermediate
        
        self._sum_group0: Optional[np.ndarray] = None
        self._sum_group1: Optional[np.ndarray] = None
        self._count_group0: Optional[np.ndarray] = None
        self._count_group1: Optional[np.ndarray] = None
    
    @property
    def attack_type(self) -> AttackType:
        return AttackType.DPA
    
    def reset(self):
        super().reset()
        self._sum_group0 = None
        self._sum_group1 = None
        self._count_group0 = None
        self._count_group1 = None
    
    def update(self, traces: np.ndarray,
               plaintexts: Optional[np.ndarray] = None,
               **kwargs) -> Optional[AttackResult]:
        if plaintexts is None:
            raise ValueError("plaintexts must be provided for DPA")
        
        num_traces = traces.shape[0]
        num_samples = traces.shape[1]
        num_bytes = len(self._key_byte_indices)
        
        if self._sum_group0 is None:
            self._sum_group0 = np.zeros((num_bytes, 256, num_samples), dtype=np.float64)
            self._sum_group1 = np.zeros((num_bytes, 256, num_samples), dtype=np.float64)
            self._count_group0 = np.zeros((num_bytes, 256), dtype=np.int64)
            self._count_group1 = np.zeros((num_bytes, 256), dtype=np.int64)
        
        for byte_idx_pos, byte_idx in enumerate(self._key_byte_indices):
            for key_guess in range(256):
                for i in range(num_traces):
                    intermediate = self._intermediate_func(plaintexts[i, byte_idx], key_guess)
                    bit_value = (intermediate >> self._target_bit) & 1
                    
                    if bit_value == 0:
                        self._sum_group0[byte_idx_pos, key_guess] += traces[i]
                        self._count_group0[byte_idx_pos, key_guess] += 1
                    else:
                        self._sum_group1[byte_idx_pos, key_guess] += traces[i]
                        self._count_group1[byte_idx_pos, key_guess] += 1
        
        self._trace_count += num_traces
        
        return None
    
    def finalize(self) -> AttackResult:
        if self._trace_count < 2:
            raise ValueError("Need at least 2 traces")
        
        recovered_bytes = {}
        confidences = []
        
        for byte_idx_pos, byte_idx in enumerate(self._key_byte_indices):
            differentials = np.zeros((256, self._sum_group0.shape[2]))
            
            for key_guess in range(256):
                n0 = self._count_group0[byte_idx_pos, key_guess]
                n1 = self._count_group1[byte_idx_pos, key_guess]
                
                if n0 > 0 and n1 > 0:
                    mean0 = self._sum_group0[byte_idx_pos, key_guess] / n0
                    mean1 = self._sum_group1[byte_idx_pos, key_guess] / n1
                    differentials[key_guess] = mean1 - mean0
            
            max_diffs = np.max(np.abs(differentials), axis=1)
            best_key = int(np.argmax(max_diffs))
            
            second_best = np.partition(max_diffs, -2)[-2]
            confidence = max_diffs[best_key] - second_best
            
            recovered_bytes[byte_idx] = best_key
            confidences.append(confidence)
        
        recovered_key = np.zeros(self._num_key_bytes, dtype=np.uint8)
        for idx, key_byte in recovered_bytes.items():
            if idx < self._num_key_bytes:
                recovered_key[idx] = key_byte
        
        self._result = AttackResult(
            attack_type=AttackType.DPA,
            success=True,
            recovered_key=recovered_key,
            recovered_bytes=recovered_bytes,
            confidence=float(np.mean(confidences)),
            num_traces_used=self._trace_count,
            metrics={
                "per_byte_confidence": {i: c for i, c in zip(self._key_byte_indices, confidences)},
                "target_bit": self._target_bit,
            }
        )
        
        return self._result
    
    def run(self, traces: TraceSet, **kwargs) -> AttackResult:
        samples = traces.get_samples_matrix()
        plaintexts = traces.get_plaintexts()
        
        if plaintexts is None:
            raise ValueError("TraceSet must contain plaintexts for DPA")
        
        self.update(samples, plaintexts)
        return self.finalize()


def run_dpa(traces: TraceSet,
            leakage_model: Optional[LeakageModel] = None,
            key_byte_indices: Optional[List[int]] = None,
            target_bit: int = 0,
            intermediate_func: Optional[Callable] = None) -> AttackResult:
    dpa = DPA(
        leakage_model=leakage_model,
        key_byte_indices=key_byte_indices,
        target_bit=target_bit,
        intermediate_func=intermediate_func,
    )
    return dpa.run(traces)


def run_multibit_dpa(traces: TraceSet,
                     leakage_model: Optional[LeakageModel] = None,
                     key_byte_indices: Optional[List[int]] = None,
                     target_bits: Optional[List[int]] = None,
                     intermediate_func: Optional[Callable] = None) -> AttackResult:
    dpa = MultiBitDPA(
        leakage_model=leakage_model,
        key_byte_indices=key_byte_indices,
        target_bits=target_bits,
        intermediate_func=intermediate_func,
    )
    return dpa.run(traces)
