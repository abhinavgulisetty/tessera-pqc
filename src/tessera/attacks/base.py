from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from ..leakage import TraceSet, LeakageModel


class AttackType(Enum):
    TTEST = "ttest"
    CPA = "cpa"
    DPA = "dpa"
    TEMPLATE = "template"
    LATTICE = "lattice"


@dataclass
class AttackResult:
    attack_type: AttackType
    success: bool
    recovered_key: Optional[np.ndarray] = None
    recovered_bytes: Optional[Dict[int, int]] = None
    confidence: float = 0.0
    num_traces_used: int = 0
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "attack_type": self.attack_type.value,
            "success": self.success,
            "recovered_key": self.recovered_key.tolist() if self.recovered_key is not None else None,
            "recovered_bytes": self.recovered_bytes,
            "confidence": self.confidence,
            "num_traces_used": self.num_traces_used,
            "metrics": self.metrics,
        }


@dataclass
class LeakageAssessmentResult:
    leakage_detected: bool
    max_statistic: float
    threshold: float
    poi_indices: np.ndarray
    statistic_per_sample: np.ndarray
    confidence_level: float = 0.99999
    num_traces_used: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "leakage_detected": self.leakage_detected,
            "max_statistic": float(self.max_statistic),
            "threshold": float(self.threshold),
            "poi_indices": self.poi_indices.tolist(),
            "confidence_level": self.confidence_level,
            "num_traces_used": self.num_traces_used,
        }


class Attack(ABC):
    
    def __init__(self, leakage_model: Optional[LeakageModel] = None):
        self._leakage_model = leakage_model
        self._result: Optional[AttackResult] = None
    
    @property
    def leakage_model(self) -> Optional[LeakageModel]:
        return self._leakage_model
    
    @leakage_model.setter
    def leakage_model(self, model: LeakageModel):
        self._leakage_model = model
    
    @property
    def result(self) -> Optional[AttackResult]:
        return self._result
    
    @property
    @abstractmethod
    def attack_type(self) -> AttackType:
        pass
    
    @abstractmethod
    def run(self, traces: TraceSet, **kwargs) -> AttackResult:
        pass
    
    def reset(self):
        self._result = None


class KeyRecoveryAttack(Attack):
    
    def __init__(self, leakage_model: Optional[LeakageModel] = None,
                 key_byte_indices: Optional[List[int]] = None,
                 num_key_bytes: int = 16):
        super().__init__(leakage_model)
        self._key_byte_indices = key_byte_indices or list(range(num_key_bytes))
        self._num_key_bytes = num_key_bytes
        self._correlation_matrix: Optional[np.ndarray] = None
        self._key_hypotheses: Optional[np.ndarray] = None
    
    @property
    def key_byte_indices(self) -> List[int]:
        return self._key_byte_indices
    
    @key_byte_indices.setter
    def key_byte_indices(self, indices: List[int]):
        self._key_byte_indices = indices
    
    @property
    def num_key_bytes(self) -> int:
        return self._num_key_bytes
    
    @property
    def correlation_matrix(self) -> Optional[np.ndarray]:
        return self._correlation_matrix
    
    @abstractmethod
    def attack_byte(self, traces: np.ndarray, 
                    plaintexts: np.ndarray,
                    byte_index: int) -> Tuple[int, float, np.ndarray]:
        pass
    
    def run(self, traces: TraceSet, **kwargs) -> AttackResult:
        samples = traces.get_samples_matrix()
        plaintexts = traces.get_plaintexts()
        
        if plaintexts is None:
            raise ValueError("TraceSet must contain plaintexts for key recovery attack")
        
        recovered_bytes = {}
        confidences = []
        
        for byte_idx in self._key_byte_indices:
            key_byte, confidence, correlations = self.attack_byte(
                samples, plaintexts, byte_idx
            )
            recovered_bytes[byte_idx] = key_byte
            confidences.append(confidence)
        
        recovered_key = np.zeros(self._num_key_bytes, dtype=np.uint8)
        for idx, key_byte in recovered_bytes.items():
            if idx < self._num_key_bytes:
                recovered_key[idx] = key_byte
        
        known_key = traces.get_keys()
        if known_key is not None:
            known_key = known_key[0]
            success = all(
                recovered_bytes.get(i, -1) == known_key[i] 
                for i in self._key_byte_indices
            )
        else:
            success = True
        
        self._result = AttackResult(
            attack_type=self.attack_type,
            success=success,
            recovered_key=recovered_key,
            recovered_bytes=recovered_bytes,
            confidence=float(np.mean(confidences)),
            num_traces_used=traces.num_traces,
            metrics={
                "per_byte_confidence": {i: c for i, c in zip(self._key_byte_indices, confidences)},
            }
        )
        
        return self._result


class LeakageAssessment(ABC):
    
    def __init__(self, confidence_level: float = 0.99999):
        self._confidence_level = confidence_level
        self._result: Optional[LeakageAssessmentResult] = None
    
    @property
    def confidence_level(self) -> float:
        return self._confidence_level
    
    @confidence_level.setter
    def confidence_level(self, value: float):
        if not 0 < value < 1:
            raise ValueError("confidence_level must be in (0, 1)")
        self._confidence_level = value
    
    @property
    def result(self) -> Optional[LeakageAssessmentResult]:
        return self._result
    
    @abstractmethod
    def assess(self, group1: np.ndarray, group2: np.ndarray) -> LeakageAssessmentResult:
        pass
    
    def reset(self):
        self._result = None


class IncrementalAttack(Attack):
    
    def __init__(self, leakage_model: Optional[LeakageModel] = None,
                 batch_size: int = 100):
        super().__init__(leakage_model)
        self._batch_size = batch_size
        self._trace_count = 0
        self._accumulated_data: Dict[str, Any] = {}
    
    @property
    def batch_size(self) -> int:
        return self._batch_size
    
    @batch_size.setter
    def batch_size(self, value: int):
        self._batch_size = value
    
    @property
    def trace_count(self) -> int:
        return self._trace_count
    
    @abstractmethod
    def update(self, traces: np.ndarray, 
               plaintexts: Optional[np.ndarray] = None,
               **kwargs) -> Optional[AttackResult]:
        pass
    
    @abstractmethod
    def finalize(self) -> AttackResult:
        pass
    
    def reset(self):
        super().reset()
        self._trace_count = 0
        self._accumulated_data.clear()


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    
    x_centered = x - np.mean(x, axis=0, keepdims=True)
    y_centered = y - np.mean(y, axis=0, keepdims=True)
    
    x_std = np.std(x, axis=0, keepdims=True)
    y_std = np.std(y, axis=0, keepdims=True)
    
    x_std = np.where(x_std < 1e-10, 1.0, x_std)
    y_std = np.where(y_std < 1e-10, 1.0, y_std)
    
    correlation = np.dot(x_centered.T, y_centered) / (len(x) * x_std.T * y_std)
    
    return correlation


def welch_t_statistic(group1: np.ndarray, group2: np.ndarray) -> np.ndarray:
    n1 = group1.shape[0]
    n2 = group2.shape[0]
    
    mean1 = np.mean(group1, axis=0)
    mean2 = np.mean(group2, axis=0)
    
    var1 = np.var(group1, axis=0, ddof=1)
    var2 = np.var(group2, axis=0, ddof=1)
    
    se = np.sqrt(var1 / n1 + var2 / n2)
    se = np.where(se < 1e-10, 1e-10, se)
    
    t_stat = (mean1 - mean2) / se
    
    return t_stat


def differential_means(group1: np.ndarray, group2: np.ndarray) -> np.ndarray:
    return np.mean(group1, axis=0) - np.mean(group2, axis=0)


def snr_distinguisher(traces: np.ndarray, labels: np.ndarray) -> np.ndarray:
    unique_labels = np.unique(labels)
    
    class_means = np.zeros((len(unique_labels), traces.shape[1]))
    class_vars = np.zeros((len(unique_labels), traces.shape[1]))
    
    for i, label in enumerate(unique_labels):
        class_traces = traces[labels == label]
        class_means[i] = np.mean(class_traces, axis=0)
        class_vars[i] = np.var(class_traces, axis=0)
    
    signal_var = np.var(class_means, axis=0)
    noise_var = np.mean(class_vars, axis=0)
    
    noise_var = np.where(noise_var < 1e-10, 1e-10, noise_var)
    snr = signal_var / noise_var
    
    return snr
