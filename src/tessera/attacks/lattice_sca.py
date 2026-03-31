from typing import Optional, Tuple, List, Dict, Any, Callable
import numpy as np
from dataclasses import dataclass, field
from enum import Enum

from .base import (
    Attack,
    AttackType,
    AttackResult,
    LeakageAssessment,
    LeakageAssessmentResult,
    pearson_correlation,
    welch_t_statistic,
)
from ..leakage import LeakageModel, HammingWeightModel, TraceSet


class LatticeOperation(Enum):
    NTT = "ntt"
    INTT = "intt"
    POLY_MUL = "poly_mul"
    POLY_ADD = "poly_add"
    SAMPLE = "sample"
    COMPRESS = "compress"
    DECOMPRESS = "decompress"


@dataclass
class CoefficientLeakageResult:
    coefficient_index: int
    recovered_value: int
    confidence: float
    correlation_trace: np.ndarray
    candidates: List[Tuple[int, float]]


@dataclass 
class NTTAttackResult:
    layer: int
    butterfly_index: int
    recovered_twiddle: Optional[int] = None
    recovered_input: Optional[Tuple[int, int]] = None
    confidence: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)


class ButterflyLeakageModel:
    
    def __init__(self, q: int = 3329, word_size: int = 16):
        self._q = q
        self._word_size = word_size
        self._hw_model = HammingWeightModel(word_size)
    
    @property
    def q(self) -> int:
        return self._q
    
    def butterfly_output_hw(self, a: int, b: int, twiddle: int) -> Tuple[int, int]:
        t = (b * twiddle) % self._q
        out_a = (a + t) % self._q
        out_b = (a - t) % self._q
        if out_b < 0:
            out_b += self._q
        return self._hw_model(out_a), self._hw_model(out_b)
    
    def butterfly_intermediate_hw(self, b: int, twiddle: int) -> int:
        t = (b * twiddle) % self._q
        return self._hw_model(t)
    
    def combined_leakage(self, a: int, b: int, twiddle: int) -> float:
        hw_a, hw_b = self.butterfly_output_hw(a, b, twiddle)
        hw_t = self.butterfly_intermediate_hw(b, twiddle)
        return hw_a + hw_b + hw_t


class NTTCoefficientAttack(Attack):
    
    def __init__(self, leakage_model: Optional[LeakageModel] = None,
                 q: int = 3329,
                 n: int = 256,
                 target_coefficients: Optional[List[int]] = None,
                 coefficient_range: Optional[Tuple[int, int]] = None):
        if leakage_model is None:
            leakage_model = HammingWeightModel(16)
        super().__init__(leakage_model)
        self._q = q
        self._n = n
        self._target_coefficients = target_coefficients or list(range(min(16, n)))
        if coefficient_range is None:
            self._coefficient_range = (0, q)
        else:
            self._coefficient_range = coefficient_range
        self._results: Dict[int, CoefficientLeakageResult] = {}
    
    @property
    def q(self) -> int:
        return self._q
    
    @property
    def n(self) -> int:
        return self._n
    
    @property
    def target_coefficients(self) -> List[int]:
        return self._target_coefficients
    
    @property
    def attack_type(self) -> AttackType:
        return AttackType.LATTICE
    
    @property
    def coefficient_results(self) -> Dict[int, CoefficientLeakageResult]:
        return self._results
    
    def attack_coefficient(self, traces: np.ndarray,
                           coeff_idx: int,
                           known_inputs: Optional[np.ndarray] = None) -> CoefficientLeakageResult:
        num_traces = traces.shape[0]
        num_samples = traces.shape[1]
        
        min_val, max_val = self._coefficient_range
        num_hypotheses = min(256, max_val - min_val)
        step = max(1, (max_val - min_val) // num_hypotheses)
        
        hypotheses = list(range(min_val, max_val, step))
        correlations = np.zeros((len(hypotheses), num_samples))
        
        for h_idx, hypothesis in enumerate(hypotheses):
            hyp_leakage = np.full(num_traces, self._leakage_model(hypothesis))
            corr = pearson_correlation(hyp_leakage, traces)
            correlations[h_idx] = corr.flatten()
        
        max_corrs = np.max(np.abs(correlations), axis=1)
        
        sorted_indices = np.argsort(max_corrs)[::-1]
        candidates = [(hypotheses[i], float(max_corrs[i])) for i in sorted_indices[:5]]
        
        best_idx = sorted_indices[0]
        best_hypothesis = hypotheses[best_idx]
        best_correlation = max_corrs[best_idx]
        
        second_best_corr = max_corrs[sorted_indices[1]] if len(sorted_indices) > 1 else 0
        confidence = best_correlation - second_best_corr
        
        result = CoefficientLeakageResult(
            coefficient_index=coeff_idx,
            recovered_value=best_hypothesis,
            confidence=confidence,
            correlation_trace=correlations[best_idx],
            candidates=candidates,
        )
        
        self._results[coeff_idx] = result
        return result
    
    def run(self, traces: TraceSet, **kwargs) -> AttackResult:
        samples = traces.get_samples_matrix()
        
        recovered_coeffs = {}
        total_confidence = 0.0
        
        for coeff_idx in self._target_coefficients:
            result = self.attack_coefficient(samples, coeff_idx)
            recovered_coeffs[coeff_idx] = result.recovered_value
            total_confidence += result.confidence
        
        avg_confidence = total_confidence / len(self._target_coefficients)
        
        self._result = AttackResult(
            attack_type=AttackType.LATTICE,
            success=True,
            recovered_key=None,
            recovered_bytes=recovered_coeffs,
            confidence=avg_confidence,
            num_traces_used=traces.num_traces,
            metrics={
                "coefficient_results": {
                    k: {
                        "value": v.recovered_value,
                        "confidence": v.confidence,
                        "candidates": v.candidates,
                    }
                    for k, v in self._results.items()
                }
            }
        )
        
        return self._result
    
    def reset(self):
        super().reset()
        self._results.clear()


class NTTButterflyAttack(Attack):
    
    def __init__(self, q: int = 3329,
                 n: int = 256,
                 twiddle_factors: Optional[np.ndarray] = None):
        super().__init__(None)
        self._q = q
        self._n = n
        self._twiddle_factors = twiddle_factors
        self._butterfly_model = ButterflyLeakageModel(q)
        self._layer_results: Dict[int, List[NTTAttackResult]] = {}
    
    @property
    def attack_type(self) -> AttackType:
        return AttackType.LATTICE
    
    def attack_butterfly(self, traces: np.ndarray,
                         layer: int,
                         butterfly_idx: int,
                         input_guesses: Optional[List[Tuple[int, int]]] = None) -> NTTAttackResult:
        num_traces = traces.shape[0]
        
        if input_guesses is None:
            input_guesses = [
                (a, b) for a in range(0, self._q, self._q // 16)
                for b in range(0, self._q, self._q // 16)
            ]
        
        best_corr = 0.0
        best_inputs = None
        
        for a, b in input_guesses:
            if self._twiddle_factors is not None:
                twiddle = self._twiddle_factors[layer * (self._n // 2) + butterfly_idx]
            else:
                twiddle = 1
            
            leakage = self._butterfly_model.combined_leakage(a, b, twiddle)
            hyp_leakage = np.full(num_traces, leakage)
            
            corr = np.abs(pearson_correlation(hyp_leakage, traces)).max()
            
            if corr > best_corr:
                best_corr = corr
                best_inputs = (a, b)
        
        result = NTTAttackResult(
            layer=layer,
            butterfly_index=butterfly_idx,
            recovered_input=best_inputs,
            confidence=best_corr,
            metrics={"num_guesses": len(input_guesses)}
        )
        
        if layer not in self._layer_results:
            self._layer_results[layer] = []
        self._layer_results[layer].append(result)
        
        return result
    
    def run(self, traces: TraceSet, **kwargs) -> AttackResult:
        samples = traces.get_samples_matrix()
        
        target_layers = kwargs.get("target_layers", [0])
        butterflies_per_layer = kwargs.get("butterflies_per_layer", 4)
        
        all_results = []
        total_confidence = 0.0
        
        for layer in target_layers:
            for bf_idx in range(butterflies_per_layer):
                result = self.attack_butterfly(samples, layer, bf_idx)
                all_results.append(result)
                total_confidence += result.confidence
        
        avg_confidence = total_confidence / len(all_results) if all_results else 0.0
        
        self._result = AttackResult(
            attack_type=AttackType.LATTICE,
            success=True,
            recovered_key=None,
            recovered_bytes=None,
            confidence=avg_confidence,
            num_traces_used=traces.num_traces,
            metrics={
                "layer_results": {
                    layer: [
                        {
                            "butterfly_index": r.butterfly_index,
                            "recovered_input": r.recovered_input,
                            "confidence": r.confidence,
                        }
                        for r in results
                    ]
                    for layer, results in self._layer_results.items()
                }
            }
        )
        
        return self._result
    
    def reset(self):
        super().reset()
        self._layer_results.clear()


class SecretKeyRecoveryAttack(Attack):
    
    def __init__(self, leakage_model: Optional[LeakageModel] = None,
                 q: int = 3329,
                 n: int = 256,
                 secret_bound: int = 2,
                 target_coefficients: Optional[List[int]] = None):
        if leakage_model is None:
            leakage_model = HammingWeightModel(16)
        super().__init__(leakage_model)
        self._q = q
        self._n = n
        self._secret_bound = secret_bound
        self._target_coefficients = target_coefficients or list(range(n))
    
    @property
    def attack_type(self) -> AttackType:
        return AttackType.LATTICE
    
    def run(self, traces: TraceSet, **kwargs) -> AttackResult:
        samples = traces.get_samples_matrix()
        public_data = kwargs.get("public_data")
        
        num_traces = traces.num_traces
        
        secret_candidates = list(range(-self._secret_bound, self._secret_bound + 1))
        
        recovered_secret = np.zeros(len(self._target_coefficients), dtype=np.int16)
        confidences = []
        
        for idx, coeff_idx in enumerate(self._target_coefficients):
            correlations = []
            
            for s_guess in secret_candidates:
                hyp_leakage = np.full(num_traces, self._leakage_model(s_guess % self._q))
                corr = np.max(np.abs(pearson_correlation(hyp_leakage, samples)))
                correlations.append(corr)
            
            best_idx = np.argmax(correlations)
            recovered_secret[idx] = secret_candidates[best_idx]
            
            sorted_corrs = sorted(correlations, reverse=True)
            confidence = sorted_corrs[0] - sorted_corrs[1] if len(sorted_corrs) > 1 else sorted_corrs[0]
            confidences.append(confidence)
        
        self._result = AttackResult(
            attack_type=AttackType.LATTICE,
            success=True,
            recovered_key=recovered_secret.astype(np.uint8),
            recovered_bytes={i: int(v) for i, v in enumerate(recovered_secret)},
            confidence=float(np.mean(confidences)),
            num_traces_used=num_traces,
            metrics={
                "per_coefficient_confidence": {
                    i: c for i, c in zip(self._target_coefficients, confidences)
                }
            }
        )
        
        return self._result


class BeliefPropagationAttack(Attack):
    
    def __init__(self, leakage_model: Optional[LeakageModel] = None,
                 q: int = 3329,
                 n: int = 256,
                 num_iterations: int = 10,
                 damping: float = 0.5):
        if leakage_model is None:
            leakage_model = HammingWeightModel(16)
        super().__init__(leakage_model)
        self._q = q
        self._n = n
        self._num_iterations = num_iterations
        self._damping = damping
    
    @property
    def attack_type(self) -> AttackType:
        return AttackType.LATTICE
    
    def _initialize_beliefs(self, traces: np.ndarray,
                            num_coefficients: int) -> np.ndarray:
        beliefs = np.ones((num_coefficients, self._q)) / self._q
        return beliefs
    
    def _update_beliefs(self, beliefs: np.ndarray,
                        coefficient_likelihoods: np.ndarray) -> np.ndarray:
        new_beliefs = beliefs * coefficient_likelihoods
        
        row_sums = np.sum(new_beliefs, axis=1, keepdims=True)
        row_sums = np.where(row_sums < 1e-300, 1e-300, row_sums)
        new_beliefs = new_beliefs / row_sums
        
        new_beliefs = self._damping * beliefs + (1 - self._damping) * new_beliefs
        
        return new_beliefs
    
    def run(self, traces: TraceSet, **kwargs) -> AttackResult:
        samples = traces.get_samples_matrix()
        num_coefficients = kwargs.get("num_coefficients", self._n)
        
        beliefs = self._initialize_beliefs(samples, num_coefficients)
        
        coefficient_likelihoods = np.ones((num_coefficients, self._q))
        for coeff_idx in range(num_coefficients):
            for val in range(self._q):
                hw = self._leakage_model(val)
                coefficient_likelihoods[coeff_idx, val] = np.exp(-hw / 10.0)
        
        for iteration in range(self._num_iterations):
            beliefs = self._update_beliefs(beliefs, coefficient_likelihoods)
        
        recovered_coeffs = np.argmax(beliefs, axis=1)
        max_beliefs = np.max(beliefs, axis=1)
        confidence = float(np.mean(max_beliefs))
        
        self._result = AttackResult(
            attack_type=AttackType.LATTICE,
            success=True,
            recovered_key=recovered_coeffs.astype(np.uint8),
            recovered_bytes={i: int(v) for i, v in enumerate(recovered_coeffs)},
            confidence=confidence,
            num_traces_used=traces.num_traces,
            metrics={
                "num_iterations": self._num_iterations,
                "final_beliefs_entropy": float(-np.sum(beliefs * np.log(beliefs + 1e-300))),
            }
        )
        
        return self._result


class LatticeLeakageAssessment(LeakageAssessment):
    
    def __init__(self, confidence_level: float = 0.99999,
                 operation: LatticeOperation = LatticeOperation.NTT):
        super().__init__(confidence_level)
        self._operation = operation
    
    @property
    def operation(self) -> LatticeOperation:
        return self._operation
    
    def assess(self, group1: np.ndarray, group2: np.ndarray) -> LeakageAssessmentResult:
        t_stat = welch_t_statistic(group1, group2)
        
        from scipy import stats
        threshold = stats.t.ppf(1 - (1 - self._confidence_level) / 2, 
                                df=min(len(group1), len(group2)) - 1)
        
        max_stat = np.max(np.abs(t_stat))
        leakage_detected = max_stat > threshold
        
        poi_indices = np.where(np.abs(t_stat) > threshold)[0]
        
        self._result = LeakageAssessmentResult(
            leakage_detected=leakage_detected,
            max_statistic=float(max_stat),
            threshold=float(threshold),
            poi_indices=poi_indices,
            statistic_per_sample=t_stat,
            confidence_level=self._confidence_level,
            num_traces_used=len(group1) + len(group2),
        )
        
        return self._result
    
    def assess_operation(self, traces_with_op: np.ndarray,
                         traces_without_op: np.ndarray) -> LeakageAssessmentResult:
        return self.assess(traces_with_op, traces_without_op)


def kyber_coefficient_attack(traces: TraceSet,
                              target_coefficients: Optional[List[int]] = None,
                              q: int = 3329) -> AttackResult:
    attack = NTTCoefficientAttack(
        q=q,
        target_coefficients=target_coefficients,
    )
    return attack.run(traces)


def dilithium_coefficient_attack(traces: TraceSet,
                                  target_coefficients: Optional[List[int]] = None,
                                  q: int = 8380417) -> AttackResult:
    attack = NTTCoefficientAttack(
        q=q,
        target_coefficients=target_coefficients,
        coefficient_range=(0, min(q, 65536)),
    )
    return attack.run(traces)


def lattice_secret_recovery(traces: TraceSet,
                            q: int = 3329,
                            n: int = 256,
                            secret_bound: int = 2,
                            target_coefficients: Optional[List[int]] = None) -> AttackResult:
    attack = SecretKeyRecoveryAttack(
        q=q,
        n=n,
        secret_bound=secret_bound,
        target_coefficients=target_coefficients,
    )
    return attack.run(traces)
