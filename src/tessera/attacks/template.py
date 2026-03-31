from typing import Optional, Tuple, List, Dict, Any, Callable
import numpy as np
from dataclasses import dataclass, field

from .base import (
    Attack,
    AttackType,
    AttackResult,
)
from .cpa import SBOX
from ..leakage import LeakageModel, HammingWeightModel, TraceSet


@dataclass
class TemplateProfile:
    means: np.ndarray
    covariance: np.ndarray
    inv_covariance: Optional[np.ndarray] = None
    poi_indices: Optional[np.ndarray] = None
    class_labels: Optional[np.ndarray] = None
    num_traces_per_class: Optional[Dict[int, int]] = None
    
    def __post_init__(self):
        if self.inv_covariance is None and self.covariance is not None:
            try:
                self.inv_covariance = np.linalg.inv(self.covariance)
            except np.linalg.LinAlgError:
                self.inv_covariance = np.linalg.pinv(self.covariance)


class TemplateAttack(Attack):
    
    def __init__(self, leakage_model: Optional[LeakageModel] = None,
                 num_poi: int = 5,
                 poi_selection: str = "snr",
                 pooled_covariance: bool = True,
                 regularization: float = 1e-6):
        if leakage_model is None:
            leakage_model = HammingWeightModel(8)
        super().__init__(leakage_model)
        self._num_poi = num_poi
        self._poi_selection = poi_selection
        self._pooled_covariance = pooled_covariance
        self._regularization = regularization
        self._templates: Dict[int, TemplateProfile] = {}
        self._poi_indices: Optional[np.ndarray] = None
    
    @property
    def num_poi(self) -> int:
        return self._num_poi
    
    @num_poi.setter
    def num_poi(self, value: int):
        self._num_poi = value
    
    @property
    def poi_selection(self) -> str:
        return self._poi_selection
    
    @property
    def pooled_covariance(self) -> bool:
        return self._pooled_covariance
    
    @property
    def templates(self) -> Dict[int, TemplateProfile]:
        return self._templates
    
    @property
    def poi_indices(self) -> Optional[np.ndarray]:
        return self._poi_indices
    
    @property
    def attack_type(self) -> AttackType:
        return AttackType.TEMPLATE
    
    def _select_poi_snr(self, traces: np.ndarray, labels: np.ndarray) -> np.ndarray:
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
        
        poi_indices = np.argsort(snr)[-self._num_poi:][::-1]
        return poi_indices
    
    def _select_poi_sost(self, traces: np.ndarray, labels: np.ndarray) -> np.ndarray:
        unique_labels = np.unique(labels)
        n_classes = len(unique_labels)
        n_samples = traces.shape[1]
        
        sost = np.zeros(n_samples)
        
        for i in range(n_classes):
            class_i_traces = traces[labels == unique_labels[i]]
            mean_i = np.mean(class_i_traces, axis=0)
            
            for j in range(i + 1, n_classes):
                class_j_traces = traces[labels == unique_labels[j]]
                mean_j = np.mean(class_j_traces, axis=0)
                
                var_i = np.var(class_i_traces, axis=0)
                var_j = np.var(class_j_traces, axis=0)
                n_i = len(class_i_traces)
                n_j = len(class_j_traces)
                
                se = np.sqrt(var_i / n_i + var_j / n_j)
                se = np.where(se < 1e-10, 1e-10, se)
                
                t_stat = np.abs(mean_i - mean_j) / se
                sost = np.maximum(sost, t_stat)
        
        poi_indices = np.argsort(sost)[-self._num_poi:][::-1]
        return poi_indices
    
    def _select_poi(self, traces: np.ndarray, labels: np.ndarray) -> np.ndarray:
        if self._poi_selection == "snr":
            return self._select_poi_snr(traces, labels)
        elif self._poi_selection == "sost":
            return self._select_poi_sost(traces, labels)
        else:
            raise ValueError(f"Unknown POI selection method: {self._poi_selection}")
    
    def build_templates(self, traces: TraceSet,
                        labels: Optional[np.ndarray] = None,
                        intermediate_func: Optional[Callable] = None) -> Dict[int, TemplateProfile]:
        samples = traces.get_samples_matrix()
        plaintexts = traces.get_plaintexts()
        keys = traces.get_keys()
        
        if labels is None:
            if intermediate_func is None:
                intermediate_func = lambda p, k: SBOX[p[0] ^ k[0]]
            
            if plaintexts is None or keys is None:
                raise ValueError("labels, or plaintexts and keys must be provided")
            
            labels = np.array([
                intermediate_func(plaintexts[i], keys[i])
                for i in range(traces.num_traces)
            ])
        
        self._poi_indices = self._select_poi(samples, labels)
        poi_traces = samples[:, self._poi_indices]
        
        unique_labels = np.unique(labels)
        class_means = {}
        class_covs = {}
        class_counts = {}
        
        for label in unique_labels:
            class_traces = poi_traces[labels == label]
            class_counts[int(label)] = len(class_traces)
            class_means[int(label)] = np.mean(class_traces, axis=0)
            if len(class_traces) > 1:
                class_covs[int(label)] = np.cov(class_traces, rowvar=False)
            else:
                class_covs[int(label)] = np.eye(self._num_poi) * self._regularization
        
        if self._pooled_covariance:
            pooled_cov = np.zeros((self._num_poi, self._num_poi))
            total_count = 0
            
            for label in unique_labels:
                n = class_counts[int(label)]
                if n > 1:
                    pooled_cov += (n - 1) * class_covs[int(label)]
                    total_count += n - 1
            
            if total_count > 0:
                pooled_cov /= total_count
            
            pooled_cov += np.eye(self._num_poi) * self._regularization
            
            try:
                pooled_inv = np.linalg.inv(pooled_cov)
            except np.linalg.LinAlgError:
                pooled_inv = np.linalg.pinv(pooled_cov)
            
            for label in unique_labels:
                self._templates[int(label)] = TemplateProfile(
                    means=class_means[int(label)],
                    covariance=pooled_cov,
                    inv_covariance=pooled_inv,
                    poi_indices=self._poi_indices,
                    class_labels=unique_labels,
                    num_traces_per_class=class_counts,
                )
        else:
            for label in unique_labels:
                cov = class_covs[int(label)]
                cov += np.eye(self._num_poi) * self._regularization
                
                try:
                    inv_cov = np.linalg.inv(cov)
                except np.linalg.LinAlgError:
                    inv_cov = np.linalg.pinv(cov)
                
                self._templates[int(label)] = TemplateProfile(
                    means=class_means[int(label)],
                    covariance=cov,
                    inv_covariance=inv_cov,
                    poi_indices=self._poi_indices,
                    class_labels=unique_labels,
                    num_traces_per_class=class_counts,
                )
        
        return self._templates
    
    def _log_likelihood(self, trace: np.ndarray, template: TemplateProfile) -> float:
        diff = trace - template.means
        
        mahalanobis_sq = np.dot(np.dot(diff, template.inv_covariance), diff)
        
        log_det = np.log(np.linalg.det(template.covariance) + 1e-300)
        n = len(trace)
        
        log_prob = -0.5 * (n * np.log(2 * np.pi) + log_det + mahalanobis_sq)
        
        return log_prob
    
    def attack_trace(self, trace: np.ndarray) -> Tuple[int, np.ndarray]:
        if self._poi_indices is None:
            raise ValueError("Templates not built. Call build_templates first.")
        
        poi_trace = trace[self._poi_indices]
        
        log_probs = {}
        for label, template in self._templates.items():
            log_probs[label] = self._log_likelihood(poi_trace, template)
        
        max_log_prob = max(log_probs.values())
        probs = {
            label: np.exp(lp - max_log_prob) 
            for label, lp in log_probs.items()
        }
        total_prob = sum(probs.values())
        probs = {label: p / total_prob for label, p in probs.items()}
        
        best_label = max(probs.keys(), key=lambda k: probs[k])
        prob_array = np.array([probs.get(i, 0.0) for i in range(256)])
        
        return best_label, prob_array
    
    def run(self, traces: TraceSet, **kwargs) -> AttackResult:
        if not self._templates:
            raise ValueError("Templates not built. Call build_templates first.")
        
        samples = traces.get_samples_matrix()
        
        predictions = []
        all_probs = []
        
        for i in range(traces.num_traces):
            pred, probs = self.attack_trace(samples[i])
            predictions.append(pred)
            all_probs.append(probs)
        
        predictions = np.array(predictions)
        all_probs = np.array(all_probs)
        
        avg_probs = np.mean(all_probs, axis=0)
        best_guess = int(np.argmax(avg_probs))
        confidence = float(avg_probs[best_guess])
        
        self._result = AttackResult(
            attack_type=AttackType.TEMPLATE,
            success=True,
            recovered_key=np.array([best_guess], dtype=np.uint8),
            recovered_bytes={0: best_guess},
            confidence=confidence,
            num_traces_used=traces.num_traces,
            metrics={
                "predictions": predictions.tolist(),
                "avg_probabilities": avg_probs.tolist(),
                "poi_indices": self._poi_indices.tolist() if self._poi_indices is not None else [],
            }
        )
        
        return self._result
    
    def reset(self):
        super().reset()
        self._templates.clear()
        self._poi_indices = None


class TemplateKeyRecovery(Attack):
    
    def __init__(self, leakage_model: Optional[LeakageModel] = None,
                 num_poi: int = 5,
                 poi_selection: str = "snr",
                 pooled_covariance: bool = True,
                 regularization: float = 1e-6,
                 key_byte_indices: Optional[List[int]] = None,
                 num_key_bytes: int = 16):
        if leakage_model is None:
            leakage_model = HammingWeightModel(8)
        super().__init__(leakage_model)
        self._num_poi = num_poi
        self._poi_selection = poi_selection
        self._pooled_covariance = pooled_covariance
        self._regularization = regularization
        self._key_byte_indices = key_byte_indices or list(range(num_key_bytes))
        self._num_key_bytes = num_key_bytes
        self._byte_templates: Dict[int, TemplateAttack] = {}
    
    @property
    def attack_type(self) -> AttackType:
        return AttackType.TEMPLATE
    
    def build_templates(self, profiling_traces: TraceSet,
                        intermediate_func: Optional[Callable] = None):
        if intermediate_func is None:
            intermediate_func = lambda p, k, byte_idx: SBOX[p[byte_idx] ^ k[byte_idx]]
        
        samples = profiling_traces.get_samples_matrix()
        plaintexts = profiling_traces.get_plaintexts()
        keys = profiling_traces.get_keys()
        
        if plaintexts is None or keys is None:
            raise ValueError("Profiling traces must have plaintexts and keys")
        
        for byte_idx in self._key_byte_indices:
            labels = np.array([
                intermediate_func(plaintexts[i], keys[i], byte_idx)
                for i in range(profiling_traces.num_traces)
            ])
            
            template = TemplateAttack(
                leakage_model=self._leakage_model,
                num_poi=self._num_poi,
                poi_selection=self._poi_selection,
                pooled_covariance=self._pooled_covariance,
                regularization=self._regularization,
            )
            template.build_templates(profiling_traces, labels)
            self._byte_templates[byte_idx] = template
    
    def run(self, traces: TraceSet, **kwargs) -> AttackResult:
        if not self._byte_templates:
            raise ValueError("Templates not built. Call build_templates first.")
        
        plaintexts = traces.get_plaintexts()
        if plaintexts is None:
            raise ValueError("Attack traces must have plaintexts")
        
        samples = traces.get_samples_matrix()
        
        recovered_bytes = {}
        confidences = []
        
        for byte_idx in self._key_byte_indices:
            template = self._byte_templates[byte_idx]
            
            key_scores = np.zeros(256)
            
            for trace_idx in range(traces.num_traces):
                _, probs = template.attack_trace(samples[trace_idx])
                
                pt_byte = plaintexts[trace_idx, byte_idx]
                for k in range(256):
                    intermediate = SBOX[pt_byte ^ k]
                    key_scores[k] += np.log(probs[intermediate] + 1e-300)
            
            best_key = int(np.argmax(key_scores))
            
            max_score = key_scores[best_key]
            second_best = np.partition(key_scores, -2)[-2]
            confidence = max_score - second_best
            
            recovered_bytes[byte_idx] = best_key
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
            attack_type=AttackType.TEMPLATE,
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
    
    def reset(self):
        super().reset()
        self._byte_templates.clear()


def run_template_attack(profiling_traces: TraceSet,
                        attack_traces: TraceSet,
                        num_poi: int = 5,
                        pooled_covariance: bool = True) -> AttackResult:
    template = TemplateAttack(
        num_poi=num_poi,
        pooled_covariance=pooled_covariance,
    )
    template.build_templates(profiling_traces)
    return template.run(attack_traces)
