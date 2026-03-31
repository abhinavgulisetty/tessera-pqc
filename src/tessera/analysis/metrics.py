from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Union, Callable
from enum import Enum
import warnings


class MetricType(Enum):
    SNR = "snr"
    TVLA = "tvla"
    SUCCESS_RATE = "success_rate"
    GUESSING_ENTROPY = "guessing_entropy"
    CORRELATION = "correlation"
    MUTUAL_INFORMATION = "mutual_information"


@dataclass
class MetricResult:
    metric_type: MetricType
    value: Union[float, np.ndarray]
    metadata: Dict = field(default_factory=dict)
    
    def __repr__(self) -> str:
        if isinstance(self.value, np.ndarray):
            return f"MetricResult({self.metric_type.value}, shape={self.value.shape})"
        return f"MetricResult({self.metric_type.value}, value={self.value:.4f})"


class SNRCalculator:
    
    def __init__(self):
        pass
    
    def compute_snr(
        self, 
        traces: np.ndarray, 
        labels: np.ndarray
    ) -> np.ndarray:
        unique_labels = np.unique(labels)
        n_classes = len(unique_labels)
        n_samples = traces.shape[1]
        
        means = np.zeros((n_classes, n_samples))
        variances = np.zeros((n_classes, n_samples))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            class_traces = traces[mask]
            means[i] = np.mean(class_traces, axis=0)
            variances[i] = np.var(class_traces, axis=0)
        
        signal_var = np.var(means, axis=0)
        noise_var = np.mean(variances, axis=0)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            snr = signal_var / noise_var
            snr = np.nan_to_num(snr, nan=0.0, posinf=0.0, neginf=0.0)
        
        return snr
    
    def compute_snr_multivariate(
        self, 
        traces: np.ndarray, 
        labels: np.ndarray,
        num_components: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        unique_labels = np.unique(labels)
        n_classes = len(unique_labels)
        
        means = []
        for label in unique_labels:
            mask = labels == label
            means.append(np.mean(traces[mask], axis=0))
        means = np.array(means)
        
        global_mean = np.mean(traces, axis=0)
        
        signal_var = np.var(means, axis=0)
        
        top_indices = np.argsort(signal_var)[-num_components:][::-1]
        top_snr = signal_var[top_indices]
        
        return top_snr, top_indices
    
    def compute_snr_by_byte(
        self,
        traces: np.ndarray,
        data: np.ndarray,
        byte_index: int = 0
    ) -> np.ndarray:
        labels = data[:, byte_index] if data.ndim > 1 else data
        return self.compute_snr(traces, labels)


class TVLAMetrics:
    
    def __init__(self, threshold: float = 4.5):
        self.threshold = threshold
    
    def welch_t_test(
        self, 
        group1: np.ndarray, 
        group2: np.ndarray
    ) -> np.ndarray:
        n1 = group1.shape[0]
        n2 = group2.shape[0]
        
        mean1 = np.mean(group1, axis=0)
        mean2 = np.mean(group2, axis=0)
        
        var1 = np.var(group1, axis=0, ddof=1)
        var2 = np.var(group2, axis=0, ddof=1)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            t_stat = (mean1 - mean2) / np.sqrt(var1/n1 + var2/n2)
            t_stat = np.nan_to_num(t_stat, nan=0.0, posinf=0.0, neginf=0.0)
        
        return t_stat
    
    def detect_leakage(
        self, 
        t_statistic: np.ndarray,
        threshold: Optional[float] = None
    ) -> Tuple[bool, np.ndarray]:
        if threshold is None:
            threshold = self.threshold
        
        leaking_points = np.abs(t_statistic) > threshold
        has_leakage = np.any(leaking_points)
        
        return has_leakage, leaking_points
    
    def leakage_summary(
        self, 
        t_statistic: np.ndarray
    ) -> Dict:
        max_abs_t = np.max(np.abs(t_statistic))
        max_t_index = np.argmax(np.abs(t_statistic))
        
        has_leakage_45, points_45 = self.detect_leakage(t_statistic, 4.5)
        has_leakage_50, points_50 = self.detect_leakage(t_statistic, 5.0)
        
        return {
            "max_t_statistic": float(max_abs_t),
            "max_t_index": int(max_t_index),
            "leakage_at_4.5": has_leakage_45,
            "leakage_at_5.0": has_leakage_50,
            "num_leaking_points_4.5": int(np.sum(points_45)),
            "num_leaking_points_5.0": int(np.sum(points_50)),
            "leaking_indices_4.5": np.where(points_45)[0].tolist(),
        }
    
    def compute_higher_order_tvla(
        self,
        traces: np.ndarray,
        labels: np.ndarray,
        order: int = 2
    ) -> np.ndarray:
        unique_labels = np.unique(labels)
        if len(unique_labels) != 2:
            raise ValueError("TVLA requires exactly 2 groups")
        
        mask0 = labels == unique_labels[0]
        mask1 = labels == unique_labels[1]
        
        group0 = traces[mask0]
        group1 = traces[mask1]
        
        mean0 = np.mean(group0, axis=0, keepdims=True)
        mean1 = np.mean(group1, axis=0, keepdims=True)
        
        centered0 = group0 - mean0
        centered1 = group1 - mean1
        
        moment0 = np.mean(centered0 ** order, axis=0)
        moment1 = np.mean(centered1 ** order, axis=0)
        
        var0 = np.var(centered0 ** order, axis=0, ddof=1)
        var1 = np.var(centered1 ** order, axis=0, ddof=1)
        
        n0, n1 = group0.shape[0], group1.shape[0]
        
        with np.errstate(divide='ignore', invalid='ignore'):
            t_stat = (moment0 - moment1) / np.sqrt(var0/n0 + var1/n1)
            t_stat = np.nan_to_num(t_stat, nan=0.0, posinf=0.0, neginf=0.0)
        
        return t_stat


class SuccessRateMetrics:
    
    def __init__(self):
        pass
    
    def compute_success_rate(
        self,
        predicted_keys: np.ndarray,
        actual_keys: np.ndarray
    ) -> float:
        correct = np.sum(predicted_keys == actual_keys)
        total = len(predicted_keys)
        return float(correct / total) if total > 0 else 0.0
    
    def compute_partial_success_rate(
        self,
        key_rankings: np.ndarray,
        actual_key: int,
        rank_threshold: int = 1
    ) -> float:
        correct_rank = key_rankings[actual_key]
        return 1.0 if correct_rank < rank_threshold else 0.0
    
    def compute_success_rate_vs_traces(
        self,
        attack_function: Callable,
        traces: np.ndarray,
        plaintexts: np.ndarray,
        actual_key: np.ndarray,
        trace_counts: List[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        success_rates = []
        
        for n_traces in trace_counts:
            if n_traces > len(traces):
                break
            
            result = attack_function(traces[:n_traces], plaintexts[:n_traces])
            predicted = result.get("recovered_key", None)
            
            if predicted is not None:
                sr = self.compute_success_rate(
                    np.array([predicted]), 
                    np.array([actual_key[0] if len(actual_key) > 0 else actual_key])
                )
            else:
                sr = 0.0
            
            success_rates.append(sr)
        
        return np.array(trace_counts[:len(success_rates)]), np.array(success_rates)
    
    def compute_rank_over_time(
        self,
        correlations_over_time: List[np.ndarray],
        correct_key: int
    ) -> np.ndarray:
        ranks = []
        
        for corr in correlations_over_time:
            sorted_indices = np.argsort(corr)[::-1]
            rank = np.where(sorted_indices == correct_key)[0][0]
            ranks.append(rank)
        
        return np.array(ranks)


class GuessingEntropy:
    
    def __init__(self):
        pass
    
    def compute_guessing_entropy(
        self,
        key_probabilities: np.ndarray,
        actual_key: int
    ) -> float:
        sorted_indices = np.argsort(key_probabilities)[::-1]
        
        for rank, idx in enumerate(sorted_indices, start=1):
            if idx == actual_key:
                return np.log2(rank)
        
        return np.log2(len(key_probabilities))
    
    def compute_guessing_entropy_from_correlations(
        self,
        correlations: np.ndarray,
        actual_key: int
    ) -> float:
        sorted_indices = np.argsort(np.abs(correlations))[::-1]
        
        for rank, idx in enumerate(sorted_indices, start=1):
            if idx == actual_key:
                return np.log2(rank)
        
        return np.log2(len(correlations))
    
    def compute_partial_guessing_entropy(
        self,
        byte_correlations: List[np.ndarray],
        actual_key_bytes: np.ndarray
    ) -> float:
        total_ge = 0.0
        
        for i, (corr, actual_byte) in enumerate(zip(byte_correlations, actual_key_bytes)):
            ge = self.compute_guessing_entropy_from_correlations(corr, actual_byte)
            total_ge += ge
        
        return total_ge


class CorrelationMetrics:
    
    def __init__(self):
        pass
    
    def pearson_correlation(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        x_centered = x - np.mean(x, axis=0)
        y_centered = y - np.mean(y, axis=0)
        
        x_std = np.std(x, axis=0)
        y_std = np.std(y, axis=0)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            corr = np.sum(x_centered * y_centered, axis=0) / (len(x) * x_std * y_std)
            corr = np.nan_to_num(corr, nan=0.0)
        
        return corr
    
    def correlation_matrix(
        self,
        traces: np.ndarray,
        hypotheses: np.ndarray
    ) -> np.ndarray:
        n_traces, n_samples = traces.shape
        n_hypotheses = hypotheses.shape[1] if hypotheses.ndim > 1 else 1
        
        if hypotheses.ndim == 1:
            hypotheses = hypotheses.reshape(-1, 1)
        
        corr_matrix = np.zeros((n_hypotheses, n_samples))
        
        for i in range(n_hypotheses):
            hyp = hypotheses[:, i]
            for j in range(n_samples):
                corr_matrix[i, j] = self.pearson_correlation(
                    hyp.flatten(), traces[:, j].flatten()
                )[0]
        
        return corr_matrix
    
    def max_correlation_by_key(
        self,
        corr_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        max_corr = np.max(np.abs(corr_matrix), axis=1)
        max_idx = np.argmax(np.abs(corr_matrix), axis=1)
        return max_corr, max_idx


class MutualInformation:
    
    def __init__(self, num_bins: int = 256):
        self.num_bins = num_bins
    
    def compute_mi(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> float:
        x_discrete = self._discretize(x)
        y_discrete = self._discretize(y)
        
        joint_hist, _, _ = np.histogram2d(
            x_discrete, y_discrete, 
            bins=self.num_bins
        )
        
        joint_prob = joint_hist / np.sum(joint_hist)
        
        marginal_x = np.sum(joint_prob, axis=1)
        marginal_y = np.sum(joint_prob, axis=0)
        
        mi = 0.0
        for i in range(joint_prob.shape[0]):
            for j in range(joint_prob.shape[1]):
                if joint_prob[i, j] > 0 and marginal_x[i] > 0 and marginal_y[j] > 0:
                    mi += joint_prob[i, j] * np.log2(
                        joint_prob[i, j] / (marginal_x[i] * marginal_y[j])
                    )
        
        return mi
    
    def _discretize(self, x: np.ndarray) -> np.ndarray:
        if x.dtype in [np.int32, np.int64, np.uint8, np.uint16]:
            return x
        
        x_min, x_max = np.min(x), np.max(x)
        if x_max == x_min:
            return np.zeros_like(x, dtype=np.int32)
        
        normalized = (x - x_min) / (x_max - x_min)
        return (normalized * (self.num_bins - 1)).astype(np.int32)
    
    def mi_by_sample(
        self,
        traces: np.ndarray,
        labels: np.ndarray
    ) -> np.ndarray:
        n_samples = traces.shape[1]
        mi_values = np.zeros(n_samples)
        
        for i in range(n_samples):
            mi_values[i] = self.compute_mi(traces[:, i], labels)
        
        return mi_values


@dataclass
class AttackPerformanceMetrics:
    success_rate: float
    guessing_entropy: float
    key_rank: int
    num_traces_used: int
    execution_time_seconds: float = 0.0
    snr_max: float = 0.0
    correlation_max: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "success_rate": self.success_rate,
            "guessing_entropy": self.guessing_entropy,
            "key_rank": self.key_rank,
            "num_traces_used": self.num_traces_used,
            "execution_time_seconds": self.execution_time_seconds,
            "snr_max": self.snr_max,
            "correlation_max": self.correlation_max,
        }


class MetricsAggregator:
    
    def __init__(self):
        self.snr_calc = SNRCalculator()
        self.tvla = TVLAMetrics()
        self.success_rate = SuccessRateMetrics()
        self.guessing_entropy = GuessingEntropy()
        self.correlation = CorrelationMetrics()
        self.mi = MutualInformation()
    
    def compute_all_metrics(
        self,
        traces: np.ndarray,
        labels: np.ndarray,
        predictions: Optional[np.ndarray] = None,
        actual_values: Optional[np.ndarray] = None
    ) -> Dict[str, MetricResult]:
        results = {}
        
        snr = self.snr_calc.compute_snr(traces, labels)
        results["snr"] = MetricResult(
            MetricType.SNR, 
            snr,
            {"max_snr": float(np.max(snr)), "max_snr_index": int(np.argmax(snr))}
        )
        
        unique_labels = np.unique(labels)
        if len(unique_labels) == 2:
            group0 = traces[labels == unique_labels[0]]
            group1 = traces[labels == unique_labels[1]]
            t_stat = self.tvla.welch_t_test(group0, group1)
            summary = self.tvla.leakage_summary(t_stat)
            results["tvla"] = MetricResult(MetricType.TVLA, t_stat, summary)
        
        if predictions is not None and actual_values is not None:
            sr = self.success_rate.compute_success_rate(predictions, actual_values)
            results["success_rate"] = MetricResult(MetricType.SUCCESS_RATE, sr)
        
        mi = self.mi.mi_by_sample(traces, labels)
        results["mutual_information"] = MetricResult(
            MetricType.MUTUAL_INFORMATION,
            mi,
            {"max_mi": float(np.max(mi)), "max_mi_index": int(np.argmax(mi))}
        )
        
        return results
    
    def generate_report(
        self,
        metrics: Dict[str, MetricResult]
    ) -> str:
        lines = ["=" * 60, "SIDE-CHANNEL ANALYSIS METRICS REPORT", "=" * 60, ""]
        
        if "snr" in metrics:
            snr_meta = metrics["snr"].metadata
            lines.append(f"Signal-to-Noise Ratio (SNR):")
            lines.append(f"  Max SNR: {snr_meta.get('max_snr', 0):.4f}")
            lines.append(f"  Max SNR Index: {snr_meta.get('max_snr_index', 0)}")
            lines.append("")
        
        if "tvla" in metrics:
            tvla_meta = metrics["tvla"].metadata
            lines.append(f"Test Vector Leakage Assessment (TVLA):")
            lines.append(f"  Max |t-statistic|: {tvla_meta.get('max_t_statistic', 0):.4f}")
            lines.append(f"  Leakage detected (threshold=4.5): {tvla_meta.get('leakage_at_4.5', False)}")
            lines.append(f"  Number of leaking points: {tvla_meta.get('num_leaking_points_4.5', 0)}")
            lines.append("")
        
        if "success_rate" in metrics:
            sr = metrics["success_rate"].value
            lines.append(f"Attack Success Rate: {sr:.2%}")
            lines.append("")
        
        if "mutual_information" in metrics:
            mi_meta = metrics["mutual_information"].metadata
            lines.append(f"Mutual Information:")
            lines.append(f"  Max MI: {mi_meta.get('max_mi', 0):.4f}")
            lines.append(f"  Max MI Index: {mi_meta.get('max_mi_index', 0)}")
            lines.append("")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
