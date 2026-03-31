from typing import Optional, Callable, Tuple
import numpy as np
from scipy import stats

from .base import (
    LeakageAssessment,
    LeakageAssessmentResult,
    welch_t_statistic,
)
from ..leakage import TraceSet


class WelchTTest(LeakageAssessment):
    
    def __init__(self, confidence_level: float = 0.99999,
                 two_tailed: bool = True):
        super().__init__(confidence_level)
        self._two_tailed = two_tailed
        self._threshold = self._compute_threshold()
    
    @property
    def two_tailed(self) -> bool:
        return self._two_tailed
    
    @two_tailed.setter
    def two_tailed(self, value: bool):
        self._two_tailed = value
        self._threshold = self._compute_threshold()
    
    @property
    def threshold(self) -> float:
        return self._threshold
    
    def _compute_threshold(self) -> float:
        if self._two_tailed:
            alpha = (1 - self._confidence_level) / 2
        else:
            alpha = 1 - self._confidence_level
        return stats.norm.ppf(1 - alpha)
    
    def assess(self, group1: np.ndarray, group2: np.ndarray) -> LeakageAssessmentResult:
        t_stat = welch_t_statistic(group1, group2)
        
        if self._two_tailed:
            leakage_mask = np.abs(t_stat) > self._threshold
        else:
            leakage_mask = t_stat > self._threshold
        
        max_stat = float(np.max(np.abs(t_stat)))
        leakage_detected = max_stat > self._threshold
        
        poi_indices = np.where(leakage_mask)[0]
        
        self._result = LeakageAssessmentResult(
            leakage_detected=leakage_detected,
            max_statistic=max_stat,
            threshold=self._threshold,
            poi_indices=poi_indices,
            statistic_per_sample=t_stat,
            confidence_level=self._confidence_level,
            num_traces_used=len(group1) + len(group2),
        )
        
        return self._result


class FixedVsRandomTVLA(LeakageAssessment):
    
    def __init__(self, confidence_level: float = 0.99999,
                 order: int = 1):
        super().__init__(confidence_level)
        self._order = order
        self._ttest = WelchTTest(confidence_level, two_tailed=True)
    
    @property
    def order(self) -> int:
        return self._order
    
    @order.setter
    def order(self, value: int):
        if value < 1:
            raise ValueError("order must be >= 1")
        self._order = value
    
    def _preprocess_order(self, traces: np.ndarray) -> np.ndarray:
        if self._order == 1:
            return traces
        
        mean = np.mean(traces, axis=0)
        centered = traces - mean
        
        if self._order == 2:
            return centered ** 2
        elif self._order == 3:
            return centered ** 3
        else:
            return centered ** self._order
    
    def assess(self, group1: np.ndarray, group2: np.ndarray) -> LeakageAssessmentResult:
        processed1 = self._preprocess_order(group1)
        processed2 = self._preprocess_order(group2)
        
        return self._ttest.assess(processed1, processed2)
    
    def assess_from_traceset(self, traces: TraceSet,
                             fixed_value: np.ndarray,
                             plaintext_selector: Optional[Callable] = None) -> LeakageAssessmentResult:
        samples = traces.get_samples_matrix()
        plaintexts = traces.get_plaintexts()
        
        if plaintexts is None:
            raise ValueError("TraceSet must contain plaintexts for fixed vs random TVLA")
        
        if plaintext_selector is None:
            is_fixed = np.all(plaintexts == fixed_value, axis=1)
        else:
            is_fixed = np.array([plaintext_selector(pt, fixed_value) for pt in plaintexts])
        
        fixed_traces = samples[is_fixed]
        random_traces = samples[~is_fixed]
        
        return self.assess(fixed_traces, random_traces)


class SemiFixedTVLA(LeakageAssessment):
    
    def __init__(self, confidence_level: float = 0.99999,
                 bit_index: int = 0):
        super().__init__(confidence_level)
        self._bit_index = bit_index
        self._ttest = WelchTTest(confidence_level, two_tailed=True)
    
    @property
    def bit_index(self) -> int:
        return self._bit_index
    
    @bit_index.setter
    def bit_index(self, value: int):
        self._bit_index = value
    
    def assess(self, group1: np.ndarray, group2: np.ndarray) -> LeakageAssessmentResult:
        return self._ttest.assess(group1, group2)
    
    def assess_by_intermediate(self, traces: TraceSet,
                               intermediate_values: np.ndarray) -> LeakageAssessmentResult:
        samples = traces.get_samples_matrix()
        
        bit_values = (intermediate_values >> self._bit_index) & 1
        
        group0 = samples[bit_values == 0]
        group1 = samples[bit_values == 1]
        
        return self.assess(group0, group1)


class NonSpecificTVLA(LeakageAssessment):
    
    def __init__(self, confidence_level: float = 0.99999,
                 num_random_tests: int = 100):
        super().__init__(confidence_level)
        self._num_random_tests = num_random_tests
        self._ttest = WelchTTest(confidence_level, two_tailed=True)
    
    @property
    def num_random_tests(self) -> int:
        return self._num_random_tests
    
    @num_random_tests.setter
    def num_random_tests(self, value: int):
        self._num_random_tests = value
    
    def assess(self, group1: np.ndarray, group2: np.ndarray) -> LeakageAssessmentResult:
        return self._ttest.assess(group1, group2)
    
    def assess_random_splits(self, traces: np.ndarray,
                             seed: Optional[int] = None) -> Tuple[LeakageAssessmentResult, np.ndarray]:
        rng = np.random.default_rng(seed)
        
        all_t_stats = np.zeros((self._num_random_tests, traces.shape[1]))
        max_stats = np.zeros(self._num_random_tests)
        
        for i in range(self._num_random_tests):
            indices = rng.permutation(len(traces))
            mid = len(traces) // 2
            group1 = traces[indices[:mid]]
            group2 = traces[indices[mid:]]
            
            t_stat = welch_t_statistic(group1, group2)
            all_t_stats[i] = t_stat
            max_stats[i] = np.max(np.abs(t_stat))
        
        mean_t_stat = np.mean(all_t_stats, axis=0)
        threshold = self._ttest.threshold
        
        leakage_mask = np.abs(mean_t_stat) > threshold
        max_stat = float(np.max(np.abs(mean_t_stat)))
        
        self._result = LeakageAssessmentResult(
            leakage_detected=max_stat > threshold,
            max_statistic=max_stat,
            threshold=threshold,
            poi_indices=np.where(leakage_mask)[0],
            statistic_per_sample=mean_t_stat,
            confidence_level=self._confidence_level,
            num_traces_used=len(traces),
        )
        
        return self._result, max_stats


class IncrementalTTest(LeakageAssessment):
    
    def __init__(self, confidence_level: float = 0.99999,
                 num_samples: Optional[int] = None):
        super().__init__(confidence_level)
        self._num_samples = num_samples
        self._n1 = 0
        self._n2 = 0
        self._mean1: Optional[np.ndarray] = None
        self._mean2: Optional[np.ndarray] = None
        self._m2_1: Optional[np.ndarray] = None
        self._m2_2: Optional[np.ndarray] = None
        self._ttest = WelchTTest(confidence_level, two_tailed=True)
    
    def reset(self):
        super().reset()
        self._n1 = 0
        self._n2 = 0
        self._mean1 = None
        self._mean2 = None
        self._m2_1 = None
        self._m2_2 = None
    
    def update_group1(self, traces: np.ndarray):
        for trace in traces:
            self._update_stats(trace, group=1)
    
    def update_group2(self, traces: np.ndarray):
        for trace in traces:
            self._update_stats(trace, group=2)
    
    def _update_stats(self, trace: np.ndarray, group: int):
        if group == 1:
            self._n1 += 1
            if self._mean1 is None:
                self._mean1 = trace.copy().astype(np.float64)
                self._m2_1 = np.zeros_like(trace, dtype=np.float64)
            else:
                delta = trace - self._mean1
                self._mean1 += delta / self._n1
                delta2 = trace - self._mean1
                self._m2_1 += delta * delta2
        else:
            self._n2 += 1
            if self._mean2 is None:
                self._mean2 = trace.copy().astype(np.float64)
                self._m2_2 = np.zeros_like(trace, dtype=np.float64)
            else:
                delta = trace - self._mean2
                self._mean2 += delta / self._n2
                delta2 = trace - self._mean2
                self._m2_2 += delta * delta2
    
    def assess(self, group1: np.ndarray = None, group2: np.ndarray = None) -> LeakageAssessmentResult:
        if group1 is not None and group2 is not None:
            self.reset()
            self.update_group1(group1)
            self.update_group2(group2)
        
        if self._n1 < 2 or self._n2 < 2:
            raise ValueError("Need at least 2 samples in each group")
        
        var1 = self._m2_1 / (self._n1 - 1)
        var2 = self._m2_2 / (self._n2 - 1)
        
        se = np.sqrt(var1 / self._n1 + var2 / self._n2)
        se = np.where(se < 1e-10, 1e-10, se)
        
        t_stat = (self._mean1 - self._mean2) / se
        
        threshold = self._ttest.threshold
        leakage_mask = np.abs(t_stat) > threshold
        max_stat = float(np.max(np.abs(t_stat)))
        
        self._result = LeakageAssessmentResult(
            leakage_detected=max_stat > threshold,
            max_statistic=max_stat,
            threshold=threshold,
            poi_indices=np.where(leakage_mask)[0],
            statistic_per_sample=t_stat,
            confidence_level=self._confidence_level,
            num_traces_used=self._n1 + self._n2,
        )
        
        return self._result


def run_tvla(traces: TraceSet,
             method: str = "fixed_vs_random",
             confidence_level: float = 0.99999,
             **kwargs) -> LeakageAssessmentResult:
    if method == "fixed_vs_random":
        fixed_value = kwargs.get("fixed_value")
        if fixed_value is None:
            raise ValueError("fixed_value must be provided for fixed_vs_random TVLA")
        
        tvla = FixedVsRandomTVLA(confidence_level, order=kwargs.get("order", 1))
        return tvla.assess_from_traceset(traces, fixed_value)
    
    elif method == "semi_fixed":
        intermediate_values = kwargs.get("intermediate_values")
        if intermediate_values is None:
            raise ValueError("intermediate_values must be provided for semi_fixed TVLA")
        
        tvla = SemiFixedTVLA(confidence_level, bit_index=kwargs.get("bit_index", 0))
        return tvla.assess_by_intermediate(traces, intermediate_values)
    
    elif method == "non_specific":
        samples = traces.get_samples_matrix()
        tvla = NonSpecificTVLA(
            confidence_level, 
            num_random_tests=kwargs.get("num_random_tests", 100)
        )
        result, _ = tvla.assess_random_splits(samples, seed=kwargs.get("seed"))
        return result
    
    else:
        raise ValueError(f"Unknown TVLA method: {method}")
