from typing import Optional, Union, Tuple, List, Callable
import numpy as np
from scipy import signal as scipy_signal
from scipy import ndimage

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


@njit(cache=True)
def _cross_correlation(ref: np.ndarray, target: np.ndarray) -> np.ndarray:
    n = len(ref)
    m = len(target)
    result_len = n + m - 1
    result = np.zeros(result_len, dtype=np.float64)
    
    for i in range(result_len):
        for j in range(n):
            k = i - j
            if 0 <= k < m:
                result[i] += ref[j] * target[k]
    
    return result


@njit(cache=True)
def _normalize_trace(trace: np.ndarray) -> np.ndarray:
    mean_val = np.mean(trace)
    std_val = np.std(trace)
    if std_val < 1e-10:
        return trace - mean_val
    return (trace - mean_val) / std_val


class TraceAligner:
    
    def __init__(self, reference: Optional[np.ndarray] = None,
                 method: str = "correlation",
                 search_window: Optional[int] = None):
        self._reference = reference
        self._method = method
        self._search_window = search_window
    
    @property
    def reference(self) -> Optional[np.ndarray]:
        return self._reference
    
    @reference.setter
    def reference(self, value: np.ndarray):
        self._reference = np.asarray(value, dtype=np.float64)
    
    @property
    def method(self) -> str:
        return self._method
    
    @method.setter
    def method(self, value: str):
        if value not in ("correlation", "peak", "threshold"):
            raise ValueError("method must be 'correlation', 'peak', or 'threshold'")
        self._method = value
    
    def _correlation_align(self, trace: np.ndarray) -> Tuple[np.ndarray, int]:
        if self._reference is None:
            return trace, 0
        
        ref = _normalize_trace(self._reference)
        target = _normalize_trace(trace)
        
        corr = np.correlate(target, ref, mode='full')
        
        if self._search_window is not None:
            center = len(corr) // 2
            start = max(0, center - self._search_window)
            end = min(len(corr), center + self._search_window)
            corr_window = corr[start:end]
            best_idx = start + np.argmax(corr_window)
        else:
            best_idx = np.argmax(corr)
        
        shift = best_idx - (len(self._reference) - 1)
        
        if shift > 0:
            aligned = np.concatenate([np.zeros(shift), trace[:-shift] if shift < len(trace) else trace])
        elif shift < 0:
            aligned = np.concatenate([trace[-shift:], np.zeros(-shift)])
        else:
            aligned = trace.copy()
        
        return aligned[:len(trace)], shift
    
    def _peak_align(self, trace: np.ndarray, 
                    peak_idx: Optional[int] = None) -> Tuple[np.ndarray, int]:
        if peak_idx is None:
            peak_idx = np.argmax(np.abs(trace))
        
        if self._reference is not None:
            ref_peak = np.argmax(np.abs(self._reference))
        else:
            ref_peak = len(trace) // 2
        
        shift = ref_peak - peak_idx
        
        if shift > 0:
            aligned = np.concatenate([np.zeros(shift), trace[:-shift] if shift < len(trace) else trace])
        elif shift < 0:
            aligned = np.concatenate([trace[-shift:], np.zeros(-shift)])
        else:
            aligned = trace.copy()
        
        return aligned[:len(trace)], shift
    
    def _threshold_align(self, trace: np.ndarray,
                         threshold: float = 0.5) -> Tuple[np.ndarray, int]:
        trace_max = np.max(np.abs(trace))
        if trace_max == 0:
            return trace, 0
        
        normalized = np.abs(trace) / trace_max
        trigger_idx = np.argmax(normalized > threshold)
        
        if self._reference is not None:
            ref_max = np.max(np.abs(self._reference))
            ref_normalized = np.abs(self._reference) / ref_max if ref_max > 0 else self._reference
            ref_trigger = np.argmax(ref_normalized > threshold)
        else:
            ref_trigger = len(trace) // 4
        
        shift = ref_trigger - trigger_idx
        
        if shift > 0:
            aligned = np.concatenate([np.zeros(shift), trace[:-shift] if shift < len(trace) else trace])
        elif shift < 0:
            aligned = np.concatenate([trace[-shift:], np.zeros(-shift)])
        else:
            aligned = trace.copy()
        
        return aligned[:len(trace)], shift
    
    def align(self, trace: np.ndarray, **kwargs) -> Tuple[np.ndarray, int]:
        trace = np.asarray(trace, dtype=np.float64)
        
        if self._method == "correlation":
            return self._correlation_align(trace)
        elif self._method == "peak":
            return self._peak_align(trace, kwargs.get("peak_idx"))
        elif self._method == "threshold":
            return self._threshold_align(trace, kwargs.get("threshold", 0.5))
        else:
            raise ValueError(f"Unknown alignment method: {self._method}")
    
    def align_traces(self, traces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        traces = np.asarray(traces, dtype=np.float64)
        
        if self._reference is None:
            self._reference = np.mean(traces, axis=0)
        
        aligned = np.zeros_like(traces)
        shifts = np.zeros(len(traces), dtype=np.int64)
        
        for i in range(len(traces)):
            aligned[i], shifts[i] = self.align(traces[i])
        
        return aligned, shifts


class TraceFilter:
    
    def __init__(self, sample_rate: float = 1e6):
        self._sample_rate = sample_rate
    
    @property
    def sample_rate(self) -> float:
        return self._sample_rate
    
    @sample_rate.setter
    def sample_rate(self, value: float):
        self._sample_rate = value
    
    def lowpass(self, trace: np.ndarray, cutoff: float, 
                order: int = 4) -> np.ndarray:
        nyquist = self._sample_rate / 2
        normalized_cutoff = cutoff / nyquist
        
        if normalized_cutoff >= 1:
            return trace
        
        b, a = scipy_signal.butter(order, normalized_cutoff, btype='low')
        return scipy_signal.filtfilt(b, a, trace)
    
    def highpass(self, trace: np.ndarray, cutoff: float,
                 order: int = 4) -> np.ndarray:
        nyquist = self._sample_rate / 2
        normalized_cutoff = cutoff / nyquist
        
        if normalized_cutoff <= 0:
            return trace
        
        b, a = scipy_signal.butter(order, normalized_cutoff, btype='high')
        return scipy_signal.filtfilt(b, a, trace)
    
    def bandpass(self, trace: np.ndarray, low_cutoff: float,
                 high_cutoff: float, order: int = 4) -> np.ndarray:
        nyquist = self._sample_rate / 2
        low = low_cutoff / nyquist
        high = high_cutoff / nyquist
        
        low = max(0.001, min(low, 0.999))
        high = max(0.001, min(high, 0.999))
        
        if low >= high:
            return trace
        
        b, a = scipy_signal.butter(order, [low, high], btype='band')
        return scipy_signal.filtfilt(b, a, trace)
    
    def notch(self, trace: np.ndarray, freq: float,
              quality: float = 30.0) -> np.ndarray:
        nyquist = self._sample_rate / 2
        normalized_freq = freq / nyquist
        
        if normalized_freq >= 1 or normalized_freq <= 0:
            return trace
        
        b, a = scipy_signal.iirnotch(normalized_freq, quality)
        return scipy_signal.filtfilt(b, a, trace)
    
    def moving_average(self, trace: np.ndarray, window: int) -> np.ndarray:
        if window < 1:
            return trace
        kernel = np.ones(window) / window
        return np.convolve(trace, kernel, mode='same')
    
    def median_filter(self, trace: np.ndarray, window: int) -> np.ndarray:
        if window < 1:
            return trace
        return ndimage.median_filter(trace, size=window)
    
    def savgol(self, trace: np.ndarray, window: int,
               order: int = 3) -> np.ndarray:
        if window < order + 2:
            window = order + 2
        if window % 2 == 0:
            window += 1
        return scipy_signal.savgol_filter(trace, window, order)
    
    def filter_traces(self, traces: np.ndarray, 
                      filter_func: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        filtered = np.zeros_like(traces)
        for i in range(len(traces)):
            filtered[i] = filter_func(traces[i])
        return filtered


class TraceNormalizer:
    
    @staticmethod
    def z_score(trace: np.ndarray) -> np.ndarray:
        mean_val = np.mean(trace)
        std_val = np.std(trace)
        if std_val < 1e-10:
            return trace - mean_val
        return (trace - mean_val) / std_val
    
    @staticmethod
    def min_max(trace: np.ndarray, 
                feature_range: Tuple[float, float] = (0, 1)) -> np.ndarray:
        min_val = np.min(trace)
        max_val = np.max(trace)
        
        if max_val - min_val < 1e-10:
            return np.full_like(trace, feature_range[0])
        
        scaled = (trace - min_val) / (max_val - min_val)
        return scaled * (feature_range[1] - feature_range[0]) + feature_range[0]
    
    @staticmethod
    def mean_center(trace: np.ndarray) -> np.ndarray:
        return trace - np.mean(trace)
    
    @staticmethod
    def normalize_traces(traces: np.ndarray,
                        method: str = "z_score") -> np.ndarray:
        normalized = np.zeros_like(traces)
        
        norm_func = {
            "z_score": TraceNormalizer.z_score,
            "min_max": TraceNormalizer.min_max,
            "mean_center": TraceNormalizer.mean_center,
        }.get(method, TraceNormalizer.z_score)
        
        for i in range(len(traces)):
            normalized[i] = norm_func(traces[i])
        
        return normalized


class TraceCompressor:
    
    @staticmethod
    def downsample(trace: np.ndarray, factor: int) -> np.ndarray:
        if factor < 1:
            return trace
        return trace[::factor]
    
    @staticmethod
    def average_downsample(trace: np.ndarray, factor: int) -> np.ndarray:
        if factor < 1:
            return trace
        
        trim_len = (len(trace) // factor) * factor
        trimmed = trace[:trim_len]
        reshaped = trimmed.reshape(-1, factor)
        return np.mean(reshaped, axis=1)
    
    @staticmethod
    def select_poi(trace: np.ndarray, 
                   poi_indices: Union[List[int], np.ndarray]) -> np.ndarray:
        return trace[poi_indices]
    
    @staticmethod
    def principal_components(traces: np.ndarray, 
                            n_components: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mean_trace = np.mean(traces, axis=0)
        centered = traces - mean_trace
        
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]
        
        components = eigenvectors[:, :n_components]
        projected = np.dot(centered, components)
        
        return projected, components, eigenvalues[:n_components]


class PointOfInterestSelector:
    
    def __init__(self, num_poi: int = 10):
        self._num_poi = num_poi
    
    @property
    def num_poi(self) -> int:
        return self._num_poi
    
    @num_poi.setter
    def num_poi(self, value: int):
        self._num_poi = value
    
    def select_by_variance(self, traces: np.ndarray) -> np.ndarray:
        variance = np.var(traces, axis=0)
        poi_indices = np.argsort(variance)[::-1][:self._num_poi]
        return np.sort(poi_indices)
    
    def select_by_snr(self, traces: np.ndarray, 
                      labels: np.ndarray) -> np.ndarray:
        unique_labels = np.unique(labels)
        
        class_means = np.zeros((len(unique_labels), traces.shape[1]))
        class_vars = np.zeros((len(unique_labels), traces.shape[1]))
        
        for i, label in enumerate(unique_labels):
            class_traces = traces[labels == label]
            class_means[i] = np.mean(class_traces, axis=0)
            class_vars[i] = np.var(class_traces, axis=0)
        
        signal_var = np.var(class_means, axis=0)
        noise_var = np.mean(class_vars, axis=0)
        
        snr = np.where(noise_var > 1e-10, signal_var / noise_var, 0)
        
        poi_indices = np.argsort(snr)[::-1][:self._num_poi]
        return np.sort(poi_indices)
    
    def select_by_correlation(self, traces: np.ndarray,
                              target: np.ndarray) -> np.ndarray:
        correlations = np.zeros(traces.shape[1])
        
        target_centered = target - np.mean(target)
        target_std = np.std(target)
        
        if target_std < 1e-10:
            return np.arange(self._num_poi)
        
        for i in range(traces.shape[1]):
            sample = traces[:, i]
            sample_centered = sample - np.mean(sample)
            sample_std = np.std(sample)
            
            if sample_std > 1e-10:
                correlations[i] = np.abs(np.mean(sample_centered * target_centered) / (sample_std * target_std))
        
        poi_indices = np.argsort(correlations)[::-1][:self._num_poi]
        return np.sort(poi_indices)
    
    def select_by_sost(self, traces: np.ndarray,
                       labels: np.ndarray) -> np.ndarray:
        unique_labels = np.unique(labels)
        
        class_means = {}
        for label in unique_labels:
            class_means[label] = np.mean(traces[labels == label], axis=0)
        
        global_mean = np.mean(traces, axis=0)
        
        sost = np.zeros(traces.shape[1])
        for label in unique_labels:
            n_k = np.sum(labels == label)
            sost += n_k * (class_means[label] - global_mean) ** 2
        
        poi_indices = np.argsort(sost)[::-1][:self._num_poi]
        return np.sort(poi_indices)


def preprocess_traces(traces: np.ndarray,
                      normalize: bool = True,
                      filter_type: Optional[str] = None,
                      filter_params: Optional[dict] = None,
                      align: bool = False,
                      compress_factor: int = 1,
                      poi_indices: Optional[np.ndarray] = None,
                      sample_rate: float = 1e6) -> np.ndarray:
    result = traces.copy()
    
    if normalize:
        result = TraceNormalizer.normalize_traces(result, method="z_score")
    
    if filter_type is not None:
        trace_filter = TraceFilter(sample_rate)
        filter_params = filter_params or {}
        
        if filter_type == "lowpass":
            cutoff = filter_params.get("cutoff", sample_rate / 4)
            result = trace_filter.filter_traces(
                result, lambda t: trace_filter.lowpass(t, cutoff))
        elif filter_type == "highpass":
            cutoff = filter_params.get("cutoff", sample_rate / 100)
            result = trace_filter.filter_traces(
                result, lambda t: trace_filter.highpass(t, cutoff))
        elif filter_type == "bandpass":
            low = filter_params.get("low_cutoff", sample_rate / 100)
            high = filter_params.get("high_cutoff", sample_rate / 4)
            result = trace_filter.filter_traces(
                result, lambda t: trace_filter.bandpass(t, low, high))
        elif filter_type == "moving_average":
            window = filter_params.get("window", 5)
            result = trace_filter.filter_traces(
                result, lambda t: trace_filter.moving_average(t, window))
    
    if align:
        aligner = TraceAligner(method="correlation")
        result, _ = aligner.align_traces(result)
    
    if compress_factor > 1:
        result = np.array([TraceCompressor.average_downsample(t, compress_factor) 
                          for t in result])
    
    if poi_indices is not None:
        result = result[:, poi_indices]
    
    return result
