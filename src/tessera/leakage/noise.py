from typing import Optional, Union, Tuple
import numpy as np

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
def _add_gaussian_noise_inplace(samples: np.ndarray, std: float, seed: int):
    np.random.seed(seed)
    for i in prange(len(samples)):
        samples[i] += np.random.normal(0, std)


class NoiseGenerator:
    
    def __init__(self, seed: Optional[int] = None):
        self._rng = np.random.default_rng(seed)
        self._seed = seed
    
    @property
    def seed(self) -> Optional[int]:
        return self._seed
    
    def set_seed(self, seed: int):
        self._seed = seed
        self._rng = np.random.default_rng(seed)
    
    def gaussian(self, shape: Union[int, Tuple[int, ...]], 
                 mean: float = 0.0, std: float = 1.0) -> np.ndarray:
        return self._rng.normal(mean, std, shape).astype(np.float64)
    
    def uniform(self, shape: Union[int, Tuple[int, ...]], 
                low: float = 0.0, high: float = 1.0) -> np.ndarray:
        return self._rng.uniform(low, high, shape).astype(np.float64)
    
    def laplace(self, shape: Union[int, Tuple[int, ...]], 
                loc: float = 0.0, scale: float = 1.0) -> np.ndarray:
        return self._rng.laplace(loc, scale, shape).astype(np.float64)
    
    def exponential(self, shape: Union[int, Tuple[int, ...]], 
                    scale: float = 1.0) -> np.ndarray:
        return self._rng.exponential(scale, shape).astype(np.float64)


class SNRController:
    
    def __init__(self, target_snr_db: Optional[float] = None):
        self._target_snr_db = target_snr_db
        self._noise_gen = NoiseGenerator()
    
    @property
    def target_snr_db(self) -> Optional[float]:
        return self._target_snr_db
    
    @target_snr_db.setter
    def target_snr_db(self, value: Optional[float]):
        self._target_snr_db = value
    
    def set_seed(self, seed: int):
        self._noise_gen.set_seed(seed)
    
    @staticmethod
    def calculate_snr(signal: np.ndarray, noise: np.ndarray) -> float:
        signal_power = np.mean(signal ** 2)
        noise_power = np.mean(noise ** 2)
        if noise_power == 0:
            return float('inf')
        return 10 * np.log10(signal_power / noise_power)
    
    @staticmethod
    def calculate_snr_from_traces(traces: np.ndarray) -> np.ndarray:
        mean_trace = np.mean(traces, axis=0)
        noise = traces - mean_trace
        
        signal_var = np.var(mean_trace)
        noise_var = np.mean(np.var(noise, axis=0))
        
        if noise_var == 0:
            return np.full(traces.shape[1], float('inf'))
        
        snr_per_sample = np.var(traces, axis=0) / noise_var
        return 10 * np.log10(np.maximum(snr_per_sample, 1e-10))
    
    def noise_std_for_snr(self, signal: np.ndarray, target_snr_db: float) -> float:
        signal_power = np.mean(signal ** 2)
        target_noise_power = signal_power / (10 ** (target_snr_db / 10))
        return np.sqrt(target_noise_power)
    
    def add_noise_for_snr(self, signal: np.ndarray, 
                          target_snr_db: Optional[float] = None) -> Tuple[np.ndarray, float]:
        if target_snr_db is None:
            target_snr_db = self._target_snr_db
        if target_snr_db is None:
            raise ValueError("target_snr_db must be specified")
        
        noise_std = self.noise_std_for_snr(signal, target_snr_db)
        noise = self._noise_gen.gaussian(signal.shape, std=noise_std)
        noisy_signal = signal + noise
        
        return noisy_signal, noise_std
    
    def add_noise_std(self, signal: np.ndarray, noise_std: float) -> np.ndarray:
        noise = self._noise_gen.gaussian(signal.shape, std=noise_std)
        return signal + noise


class ElectronicNoise:
    
    def __init__(self, thermal_std: float = 0.01,
                 shot_scale: float = 0.001,
                 flicker_amplitude: float = 0.005,
                 quantization_levels: Optional[int] = None):
        self._thermal_std = thermal_std
        self._shot_scale = shot_scale
        self._flicker_amplitude = flicker_amplitude
        self._quantization_levels = quantization_levels
        self._noise_gen = NoiseGenerator()
    
    @property
    def thermal_std(self) -> float:
        return self._thermal_std
    
    @thermal_std.setter
    def thermal_std(self, value: float):
        self._thermal_std = value
    
    @property
    def shot_scale(self) -> float:
        return self._shot_scale
    
    @shot_scale.setter
    def shot_scale(self, value: float):
        self._shot_scale = value
    
    @property
    def flicker_amplitude(self) -> float:
        return self._flicker_amplitude
    
    @flicker_amplitude.setter
    def flicker_amplitude(self, value: float):
        self._flicker_amplitude = value
    
    def set_seed(self, seed: int):
        self._noise_gen.set_seed(seed)
    
    def thermal_noise(self, shape: Union[int, Tuple[int, ...]]) -> np.ndarray:
        return self._noise_gen.gaussian(shape, std=self._thermal_std)
    
    def shot_noise(self, signal: np.ndarray) -> np.ndarray:
        return self._noise_gen.gaussian(signal.shape, std=self._shot_scale * np.sqrt(np.abs(signal) + 1e-10))
    
    def flicker_noise(self, num_samples: int) -> np.ndarray:
        white_noise = self._noise_gen.gaussian(num_samples * 2)
        
        fft_noise = np.fft.rfft(white_noise)
        freqs = np.fft.rfftfreq(len(white_noise))
        freqs[0] = 1e-10
        fft_noise = fft_noise / np.sqrt(freqs)
        pink_noise = np.fft.irfft(fft_noise)[:num_samples]
        
        pink_noise = pink_noise / np.std(pink_noise) * self._flicker_amplitude
        return pink_noise
    
    def quantization_noise(self, signal: np.ndarray) -> np.ndarray:
        if self._quantization_levels is None:
            return np.zeros_like(signal)
        
        signal_range = np.max(signal) - np.min(signal)
        if signal_range == 0:
            return np.zeros_like(signal)
        
        step = signal_range / self._quantization_levels
        quantized = np.round((signal - np.min(signal)) / step) * step + np.min(signal)
        return quantized - signal
    
    def apply_all(self, signal: np.ndarray, 
                  include_thermal: bool = True,
                  include_shot: bool = True,
                  include_flicker: bool = True,
                  include_quantization: bool = True) -> np.ndarray:
        result = signal.copy()
        
        if include_thermal and self._thermal_std > 0:
            result += self.thermal_noise(signal.shape)
        
        if include_shot and self._shot_scale > 0:
            result += self.shot_noise(signal)
        
        if include_flicker and self._flicker_amplitude > 0:
            if signal.ndim == 1:
                result += self.flicker_noise(len(signal))
            else:
                for i in range(signal.shape[0]):
                    result[i] += self.flicker_noise(signal.shape[1])
        
        if include_quantization and self._quantization_levels is not None:
            result += self.quantization_noise(result)
        
        return result


class EnvironmentalNoise:
    
    def __init__(self, em_frequency: float = 50.0,
                 em_amplitude: float = 0.01,
                 sample_rate: float = 1e6,
                 drift_rate: float = 0.0001):
        self._em_frequency = em_frequency
        self._em_amplitude = em_amplitude
        self._sample_rate = sample_rate
        self._drift_rate = drift_rate
        self._noise_gen = NoiseGenerator()
    
    @property
    def em_frequency(self) -> float:
        return self._em_frequency
    
    @em_frequency.setter
    def em_frequency(self, value: float):
        self._em_frequency = value
    
    @property
    def em_amplitude(self) -> float:
        return self._em_amplitude
    
    @em_amplitude.setter
    def em_amplitude(self, value: float):
        self._em_amplitude = value
    
    def set_seed(self, seed: int):
        self._noise_gen.set_seed(seed)
    
    def em_interference(self, num_samples: int, 
                        phase_offset: Optional[float] = None) -> np.ndarray:
        if phase_offset is None:
            phase_offset = self._noise_gen.uniform(1, 0, 2 * np.pi)[0]
        
        t = np.arange(num_samples) / self._sample_rate
        return self._em_amplitude * np.sin(2 * np.pi * self._em_frequency * t + phase_offset)
    
    def temperature_drift(self, num_samples: int) -> np.ndarray:
        drift = np.cumsum(self._noise_gen.gaussian(num_samples, std=self._drift_rate))
        return drift
    
    def apply_all(self, signal: np.ndarray,
                  include_em: bool = True,
                  include_drift: bool = True) -> np.ndarray:
        result = signal.copy()
        
        if signal.ndim == 1:
            if include_em and self._em_amplitude > 0:
                result += self.em_interference(len(signal))
            if include_drift and self._drift_rate > 0:
                result += self.temperature_drift(len(signal))
        else:
            for i in range(signal.shape[0]):
                if include_em and self._em_amplitude > 0:
                    result[i] += self.em_interference(signal.shape[1])
                if include_drift and self._drift_rate > 0:
                    result[i] += self.temperature_drift(signal.shape[1])
        
        return result


class RealisticNoiseModel:
    
    def __init__(self, target_snr_db: float = 10.0,
                 electronic_noise: Optional[ElectronicNoise] = None,
                 environmental_noise: Optional[EnvironmentalNoise] = None):
        self._snr_controller = SNRController(target_snr_db)
        self._electronic = electronic_noise or ElectronicNoise()
        self._environmental = environmental_noise or EnvironmentalNoise()
    
    @property
    def target_snr_db(self) -> float:
        return self._snr_controller.target_snr_db or 10.0
    
    @target_snr_db.setter
    def target_snr_db(self, value: float):
        self._snr_controller.target_snr_db = value
    
    def set_seed(self, seed: int):
        self._snr_controller.set_seed(seed)
        self._electronic.set_seed(seed + 1)
        self._environmental.set_seed(seed + 2)
    
    def apply(self, signal: np.ndarray,
              use_snr_control: bool = True,
              use_electronic: bool = True,
              use_environmental: bool = True) -> np.ndarray:
        result = signal.copy()
        
        if use_snr_control:
            result, _ = self._snr_controller.add_noise_for_snr(result)
        
        if use_electronic:
            result = self._electronic.apply_all(result)
        
        if use_environmental:
            result = self._environmental.apply_all(result)
        
        return result
