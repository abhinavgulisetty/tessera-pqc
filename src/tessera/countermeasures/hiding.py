from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable, Union
import secrets
from enum import Enum


class NoiseType(Enum):
    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"
    LAPLACIAN = "laplacian"
    SHOT = "shot"


class HidingTechnique(Enum):
    NOISE_INJECTION = "noise_injection"
    AMPLITUDE_RANDOMIZATION = "amplitude_randomization"
    TEMPORAL_JITTER = "temporal_jitter"
    POWER_BALANCING = "power_balancing"
    DUAL_RAIL = "dual_rail"


@dataclass
class HidingConfig:
    technique: HidingTechnique = HidingTechnique.NOISE_INJECTION
    noise_type: NoiseType = NoiseType.GAUSSIAN
    noise_amplitude: float = 0.1
    temporal_jitter_samples: int = 10
    balance_target: float = 0.5


class NoiseInjector:
    
    def __init__(self, noise_type: NoiseType = NoiseType.GAUSSIAN, seed: Optional[int] = None):
        self.noise_type = noise_type
        self.rng = np.random.default_rng(seed)
    
    def generate_noise(self, shape: Tuple[int, ...], amplitude: float = 1.0) -> np.ndarray:
        if self.noise_type == NoiseType.GAUSSIAN:
            return self.rng.normal(0, amplitude, shape)
        elif self.noise_type == NoiseType.UNIFORM:
            return self.rng.uniform(-amplitude, amplitude, shape)
        elif self.noise_type == NoiseType.LAPLACIAN:
            return self.rng.laplace(0, amplitude / np.sqrt(2), shape)
        elif self.noise_type == NoiseType.SHOT:
            return self.rng.poisson(amplitude, shape).astype(float) - amplitude
        raise ValueError(f"Unknown noise type: {self.noise_type}")
    
    def inject_noise(self, signal: np.ndarray, snr_db: float) -> np.ndarray:
        signal_power = np.mean(signal ** 2)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        noise_amplitude = np.sqrt(noise_power)
        
        noise = self.generate_noise(signal.shape, noise_amplitude)
        return signal + noise
    
    def inject_additive_noise(self, signal: np.ndarray, amplitude: float) -> np.ndarray:
        noise = self.generate_noise(signal.shape, amplitude)
        return signal + noise
    
    def inject_multiplicative_noise(self, signal: np.ndarray, amplitude: float) -> np.ndarray:
        noise = self.generate_noise(signal.shape, amplitude)
        return signal * (1 + noise)


class AmplitudeRandomizer:
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
    
    def randomize_amplitude(
        self, 
        signal: np.ndarray, 
        min_scale: float = 0.8, 
        max_scale: float = 1.2
    ) -> np.ndarray:
        scale = self.rng.uniform(min_scale, max_scale)
        return signal * scale
    
    def randomize_dc_offset(
        self, 
        signal: np.ndarray, 
        max_offset: float = 0.1
    ) -> np.ndarray:
        offset = self.rng.uniform(-max_offset, max_offset)
        return signal + offset
    
    def randomize_per_sample(
        self, 
        signal: np.ndarray, 
        amplitude: float = 0.05
    ) -> np.ndarray:
        scales = self.rng.uniform(1 - amplitude, 1 + amplitude, signal.shape)
        return signal * scales


class TemporalJitter:
    
    def __init__(self, max_jitter_samples: int = 10, seed: Optional[int] = None):
        self.max_jitter_samples = max_jitter_samples
        self.rng = np.random.default_rng(seed)
    
    def apply_random_shift(self, signal: np.ndarray) -> np.ndarray:
        shift = self.rng.integers(-self.max_jitter_samples, self.max_jitter_samples + 1)
        return np.roll(signal, shift)
    
    def apply_random_stretch(
        self, 
        signal: np.ndarray, 
        min_factor: float = 0.95, 
        max_factor: float = 1.05
    ) -> np.ndarray:
        factor = self.rng.uniform(min_factor, max_factor)
        new_length = int(len(signal) * factor)
        x_old = np.linspace(0, 1, len(signal))
        x_new = np.linspace(0, 1, new_length)
        
        stretched = np.interp(x_new, x_old, signal)
        
        if new_length > len(signal):
            return stretched[:len(signal)]
        else:
            result = np.zeros_like(signal)
            result[:new_length] = stretched
            return result
    
    def apply_local_jitter(
        self, 
        signal: np.ndarray, 
        segment_size: int = 100
    ) -> np.ndarray:
        result = signal.copy()
        num_segments = len(signal) // segment_size
        
        for i in range(num_segments):
            start = i * segment_size
            end = min((i + 1) * segment_size, len(signal))
            jitter = self.rng.integers(-self.max_jitter_samples // 2, self.max_jitter_samples // 2 + 1)
            
            segment = result[start:end]
            result[start:end] = np.roll(segment, jitter)
        
        return result


class PowerBalancer:
    
    def __init__(self, target_power: float = 0.5):
        self.target_power = target_power
    
    def balance_hamming_weight(self, value: int, bit_width: int = 8) -> Tuple[int, int]:
        complement = (~value) & ((1 << bit_width) - 1)
        return value, complement
    
    def balance_array(self, values: np.ndarray, bit_width: int = 8) -> Tuple[np.ndarray, np.ndarray]:
        mask = (1 << bit_width) - 1
        complements = (~values) & mask
        return values.copy(), complements
    
    def create_balanced_encoding(self, value: int, bit_width: int = 8) -> np.ndarray:
        encoding = np.zeros(bit_width * 2, dtype=np.uint8)
        for i in range(bit_width):
            bit = (value >> i) & 1
            encoding[2 * i] = bit
            encoding[2 * i + 1] = 1 - bit
        return encoding
    
    def decode_balanced(self, encoding: np.ndarray) -> int:
        bit_width = len(encoding) // 2
        value = 0
        for i in range(bit_width):
            bit = encoding[2 * i]
            value |= (int(bit) << i)
        return value


class DualRailLogic:
    
    def __init__(self, bit_width: int = 64):
        self.bit_width = bit_width
    
    def encode(self, value: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mask = (1 << self.bit_width) - 1
        true_rail = value & mask
        false_rail = (~value) & mask
        return true_rail, false_rail
    
    def decode(self, true_rail: np.ndarray, false_rail: np.ndarray) -> np.ndarray:
        return true_rail
    
    def dual_and(
        self, 
        a: Tuple[np.ndarray, np.ndarray], 
        b: Tuple[np.ndarray, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        a_true, a_false = a
        b_true, b_false = b
        
        result_true = a_true & b_true
        result_false = a_false | b_false
        
        return result_true, result_false
    
    def dual_or(
        self, 
        a: Tuple[np.ndarray, np.ndarray], 
        b: Tuple[np.ndarray, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        a_true, a_false = a
        b_true, b_false = b
        
        result_true = a_true | b_true
        result_false = a_false & b_false
        
        return result_true, result_false
    
    def dual_xor(
        self, 
        a: Tuple[np.ndarray, np.ndarray], 
        b: Tuple[np.ndarray, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        a_true, a_false = a
        b_true, b_false = b
        
        result_true = (a_true & b_false) | (a_false & b_true)
        result_false = (a_true & b_true) | (a_false & b_false)
        
        return result_true, result_false
    
    def dual_not(
        self, 
        a: Tuple[np.ndarray, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        a_true, a_false = a
        return a_false, a_true


class HidingCountermeasure:
    
    def __init__(self, config: HidingConfig = None):
        self.config = config or HidingConfig()
        self._setup_components()
    
    def _setup_components(self):
        self.noise_injector = NoiseInjector(self.config.noise_type)
        self.amplitude_randomizer = AmplitudeRandomizer()
        self.temporal_jitter = TemporalJitter(self.config.temporal_jitter_samples)
        self.power_balancer = PowerBalancer(self.config.balance_target)
        self.dual_rail = DualRailLogic()
    
    def apply(self, signal: np.ndarray) -> np.ndarray:
        if self.config.technique == HidingTechnique.NOISE_INJECTION:
            return self.noise_injector.inject_additive_noise(
                signal, self.config.noise_amplitude
            )
        elif self.config.technique == HidingTechnique.AMPLITUDE_RANDOMIZATION:
            return self.amplitude_randomizer.randomize_amplitude(signal)
        elif self.config.technique == HidingTechnique.TEMPORAL_JITTER:
            return self.temporal_jitter.apply_random_shift(signal)
        elif self.config.technique == HidingTechnique.POWER_BALANCING:
            return signal
        elif self.config.technique == HidingTechnique.DUAL_RAIL:
            return signal
        return signal
    
    def apply_all(self, signal: np.ndarray) -> np.ndarray:
        result = signal.copy()
        result = self.noise_injector.inject_additive_noise(
            result, self.config.noise_amplitude
        )
        result = self.amplitude_randomizer.randomize_amplitude(result)
        result = self.temporal_jitter.apply_random_shift(result)
        return result


class OperationHider:
    
    def __init__(self, config: HidingConfig = None):
        self.config = config or HidingConfig()
        self.noise_injector = NoiseInjector(self.config.noise_type)
    
    def hide_memory_access(
        self, 
        array: np.ndarray, 
        index: int, 
        num_dummy_accesses: int = 4
    ) -> np.ndarray:
        dummy_indices = [secrets.randbelow(len(array)) for _ in range(num_dummy_accesses)]
        real_position = secrets.randbelow(num_dummy_accesses + 1)
        
        accesses = dummy_indices[:real_position] + [index] + dummy_indices[real_position:]
        
        results = []
        for idx in accesses:
            results.append(array[idx])
        
        return results[real_position]
    
    def hide_conditional(
        self, 
        condition: bool, 
        true_value: np.ndarray, 
        false_value: np.ndarray
    ) -> np.ndarray:
        both_computed = [true_value.copy(), false_value.copy()]
        
        true_mask = int(condition)
        false_mask = 1 - true_mask
        
        result = (both_computed[0] * true_mask + both_computed[1] * false_mask)
        return result
    
    def constant_time_select(
        self, 
        condition: int, 
        true_value: np.ndarray, 
        false_value: np.ndarray
    ) -> np.ndarray:
        mask = -condition
        return (true_value & mask) | (false_value & ~mask)


class TraceHider:
    
    def __init__(self, config: HidingConfig = None):
        self.config = config or HidingConfig()
        self.hiding = HidingCountermeasure(config)
    
    def hide_trace(self, trace: np.ndarray) -> np.ndarray:
        return self.hiding.apply_all(trace)
    
    def hide_traces(self, traces: np.ndarray) -> np.ndarray:
        hidden = np.empty_like(traces)
        for i in range(len(traces)):
            hidden[i] = self.hide_trace(traces[i])
        return hidden
    
    def add_dummy_traces(
        self, 
        traces: np.ndarray, 
        num_dummy: int, 
        insert_randomly: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        mean_trace = np.mean(traces, axis=0)
        std_trace = np.std(traces, axis=0)
        
        dummy_traces = np.random.normal(
            mean_trace, std_trace, 
            (num_dummy, traces.shape[1])
        )
        
        if insert_randomly:
            combined = np.vstack([traces, dummy_traces])
            perm = np.random.permutation(len(combined))
            real_indices = np.where(perm < len(traces))[0]
            return combined[perm], real_indices
        else:
            real_indices = np.arange(len(traces))
            return np.vstack([traces, dummy_traces]), real_indices


@dataclass
class CombinedHidingConfig:
    noise_amplitude: float = 0.1
    noise_type: NoiseType = NoiseType.GAUSSIAN
    amplitude_randomization: bool = True
    min_amplitude_scale: float = 0.9
    max_amplitude_scale: float = 1.1
    temporal_jitter: bool = True
    max_jitter_samples: int = 5
    use_dual_rail: bool = False


class CombinedHiding:
    
    def __init__(self, config: CombinedHidingConfig = None):
        self.config = config or CombinedHidingConfig()
        self.noise_injector = NoiseInjector(self.config.noise_type)
        self.amplitude_randomizer = AmplitudeRandomizer()
        self.temporal_jitter = TemporalJitter(self.config.max_jitter_samples)
        self.dual_rail = DualRailLogic() if self.config.use_dual_rail else None
    
    def apply(self, signal: np.ndarray) -> np.ndarray:
        result = signal.copy()
        
        result = self.noise_injector.inject_additive_noise(
            result, self.config.noise_amplitude
        )
        
        if self.config.amplitude_randomization:
            result = self.amplitude_randomizer.randomize_amplitude(
                result,
                self.config.min_amplitude_scale,
                self.config.max_amplitude_scale
            )
        
        if self.config.temporal_jitter:
            result = self.temporal_jitter.apply_random_shift(result)
        
        return result
    
    def apply_batch(self, signals: np.ndarray) -> np.ndarray:
        result = np.empty_like(signals)
        for i in range(len(signals)):
            result[i] = self.apply(signals[i])
        return result
