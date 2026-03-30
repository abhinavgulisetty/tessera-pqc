from abc import ABC, abstractmethod
from typing import Union, List, Optional, Callable
import numpy as np
from enum import Enum

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


class LeakageType(Enum):
    HAMMING_WEIGHT = "hamming_weight"
    HAMMING_DISTANCE = "hamming_distance"
    IDENTITY = "identity"
    LINEAR = "linear"
    CUSTOM = "custom"


@njit(cache=True)
def _hamming_weight_u8(value: int) -> int:
    count = 0
    for _ in range(8):
        count += value & 1
        value >>= 1
    return count


@njit(cache=True)
def _hamming_weight_u16(value: int) -> int:
    count = 0
    for _ in range(16):
        count += value & 1
        value >>= 1
    return count


@njit(cache=True)
def _hamming_weight_u32(value: int) -> int:
    count = 0
    for _ in range(32):
        count += value & 1
        value >>= 1
    return count


@njit(cache=True)
def _hamming_weight_u64(value: int) -> int:
    count = 0
    for _ in range(64):
        count += value & 1
        value >>= 1
    return count


@njit(cache=True)
def _hamming_weight_array_u8(arr: np.ndarray) -> np.ndarray:
    result = np.zeros(len(arr), dtype=np.float64)
    for i in prange(len(arr)):
        result[i] = _hamming_weight_u8(arr[i])
    return result


@njit(cache=True)
def _hamming_weight_array_u16(arr: np.ndarray) -> np.ndarray:
    result = np.zeros(len(arr), dtype=np.float64)
    for i in prange(len(arr)):
        result[i] = _hamming_weight_u16(arr[i])
    return result


@njit(cache=True)
def _hamming_weight_array_u32(arr: np.ndarray) -> np.ndarray:
    result = np.zeros(len(arr), dtype=np.float64)
    for i in prange(len(arr)):
        result[i] = _hamming_weight_u32(arr[i])
    return result


@njit(cache=True)
def _hamming_distance_arrays(arr1: np.ndarray, arr2: np.ndarray, bits: int) -> np.ndarray:
    result = np.zeros(len(arr1), dtype=np.float64)
    for i in prange(len(arr1)):
        xor_val = arr1[i] ^ arr2[i]
        count = 0
        for _ in range(bits):
            count += xor_val & 1
            xor_val >>= 1
        result[i] = count
    return result


class LeakageModel(ABC):
    
    @abstractmethod
    def __call__(self, data: Union[int, np.ndarray], 
                 prev_data: Optional[Union[int, np.ndarray]] = None) -> Union[float, np.ndarray]:
        pass
    
    @property
    @abstractmethod
    def leakage_type(self) -> LeakageType:
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass


class HammingWeightModel(LeakageModel):
    
    def __init__(self, bit_width: int = 8):
        if bit_width not in (8, 16, 32, 64):
            raise ValueError("bit_width must be 8, 16, 32, or 64")
        self._bit_width = bit_width
        self._hw_func = {
            8: _hamming_weight_u8,
            16: _hamming_weight_u16,
            32: _hamming_weight_u32,
            64: _hamming_weight_u64,
        }[bit_width]
        self._hw_array_func = {
            8: _hamming_weight_array_u8,
            16: _hamming_weight_array_u16,
            32: _hamming_weight_array_u32,
            64: None,
        }[bit_width]
    
    @property
    def bit_width(self) -> int:
        return self._bit_width
    
    @property
    def leakage_type(self) -> LeakageType:
        return LeakageType.HAMMING_WEIGHT
    
    @property
    def name(self) -> str:
        return f"HW{self._bit_width}"
    
    def __call__(self, data: Union[int, np.ndarray],
                 prev_data: Optional[Union[int, np.ndarray]] = None) -> Union[float, np.ndarray]:
        if isinstance(data, (int, np.integer)):
            return float(self._hw_func(int(data)))
        
        data = np.asarray(data)
        if self._hw_array_func is not None:
            return self._hw_array_func(data.astype(np.uint64))
        
        result = np.zeros(len(data), dtype=np.float64)
        for i, val in enumerate(data):
            result[i] = self._hw_func(int(val))
        return result


class HammingDistanceModel(LeakageModel):
    
    def __init__(self, bit_width: int = 8, reference: Optional[Union[int, np.ndarray]] = None):
        if bit_width not in (8, 16, 32, 64):
            raise ValueError("bit_width must be 8, 16, 32, or 64")
        self._bit_width = bit_width
        self._reference = reference
        self._hw_model = HammingWeightModel(bit_width)
    
    @property
    def bit_width(self) -> int:
        return self._bit_width
    
    @property
    def reference(self) -> Optional[Union[int, np.ndarray]]:
        return self._reference
    
    @reference.setter
    def reference(self, value: Optional[Union[int, np.ndarray]]):
        self._reference = value
    
    @property
    def leakage_type(self) -> LeakageType:
        return LeakageType.HAMMING_DISTANCE
    
    @property
    def name(self) -> str:
        return f"HD{self._bit_width}"
    
    def __call__(self, data: Union[int, np.ndarray],
                 prev_data: Optional[Union[int, np.ndarray]] = None) -> Union[float, np.ndarray]:
        ref = prev_data if prev_data is not None else self._reference
        if ref is None:
            ref = 0
        
        if isinstance(data, (int, np.integer)):
            xor_val = int(data) ^ int(ref)
            return self._hw_model(xor_val)
        
        data = np.asarray(data)
        if isinstance(ref, (int, np.integer)):
            ref = np.full_like(data, ref)
        ref = np.asarray(ref)
        
        return _hamming_distance_arrays(
            data.astype(np.uint64), 
            ref.astype(np.uint64), 
            self._bit_width
        )


class IdentityModel(LeakageModel):
    
    def __init__(self, scale: float = 1.0, offset: float = 0.0):
        self._scale = scale
        self._offset = offset
    
    @property
    def scale(self) -> float:
        return self._scale
    
    @property
    def offset(self) -> float:
        return self._offset
    
    @property
    def leakage_type(self) -> LeakageType:
        return LeakageType.IDENTITY
    
    @property
    def name(self) -> str:
        return "Identity"
    
    def __call__(self, data: Union[int, np.ndarray],
                 prev_data: Optional[Union[int, np.ndarray]] = None) -> Union[float, np.ndarray]:
        if isinstance(data, (int, np.integer)):
            return float(data) * self._scale + self._offset
        return np.asarray(data, dtype=np.float64) * self._scale + self._offset


class LinearModel(LeakageModel):
    
    def __init__(self, coefficients: np.ndarray, intercept: float = 0.0):
        self._coefficients = np.asarray(coefficients, dtype=np.float64)
        self._intercept = intercept
    
    @property
    def coefficients(self) -> np.ndarray:
        return self._coefficients.copy()
    
    @property
    def intercept(self) -> float:
        return self._intercept
    
    @property
    def leakage_type(self) -> LeakageType:
        return LeakageType.LINEAR
    
    @property
    def name(self) -> str:
        return "Linear"
    
    def __call__(self, data: Union[int, np.ndarray],
                 prev_data: Optional[Union[int, np.ndarray]] = None) -> Union[float, np.ndarray]:
        data = np.asarray(data, dtype=np.float64)
        if data.ndim == 1:
            return np.dot(data, self._coefficients) + self._intercept
        return np.dot(data, self._coefficients) + self._intercept


class BitLeakageModel(LeakageModel):
    
    def __init__(self, bit_position: int, bit_width: int = 8):
        if bit_position < 0 or bit_position >= bit_width:
            raise ValueError(f"bit_position must be in [0, {bit_width})")
        self._bit_position = bit_position
        self._bit_width = bit_width
        self._mask = 1 << bit_position
    
    @property
    def bit_position(self) -> int:
        return self._bit_position
    
    @property
    def bit_width(self) -> int:
        return self._bit_width
    
    @property
    def leakage_type(self) -> LeakageType:
        return LeakageType.CUSTOM
    
    @property
    def name(self) -> str:
        return f"Bit{self._bit_position}"
    
    def __call__(self, data: Union[int, np.ndarray],
                 prev_data: Optional[Union[int, np.ndarray]] = None) -> Union[float, np.ndarray]:
        if isinstance(data, (int, np.integer)):
            return float((int(data) >> self._bit_position) & 1)
        data = np.asarray(data, dtype=np.uint64)
        return ((data >> self._bit_position) & 1).astype(np.float64)


class CustomLeakageModel(LeakageModel):
    
    def __init__(self, func: Callable, name: str = "Custom"):
        self._func = func
        self._name = name
    
    @property
    def leakage_type(self) -> LeakageType:
        return LeakageType.CUSTOM
    
    @property
    def name(self) -> str:
        return self._name
    
    def __call__(self, data: Union[int, np.ndarray],
                 prev_data: Optional[Union[int, np.ndarray]] = None) -> Union[float, np.ndarray]:
        return self._func(data, prev_data)


class CombinedLeakageModel(LeakageModel):
    
    def __init__(self, models: List[LeakageModel], weights: Optional[List[float]] = None):
        self._models = models
        if weights is None:
            weights = [1.0] * len(models)
        if len(weights) != len(models):
            raise ValueError("weights must have same length as models")
        self._weights = np.array(weights, dtype=np.float64)
    
    @property
    def models(self) -> List[LeakageModel]:
        return list(self._models)
    
    @property
    def weights(self) -> np.ndarray:
        return self._weights.copy()
    
    @property
    def leakage_type(self) -> LeakageType:
        return LeakageType.CUSTOM
    
    @property
    def name(self) -> str:
        return "Combined(" + "+".join(m.name for m in self._models) + ")"
    
    def __call__(self, data: Union[int, np.ndarray],
                 prev_data: Optional[Union[int, np.ndarray]] = None) -> Union[float, np.ndarray]:
        result = None
        for model, weight in zip(self._models, self._weights):
            contribution = model(data, prev_data)
            if isinstance(contribution, np.ndarray):
                contribution = contribution * weight
            else:
                contribution = float(contribution) * weight
            
            if result is None:
                result = contribution
            else:
                result = result + contribution
        return result


def create_leakage_model(model_type: Union[str, LeakageType], **kwargs) -> LeakageModel:
    if isinstance(model_type, str):
        model_type = LeakageType(model_type.lower())
    
    if model_type == LeakageType.HAMMING_WEIGHT:
        return HammingWeightModel(bit_width=kwargs.get('bit_width', 8))
    elif model_type == LeakageType.HAMMING_DISTANCE:
        return HammingDistanceModel(
            bit_width=kwargs.get('bit_width', 8),
            reference=kwargs.get('reference', None)
        )
    elif model_type == LeakageType.IDENTITY:
        return IdentityModel(
            scale=kwargs.get('scale', 1.0),
            offset=kwargs.get('offset', 0.0)
        )
    elif model_type == LeakageType.LINEAR:
        coefficients = kwargs.get('coefficients')
        if coefficients is None:
            raise ValueError("Linear model requires 'coefficients' parameter")
        return LinearModel(
            coefficients=coefficients,
            intercept=kwargs.get('intercept', 0.0)
        )
    else:
        raise ValueError(f"Unknown leakage model type: {model_type}")
