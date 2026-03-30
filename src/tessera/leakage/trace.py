from typing import Optional, List, Dict, Any, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import json
import struct
from datetime import datetime

from .models import LeakageModel, HammingWeightModel


@dataclass
class TraceMetadata:
    algorithm: str = "unknown"
    operation: str = "unknown"
    parameters: Dict[str, Any] = field(default_factory=dict)
    sample_rate: float = 1e6
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "operation": self.operation,
            "parameters": self.parameters,
            "sample_rate": self.sample_rate,
            "timestamp": self.timestamp,
            "description": self.description,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TraceMetadata":
        return cls(
            algorithm=d.get("algorithm", "unknown"),
            operation=d.get("operation", "unknown"),
            parameters=d.get("parameters", {}),
            sample_rate=d.get("sample_rate", 1e6),
            timestamp=d.get("timestamp", ""),
            description=d.get("description", ""),
        )


class Trace:
    
    def __init__(self, samples: np.ndarray, 
                 plaintext: Optional[np.ndarray] = None,
                 ciphertext: Optional[np.ndarray] = None,
                 key: Optional[np.ndarray] = None,
                 intermediate: Optional[Dict[str, np.ndarray]] = None,
                 metadata: Optional[TraceMetadata] = None):
        self._samples = np.asarray(samples, dtype=np.float64)
        self._plaintext = np.asarray(plaintext) if plaintext is not None else None
        self._ciphertext = np.asarray(ciphertext) if ciphertext is not None else None
        self._key = np.asarray(key) if key is not None else None
        self._intermediate = intermediate or {}
        self._metadata = metadata or TraceMetadata()
    
    @property
    def samples(self) -> np.ndarray:
        return self._samples
    
    @samples.setter
    def samples(self, value: np.ndarray):
        self._samples = np.asarray(value, dtype=np.float64)
    
    @property
    def plaintext(self) -> Optional[np.ndarray]:
        return self._plaintext
    
    @plaintext.setter
    def plaintext(self, value: Optional[np.ndarray]):
        self._plaintext = np.asarray(value) if value is not None else None
    
    @property
    def ciphertext(self) -> Optional[np.ndarray]:
        return self._ciphertext
    
    @ciphertext.setter
    def ciphertext(self, value: Optional[np.ndarray]):
        self._ciphertext = np.asarray(value) if value is not None else None
    
    @property
    def key(self) -> Optional[np.ndarray]:
        return self._key
    
    @key.setter
    def key(self, value: Optional[np.ndarray]):
        self._key = np.asarray(value) if value is not None else None
    
    @property
    def intermediate(self) -> Dict[str, np.ndarray]:
        return self._intermediate
    
    @property
    def metadata(self) -> TraceMetadata:
        return self._metadata
    
    @property
    def num_samples(self) -> int:
        return len(self._samples)
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: Union[int, slice]) -> np.ndarray:
        return self._samples[idx]
    
    def copy(self) -> "Trace":
        return Trace(
            samples=self._samples.copy(),
            plaintext=self._plaintext.copy() if self._plaintext is not None else None,
            ciphertext=self._ciphertext.copy() if self._ciphertext is not None else None,
            key=self._key.copy() if self._key is not None else None,
            intermediate={k: v.copy() for k, v in self._intermediate.items()},
            metadata=TraceMetadata(**self._metadata.to_dict()),
        )


class TraceSet:
    
    def __init__(self, traces: Optional[List[Trace]] = None,
                 metadata: Optional[TraceMetadata] = None):
        self._traces: List[Trace] = traces or []
        self._metadata = metadata or TraceMetadata()
    
    @property
    def traces(self) -> List[Trace]:
        return self._traces
    
    @property
    def metadata(self) -> TraceMetadata:
        return self._metadata
    
    @property
    def num_traces(self) -> int:
        return len(self._traces)
    
    @property
    def num_samples(self) -> int:
        if not self._traces:
            return 0
        return self._traces[0].num_samples
    
    def add_trace(self, trace: Trace):
        self._traces.append(trace)
    
    def __len__(self) -> int:
        return self.num_traces
    
    def __getitem__(self, idx: Union[int, slice]) -> Union[Trace, List[Trace]]:
        return self._traces[idx]
    
    def __iter__(self):
        return iter(self._traces)
    
    def get_samples_matrix(self) -> np.ndarray:
        if not self._traces:
            return np.array([], dtype=np.float64)
        return np.vstack([t.samples for t in self._traces])
    
    def get_plaintexts(self) -> Optional[np.ndarray]:
        if not self._traces or self._traces[0].plaintext is None:
            return None
        return np.vstack([t.plaintext for t in self._traces])
    
    def get_ciphertexts(self) -> Optional[np.ndarray]:
        if not self._traces or self._traces[0].ciphertext is None:
            return None
        return np.vstack([t.ciphertext for t in self._traces])
    
    def get_keys(self) -> Optional[np.ndarray]:
        if not self._traces or self._traces[0].key is None:
            return None
        return np.vstack([t.key for t in self._traces])
    
    def get_intermediate(self, name: str) -> Optional[np.ndarray]:
        if not self._traces or name not in self._traces[0].intermediate:
            return None
        return np.vstack([t.intermediate[name] for t in self._traces])
    
    def subset(self, indices: Union[List[int], np.ndarray]) -> "TraceSet":
        traces = [self._traces[i] for i in indices]
        return TraceSet(traces=traces, metadata=self._metadata)
    
    def sample_subset(self, start: int, end: int) -> "TraceSet":
        traces = []
        for t in self._traces:
            new_trace = Trace(
                samples=t.samples[start:end],
                plaintext=t.plaintext,
                ciphertext=t.ciphertext,
                key=t.key,
                intermediate=t.intermediate,
                metadata=t.metadata,
            )
            traces.append(new_trace)
        return TraceSet(traces=traces, metadata=self._metadata)
    
    def mean_trace(self) -> np.ndarray:
        return np.mean(self.get_samples_matrix(), axis=0)
    
    def std_trace(self) -> np.ndarray:
        return np.std(self.get_samples_matrix(), axis=0)


class TraceGenerator:
    
    def __init__(self, leakage_model: Optional[LeakageModel] = None,
                 num_samples_per_op: int = 1,
                 noise_std: float = 0.0):
        self._leakage_model = leakage_model or HammingWeightModel(8)
        self._num_samples_per_op = num_samples_per_op
        self._noise_std = noise_std
        self._operations: List[Dict[str, Any]] = []
    
    @property
    def leakage_model(self) -> LeakageModel:
        return self._leakage_model
    
    @leakage_model.setter
    def leakage_model(self, model: LeakageModel):
        self._leakage_model = model
    
    @property
    def num_samples_per_op(self) -> int:
        return self._num_samples_per_op
    
    @num_samples_per_op.setter
    def num_samples_per_op(self, value: int):
        self._num_samples_per_op = value
    
    @property
    def noise_std(self) -> float:
        return self._noise_std
    
    @noise_std.setter
    def noise_std(self, value: float):
        self._noise_std = value
    
    def reset(self):
        self._operations.clear()
    
    def record_operation(self, name: str, data: Union[int, np.ndarray],
                        prev_data: Optional[Union[int, np.ndarray]] = None):
        leakage = self._leakage_model(data, prev_data)
        self._operations.append({
            "name": name,
            "data": data,
            "leakage": leakage,
        })
    
    def generate_trace(self, plaintext: Optional[np.ndarray] = None,
                       ciphertext: Optional[np.ndarray] = None,
                       key: Optional[np.ndarray] = None,
                       metadata: Optional[TraceMetadata] = None) -> Trace:
        samples = []
        intermediate = {}
        
        for op in self._operations:
            leakage = op["leakage"]
            if isinstance(leakage, np.ndarray):
                for val in leakage:
                    sample_chunk = np.full(self._num_samples_per_op, val, dtype=np.float64)
                    if self._noise_std > 0:
                        sample_chunk += np.random.normal(0, self._noise_std, self._num_samples_per_op)
                    samples.append(sample_chunk)
            else:
                sample_chunk = np.full(self._num_samples_per_op, leakage, dtype=np.float64)
                if self._noise_std > 0:
                    sample_chunk += np.random.normal(0, self._noise_std, self._num_samples_per_op)
                samples.append(sample_chunk)
            
            if isinstance(op["data"], np.ndarray):
                intermediate[op["name"]] = op["data"].copy()
        
        if samples:
            trace_samples = np.concatenate(samples)
        else:
            trace_samples = np.array([], dtype=np.float64)
        
        return Trace(
            samples=trace_samples,
            plaintext=plaintext,
            ciphertext=ciphertext,
            key=key,
            intermediate=intermediate,
            metadata=metadata,
        )


class TraceStorage:
    
    @staticmethod
    def save_numpy(trace_set: TraceSet, path: Union[str, Path]):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "samples": trace_set.get_samples_matrix(),
            "metadata": json.dumps(trace_set.metadata.to_dict()),
        }
        
        plaintexts = trace_set.get_plaintexts()
        if plaintexts is not None:
            data["plaintexts"] = plaintexts
        
        ciphertexts = trace_set.get_ciphertexts()
        if ciphertexts is not None:
            data["ciphertexts"] = ciphertexts
        
        keys = trace_set.get_keys()
        if keys is not None:
            data["keys"] = keys
        
        np.savez_compressed(path, **data)
    
    @staticmethod
    def load_numpy(path: Union[str, Path]) -> TraceSet:
        path = Path(path)
        data = np.load(path, allow_pickle=True)
        
        samples = data["samples"]
        metadata_dict = json.loads(str(data["metadata"]))
        metadata = TraceMetadata.from_dict(metadata_dict)
        
        plaintexts = data.get("plaintexts", None)
        ciphertexts = data.get("ciphertexts", None)
        keys = data.get("keys", None)
        
        traces = []
        for i in range(len(samples)):
            trace = Trace(
                samples=samples[i],
                plaintext=plaintexts[i] if plaintexts is not None else None,
                ciphertext=ciphertexts[i] if ciphertexts is not None else None,
                key=keys[i] if keys is not None else None,
                metadata=metadata,
            )
            traces.append(trace)
        
        return TraceSet(traces=traces, metadata=metadata)
    
    @staticmethod
    def save_trs(trace_set: TraceSet, path: Union[str, Path]):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        samples = trace_set.get_samples_matrix()
        num_traces, num_samples = samples.shape
        
        with open(path, "wb") as f:
            f.write(struct.pack("<I", 0x41))
            f.write(struct.pack("<I", num_traces))
            f.write(struct.pack("<I", 0x42))
            f.write(struct.pack("<I", num_samples))
            f.write(struct.pack("<I", 0x43))
            f.write(struct.pack("<I", 4))
            f.write(struct.pack("<I", 0x5F))
            f.write(struct.pack("<I", 0))
            
            for i in range(num_traces):
                f.write(samples[i].astype(np.float32).tobytes())
    
    @staticmethod
    def save_csv(trace_set: TraceSet, path: Union[str, Path],
                 include_plaintext: bool = True,
                 include_key: bool = False):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        samples = trace_set.get_samples_matrix()
        plaintexts = trace_set.get_plaintexts() if include_plaintext else None
        keys = trace_set.get_keys() if include_key else None
        
        with open(path, "w") as f:
            header = []
            if plaintexts is not None:
                header.extend([f"pt_{i}" for i in range(plaintexts.shape[1])])
            if keys is not None:
                header.extend([f"key_{i}" for i in range(keys.shape[1])])
            header.extend([f"s_{i}" for i in range(samples.shape[1])])
            f.write(",".join(header) + "\n")
            
            for i in range(len(samples)):
                row = []
                if plaintexts is not None:
                    row.extend([str(int(x)) for x in plaintexts[i]])
                if keys is not None:
                    row.extend([str(int(x)) for x in keys[i]])
                row.extend([f"{x:.6f}" for x in samples[i]])
                f.write(",".join(row) + "\n")


def generate_random_traces(num_traces: int,
                          num_samples: int,
                          leakage_model: Optional[LeakageModel] = None,
                          noise_std: float = 1.0,
                          plaintext_bytes: int = 16,
                          key: Optional[np.ndarray] = None,
                          poi_indices: Optional[List[int]] = None,
                          metadata: Optional[TraceMetadata] = None) -> TraceSet:
    leakage_model = leakage_model or HammingWeightModel(8)
    
    if key is None:
        key = np.random.randint(0, 256, plaintext_bytes, dtype=np.uint8)
    
    if poi_indices is None:
        poi_indices = list(range(0, min(plaintext_bytes, num_samples)))
    
    traces = []
    for _ in range(num_traces):
        plaintext = np.random.randint(0, 256, plaintext_bytes, dtype=np.uint8)
        
        samples = np.random.normal(0, noise_std, num_samples)
        
        for i, poi in enumerate(poi_indices):
            if poi < num_samples and i < len(plaintext):
                intermediate = plaintext[i] ^ key[i]
                leakage = leakage_model(intermediate)
                samples[poi] += leakage
        
        trace = Trace(
            samples=samples,
            plaintext=plaintext,
            key=key.copy(),
            metadata=metadata,
        )
        traces.append(trace)
    
    return TraceSet(traces=traces, metadata=metadata or TraceMetadata())
