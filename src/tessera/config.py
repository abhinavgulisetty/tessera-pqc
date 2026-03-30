from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml


@dataclass
class AlgorithmConfig:
    name: str = "kyber1024"
    variant: str = "optimized"
    
    @classmethod
    def from_dict(cls, d: dict) -> AlgorithmConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class HardwareConfig:
    mcu: str = "generic"
    nvm_type: str = "fram"
    capacitor_uf: float = 100.0
    clock_mhz: float = 16.0
    
    @classmethod
    def from_dict(cls, d: dict) -> HardwareConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class PowerConfig:
    model: str = "exponential"
    on_time_avg: float = 100.0
    off_time_avg: float = 50.0
    
    @classmethod
    def from_dict(cls, d: dict) -> PowerConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class LeakageConfig:
    model: str = "hamming_weight"
    noise_stddev: float = 0.0
    record_trace: bool = True
    
    @classmethod
    def from_dict(cls, d: dict) -> LeakageConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class AttackConfig:
    type: str = "cpa"
    target_byte: int = 0
    num_traces: int = 1000
    
    @classmethod
    def from_dict(cls, d: dict) -> AttackConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class SimulationConfig:
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    power: PowerConfig = field(default_factory=PowerConfig)
    leakage: LeakageConfig = field(default_factory=LeakageConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    duration: int = 1000
    seed: int | None = None
    output_dir: str = "results"
    
    @classmethod
    def from_dict(cls, d: dict) -> SimulationConfig:
        return cls(
            algorithm=AlgorithmConfig.from_dict(d.get("algorithm", {})),
            hardware=HardwareConfig.from_dict(d.get("hardware", {})),
            power=PowerConfig.from_dict(d.get("power", {})),
            leakage=LeakageConfig.from_dict(d.get("leakage", {})),
            attack=AttackConfig.from_dict(d.get("attack", {})),
            duration=d.get("duration", 1000),
            seed=d.get("seed"),
            output_dir=d.get("output_dir", "results"),
        )
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": {
                "name": self.algorithm.name,
                "variant": self.algorithm.variant,
            },
            "hardware": {
                "mcu": self.hardware.mcu,
                "nvm_type": self.hardware.nvm_type,
                "capacitor_uf": self.hardware.capacitor_uf,
                "clock_mhz": self.hardware.clock_mhz,
            },
            "power": {
                "model": self.power.model,
                "on_time_avg": self.power.on_time_avg,
                "off_time_avg": self.power.off_time_avg,
            },
            "leakage": {
                "model": self.leakage.model,
                "noise_stddev": self.leakage.noise_stddev,
                "record_trace": self.leakage.record_trace,
            },
            "attack": {
                "type": self.attack.type,
                "target_byte": self.attack.target_byte,
                "num_traces": self.attack.num_traces,
            },
            "duration": self.duration,
            "seed": self.seed,
            "output_dir": self.output_dir,
        }


def load_config(path: Path | str) -> SimulationConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        data = yaml.safe_load(f)
    return SimulationConfig.from_dict(data or {})


def save_config(config: SimulationConfig, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)


def get_default_config() -> SimulationConfig:
    return SimulationConfig()
