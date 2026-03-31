from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Union, Any, IO
from pathlib import Path
import json
import io
import datetime

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None


def require_pandas():
    if not HAS_PANDAS:
        raise ImportError("pandas is required for DataFrame export. Install with: pip install pandas")


@dataclass
class ExportMetadata:
    name: str = ""
    description: str = ""
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    version: str = "1.0"
    author: str = ""
    custom: Dict = field(default_factory=dict)


class CSVExporter:
    
    def __init__(self, delimiter: str = ","):
        self.delimiter = delimiter
    
    def export_traces(
        self,
        traces: np.ndarray,
        path: Union[str, Path],
        include_header: bool = True,
        include_index: bool = True
    ):
        with open(path, 'w') as f:
            if include_header:
                n_samples = traces.shape[1]
                if include_index:
                    header = "trace_index" + self.delimiter
                else:
                    header = ""
                header += self.delimiter.join([f"s{i}" for i in range(n_samples)])
                f.write(header + "\n")
            
            for i, trace in enumerate(traces):
                if include_index:
                    line = f"{i}" + self.delimiter
                else:
                    line = ""
                line += self.delimiter.join([f"{v:.6f}" for v in trace])
                f.write(line + "\n")
    
    def export_correlations(
        self,
        correlations: np.ndarray,
        path: Union[str, Path],
        key_labels: Optional[List[str]] = None
    ):
        n_keys, n_samples = correlations.shape
        
        if key_labels is None:
            key_labels = [f"key_{i}" for i in range(n_keys)]
        
        with open(path, 'w') as f:
            header = "sample" + self.delimiter + self.delimiter.join(key_labels)
            f.write(header + "\n")
            
            for sample_idx in range(n_samples):
                values = correlations[:, sample_idx]
                line = f"{sample_idx}" + self.delimiter
                line += self.delimiter.join([f"{v:.6f}" for v in values])
                f.write(line + "\n")
    
    def export_metrics(
        self,
        metrics: Dict[str, Union[float, np.ndarray]],
        path: Union[str, Path]
    ):
        with open(path, 'w') as f:
            f.write("metric" + self.delimiter + "value\n")
            
            for name, value in metrics.items():
                if isinstance(value, np.ndarray):
                    for i, v in enumerate(value):
                        f.write(f"{name}_{i}" + self.delimiter + f"{v:.6f}\n")
                else:
                    f.write(f"{name}" + self.delimiter + f"{value:.6f}\n")
    
    def export_attack_results(
        self,
        results: List[Dict],
        path: Union[str, Path]
    ):
        if not results:
            return
        
        keys = list(results[0].keys())
        
        with open(path, 'w') as f:
            f.write(self.delimiter.join(keys) + "\n")
            
            for result in results:
                values = []
                for key in keys:
                    val = result.get(key, "")
                    if isinstance(val, float):
                        values.append(f"{val:.6f}")
                    elif isinstance(val, np.ndarray):
                        values.append(str(val.tolist()))
                    else:
                        values.append(str(val))
                f.write(self.delimiter.join(values) + "\n")


class DataFrameExporter:
    
    def __init__(self):
        require_pandas()
    
    def traces_to_dataframe(
        self,
        traces: np.ndarray,
        metadata: Optional[np.ndarray] = None,
        metadata_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        n_traces, n_samples = traces.shape
        
        columns = [f"s{i}" for i in range(n_samples)]
        df = pd.DataFrame(traces, columns=columns)
        
        if metadata is not None and metadata_columns is not None:
            for i, col_name in enumerate(metadata_columns):
                df[col_name] = metadata[:, i] if metadata.ndim > 1 else metadata
        
        return df
    
    def correlations_to_dataframe(
        self,
        correlations: np.ndarray,
        key_labels: Optional[List[str]] = None
    ) -> pd.DataFrame:
        n_keys, n_samples = correlations.shape
        
        if key_labels is None:
            key_labels = [f"key_{i}" for i in range(n_keys)]
        
        df = pd.DataFrame(correlations.T, columns=key_labels)
        df.index.name = "sample"
        
        return df
    
    def metrics_to_dataframe(
        self,
        metrics: Dict[str, Any]
    ) -> pd.DataFrame:
        rows = []
        
        for name, value in metrics.items():
            if isinstance(value, np.ndarray):
                rows.append({
                    "metric": name,
                    "value": np.max(value),
                    "type": "array",
                    "length": len(value)
                })
            elif isinstance(value, dict):
                for sub_key, sub_val in value.items():
                    rows.append({
                        "metric": f"{name}.{sub_key}",
                        "value": sub_val,
                        "type": "scalar",
                        "length": 1
                    })
            else:
                rows.append({
                    "metric": name,
                    "value": value,
                    "type": "scalar",
                    "length": 1
                })
        
        return pd.DataFrame(rows)
    
    def attack_results_to_dataframe(
        self,
        results: List[Dict]
    ) -> pd.DataFrame:
        return pd.DataFrame(results)


class LaTeXExporter:
    
    def __init__(
        self,
        float_format: str = ".4f",
        table_env: str = "table",
        position: str = "htbp"
    ):
        self.float_format = float_format
        self.table_env = table_env
        self.position = position
    
    def _format_value(self, value: Any) -> str:
        if isinstance(value, float):
            return f"{value:{self.float_format}}"
        elif isinstance(value, np.floating):
            return f"{float(value):{self.float_format}}"
        elif isinstance(value, bool):
            return "Yes" if value else "No"
        elif isinstance(value, np.ndarray):
            return f"array({len(value)})"
        return str(value)
    
    def export_metrics_table(
        self,
        metrics: Dict[str, Any],
        caption: str = "Side-Channel Analysis Metrics",
        label: str = "tab:metrics"
    ) -> str:
        lines = []
        
        lines.append(f"\\begin{{{self.table_env}}}[{self.position}]")
        lines.append("\\centering")
        lines.append("\\begin{tabular}{ll}")
        lines.append("\\toprule")
        lines.append("Metric & Value \\\\")
        lines.append("\\midrule")
        
        for name, value in metrics.items():
            if isinstance(value, dict):
                for sub_name, sub_value in value.items():
                    formatted = self._format_value(sub_value)
                    lines.append(f"{name}.{sub_name} & {formatted} \\\\")
            else:
                formatted = self._format_value(value)
                lines.append(f"{name} & {formatted} \\\\")
        
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append(f"\\caption{{{caption}}}")
        lines.append(f"\\label{{{label}}}")
        lines.append(f"\\end{{{self.table_env}}}")
        
        return "\n".join(lines)
    
    def export_attack_results_table(
        self,
        results: List[Dict],
        columns: Optional[List[str]] = None,
        caption: str = "Attack Results",
        label: str = "tab:attack_results"
    ) -> str:
        if not results:
            return ""
        
        if columns is None:
            columns = list(results[0].keys())
        
        lines = []
        
        lines.append(f"\\begin{{{self.table_env}}}[{self.position}]")
        lines.append("\\centering")
        
        col_spec = "l" * len(columns)
        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        lines.append("\\toprule")
        
        header = " & ".join(columns) + " \\\\"
        lines.append(header)
        lines.append("\\midrule")
        
        for result in results:
            row_values = [self._format_value(result.get(col, "")) for col in columns]
            lines.append(" & ".join(row_values) + " \\\\")
        
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append(f"\\caption{{{caption}}}")
        lines.append(f"\\label{{{label}}}")
        lines.append(f"\\end{{{self.table_env}}}")
        
        return "\n".join(lines)
    
    def export_comparison_table(
        self,
        data: Dict[str, Dict[str, Any]],
        caption: str = "Comparison",
        label: str = "tab:comparison"
    ) -> str:
        if not data:
            return ""
        
        methods = list(data.keys())
        metrics = list(data[methods[0]].keys())
        
        lines = []
        
        lines.append(f"\\begin{{{self.table_env}}}[{self.position}]")
        lines.append("\\centering")
        
        col_spec = "l" + "c" * len(methods)
        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        lines.append("\\toprule")
        
        header = "Metric & " + " & ".join(methods) + " \\\\"
        lines.append(header)
        lines.append("\\midrule")
        
        for metric in metrics:
            values = [self._format_value(data[method].get(metric, "")) for method in methods]
            lines.append(f"{metric} & " + " & ".join(values) + " \\\\")
        
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append(f"\\caption{{{caption}}}")
        lines.append(f"\\label{{{label}}}")
        lines.append(f"\\end{{{self.table_env}}}")
        
        return "\n".join(lines)


class JSONExporter:
    
    def __init__(self, indent: int = 2):
        self.indent = indent
    
    def _convert_numpy(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy(v) for v in obj]
        return obj
    
    def export(
        self,
        data: Dict,
        path: Union[str, Path],
        metadata: Optional[ExportMetadata] = None
    ):
        export_data = {
            "data": self._convert_numpy(data)
        }
        
        if metadata is not None:
            export_data["metadata"] = asdict(metadata)
        
        with open(path, 'w') as f:
            json.dump(export_data, f, indent=self.indent)
    
    def export_traces(
        self,
        traces: np.ndarray,
        path: Union[str, Path],
        metadata: Optional[ExportMetadata] = None,
        additional_data: Optional[Dict] = None
    ):
        data = {
            "traces": traces.tolist(),
            "shape": list(traces.shape),
            "dtype": str(traces.dtype)
        }
        
        if additional_data:
            data.update(self._convert_numpy(additional_data))
        
        self.export(data, path, metadata)
    
    def export_attack_results(
        self,
        results: Dict,
        path: Union[str, Path],
        metadata: Optional[ExportMetadata] = None
    ):
        self.export(results, path, metadata)
    
    def load(self, path: Union[str, Path]) -> Tuple[Dict, Optional[ExportMetadata]]:
        with open(path, 'r') as f:
            export_data = json.load(f)
        
        data = export_data.get("data", {})
        metadata_dict = export_data.get("metadata", None)
        
        metadata = None
        if metadata_dict:
            metadata = ExportMetadata(**metadata_dict)
        
        return data, metadata


class NPZExporter:
    
    def export_traces(
        self,
        path: Union[str, Path],
        traces: np.ndarray,
        labels: Optional[np.ndarray] = None,
        plaintexts: Optional[np.ndarray] = None,
        ciphertexts: Optional[np.ndarray] = None,
        keys: Optional[np.ndarray] = None,
        **kwargs
    ):
        data = {"traces": traces}
        
        if labels is not None:
            data["labels"] = labels
        if plaintexts is not None:
            data["plaintexts"] = plaintexts
        if ciphertexts is not None:
            data["ciphertexts"] = ciphertexts
        if keys is not None:
            data["keys"] = keys
        
        data.update(kwargs)
        
        np.savez_compressed(path, **data)
    
    def load_traces(
        self,
        path: Union[str, Path]
    ) -> Dict[str, np.ndarray]:
        with np.load(path) as data:
            return {key: data[key] for key in data.files}
    
    def export_attack_state(
        self,
        path: Union[str, Path],
        correlations: np.ndarray,
        hypotheses: np.ndarray,
        num_traces_processed: int,
        **kwargs
    ):
        data = {
            "correlations": correlations,
            "hypotheses": hypotheses,
            "num_traces_processed": np.array([num_traces_processed])
        }
        data.update(kwargs)
        
        np.savez_compressed(path, **data)


class ReportGenerator:
    
    def __init__(self):
        self.csv_exporter = CSVExporter()
        self.latex_exporter = LaTeXExporter()
        self.json_exporter = JSONExporter()
    
    def generate_text_report(
        self,
        metrics: Dict,
        attack_results: Optional[Dict] = None,
        title: str = "Side-Channel Analysis Report"
    ) -> str:
        lines = []
        
        lines.append("=" * 70)
        lines.append(title.center(70))
        lines.append("=" * 70)
        lines.append(f"\nGenerated: {datetime.datetime.now().isoformat()}\n")
        
        lines.append("-" * 70)
        lines.append("METRICS SUMMARY")
        lines.append("-" * 70)
        
        for name, value in metrics.items():
            if isinstance(value, dict):
                lines.append(f"\n{name}:")
                for sub_name, sub_value in value.items():
                    if isinstance(sub_value, float):
                        lines.append(f"  {sub_name}: {sub_value:.6f}")
                    else:
                        lines.append(f"  {sub_name}: {sub_value}")
            elif isinstance(value, np.ndarray):
                lines.append(f"{name}: array of shape {value.shape}")
                lines.append(f"  max: {np.max(value):.6f}")
                lines.append(f"  min: {np.min(value):.6f}")
                lines.append(f"  mean: {np.mean(value):.6f}")
            elif isinstance(value, float):
                lines.append(f"{name}: {value:.6f}")
            else:
                lines.append(f"{name}: {value}")
        
        if attack_results:
            lines.append("\n" + "-" * 70)
            lines.append("ATTACK RESULTS")
            lines.append("-" * 70)
            
            for name, value in attack_results.items():
                if isinstance(value, float):
                    lines.append(f"{name}: {value:.6f}")
                elif isinstance(value, np.ndarray):
                    lines.append(f"{name}: {value}")
                else:
                    lines.append(f"{name}: {value}")
        
        lines.append("\n" + "=" * 70)
        
        return "\n".join(lines)
    
    def export_full_report(
        self,
        output_dir: Union[str, Path],
        traces: Optional[np.ndarray] = None,
        metrics: Optional[Dict] = None,
        attack_results: Optional[Dict] = None,
        correlations: Optional[np.ndarray] = None,
        name: str = "analysis"
    ):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if traces is not None:
            self.csv_exporter.export_traces(
                traces[:100] if len(traces) > 100 else traces,
                output_dir / f"{name}_traces.csv"
            )
        
        if metrics is not None:
            scalar_metrics = {}
            for k, v in metrics.items():
                if isinstance(v, dict):
                    scalar_metrics.update({f"{k}.{sk}": sv for sk, sv in v.items() 
                                          if not isinstance(sv, np.ndarray)})
                elif not isinstance(v, np.ndarray):
                    scalar_metrics[k] = v
            
            self.csv_exporter.export_metrics(scalar_metrics, output_dir / f"{name}_metrics.csv")
            
            latex_table = self.latex_exporter.export_metrics_table(
                scalar_metrics,
                caption=f"{name.title()} Metrics",
                label=f"tab:{name}_metrics"
            )
            with open(output_dir / f"{name}_metrics.tex", 'w') as f:
                f.write(latex_table)
        
        if correlations is not None:
            self.csv_exporter.export_correlations(
                correlations,
                output_dir / f"{name}_correlations.csv"
            )
        
        if attack_results is not None:
            self.json_exporter.export_attack_results(
                attack_results,
                output_dir / f"{name}_results.json",
                ExportMetadata(name=name, description="Attack results")
            )
        
        if metrics is not None or attack_results is not None:
            text_report = self.generate_text_report(
                metrics or {},
                attack_results,
                title=f"{name.title()} Analysis Report"
            )
            with open(output_dir / f"{name}_report.txt", 'w') as f:
                f.write(text_report)
