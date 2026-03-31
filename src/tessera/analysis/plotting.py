from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import warnings

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    Figure = None
    Axes = None


def require_matplotlib():
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")


@dataclass
class PlotStyle:
    figsize: Tuple[float, float] = (10, 6)
    dpi: int = 100
    line_width: float = 1.0
    marker_size: float = 6.0
    font_size: int = 12
    title_size: int = 14
    grid: bool = True
    grid_alpha: float = 0.3
    colormap: str = "viridis"
    
    def apply(self, fig: Figure, ax: Axes):
        if not HAS_MATPLOTLIB:
            return
        ax.tick_params(labelsize=self.font_size)
        if self.grid:
            ax.grid(True, alpha=self.grid_alpha)


class TracePlotter:
    
    def __init__(self, style: PlotStyle = None):
        require_matplotlib()
        self.style = style or PlotStyle()
    
    def plot_single_trace(
        self,
        trace: np.ndarray,
        title: str = "Power Trace",
        xlabel: str = "Sample",
        ylabel: str = "Power",
        ax: Axes = None
    ) -> Tuple[Figure, Axes]:
        if ax is None:
            fig, ax = plt.subplots(figsize=self.style.figsize, dpi=self.style.dpi)
        else:
            fig = ax.figure
        
        ax.plot(trace, linewidth=self.style.line_width)
        ax.set_xlabel(xlabel, fontsize=self.style.font_size)
        ax.set_ylabel(ylabel, fontsize=self.style.font_size)
        ax.set_title(title, fontsize=self.style.title_size)
        self.style.apply(fig, ax)
        
        return fig, ax
    
    def plot_multiple_traces(
        self,
        traces: np.ndarray,
        num_traces: int = 10,
        title: str = "Power Traces",
        alpha: float = 0.5,
        ax: Axes = None
    ) -> Tuple[Figure, Axes]:
        if ax is None:
            fig, ax = plt.subplots(figsize=self.style.figsize, dpi=self.style.dpi)
        else:
            fig = ax.figure
        
        n = min(num_traces, len(traces))
        for i in range(n):
            ax.plot(traces[i], alpha=alpha, linewidth=self.style.line_width)
        
        ax.set_xlabel("Sample", fontsize=self.style.font_size)
        ax.set_ylabel("Power", fontsize=self.style.font_size)
        ax.set_title(title, fontsize=self.style.title_size)
        self.style.apply(fig, ax)
        
        return fig, ax
    
    def plot_mean_trace_with_std(
        self,
        traces: np.ndarray,
        title: str = "Mean Trace with Standard Deviation",
        ax: Axes = None
    ) -> Tuple[Figure, Axes]:
        if ax is None:
            fig, ax = plt.subplots(figsize=self.style.figsize, dpi=self.style.dpi)
        else:
            fig = ax.figure
        
        mean = np.mean(traces, axis=0)
        std = np.std(traces, axis=0)
        x = np.arange(len(mean))
        
        ax.plot(x, mean, linewidth=self.style.line_width, label="Mean")
        ax.fill_between(x, mean - std, mean + std, alpha=0.3, label="±1 std")
        
        ax.set_xlabel("Sample", fontsize=self.style.font_size)
        ax.set_ylabel("Power", fontsize=self.style.font_size)
        ax.set_title(title, fontsize=self.style.title_size)
        ax.legend()
        self.style.apply(fig, ax)
        
        return fig, ax
    
    def plot_trace_comparison(
        self,
        trace1: np.ndarray,
        trace2: np.ndarray,
        label1: str = "Trace 1",
        label2: str = "Trace 2",
        title: str = "Trace Comparison",
        ax: Axes = None
    ) -> Tuple[Figure, Axes]:
        if ax is None:
            fig, ax = plt.subplots(figsize=self.style.figsize, dpi=self.style.dpi)
        else:
            fig = ax.figure
        
        ax.plot(trace1, linewidth=self.style.line_width, label=label1)
        ax.plot(trace2, linewidth=self.style.line_width, label=label2)
        
        ax.set_xlabel("Sample", fontsize=self.style.font_size)
        ax.set_ylabel("Power", fontsize=self.style.font_size)
        ax.set_title(title, fontsize=self.style.title_size)
        ax.legend()
        self.style.apply(fig, ax)
        
        return fig, ax
    
    def plot_trace_heatmap(
        self,
        traces: np.ndarray,
        title: str = "Trace Heatmap",
        ax: Axes = None
    ) -> Tuple[Figure, Axes]:
        if ax is None:
            fig, ax = plt.subplots(figsize=self.style.figsize, dpi=self.style.dpi)
        else:
            fig = ax.figure
        
        im = ax.imshow(traces, aspect='auto', cmap=self.style.colormap)
        fig.colorbar(im, ax=ax, label="Power")
        
        ax.set_xlabel("Sample", fontsize=self.style.font_size)
        ax.set_ylabel("Trace Index", fontsize=self.style.font_size)
        ax.set_title(title, fontsize=self.style.title_size)
        
        return fig, ax


class AttackPlotter:
    
    def __init__(self, style: PlotStyle = None):
        require_matplotlib()
        self.style = style or PlotStyle()
    
    def plot_correlation_traces(
        self,
        correlations: np.ndarray,
        correct_key: Optional[int] = None,
        title: str = "Correlation Traces",
        ax: Axes = None
    ) -> Tuple[Figure, Axes]:
        if ax is None:
            fig, ax = plt.subplots(figsize=self.style.figsize, dpi=self.style.dpi)
        else:
            fig = ax.figure
        
        n_keys = correlations.shape[0]
        
        for i in range(n_keys):
            if correct_key is not None and i == correct_key:
                ax.plot(correlations[i], 'r-', linewidth=self.style.line_width * 2, 
                       label=f"Key {i} (correct)")
            else:
                ax.plot(correlations[i], 'gray', alpha=0.3, linewidth=self.style.line_width)
        
        ax.set_xlabel("Sample", fontsize=self.style.font_size)
        ax.set_ylabel("Correlation", fontsize=self.style.font_size)
        ax.set_title(title, fontsize=self.style.title_size)
        if correct_key is not None:
            ax.legend()
        self.style.apply(fig, ax)
        
        return fig, ax
    
    def plot_correlation_heatmap(
        self,
        correlations: np.ndarray,
        title: str = "Correlation Matrix",
        ax: Axes = None
    ) -> Tuple[Figure, Axes]:
        if ax is None:
            fig, ax = plt.subplots(figsize=self.style.figsize, dpi=self.style.dpi)
        else:
            fig = ax.figure
        
        im = ax.imshow(correlations, aspect='auto', cmap='RdBu_r', 
                       vmin=-np.max(np.abs(correlations)), 
                       vmax=np.max(np.abs(correlations)))
        fig.colorbar(im, ax=ax, label="Correlation")
        
        ax.set_xlabel("Sample", fontsize=self.style.font_size)
        ax.set_ylabel("Key Hypothesis", fontsize=self.style.font_size)
        ax.set_title(title, fontsize=self.style.title_size)
        
        return fig, ax
    
    def plot_key_ranking(
        self,
        correlations: np.ndarray,
        correct_key: Optional[int] = None,
        title: str = "Key Ranking",
        ax: Axes = None
    ) -> Tuple[Figure, Axes]:
        if ax is None:
            fig, ax = plt.subplots(figsize=self.style.figsize, dpi=self.style.dpi)
        else:
            fig = ax.figure
        
        max_corr = np.max(np.abs(correlations), axis=1)
        sorted_indices = np.argsort(max_corr)[::-1]
        
        colors = ['red' if correct_key is not None and i == correct_key else 'steelblue' 
                  for i in sorted_indices]
        
        ax.bar(range(len(max_corr)), max_corr[sorted_indices], color=colors)
        
        ax.set_xlabel("Rank", fontsize=self.style.font_size)
        ax.set_ylabel("Max Correlation", fontsize=self.style.font_size)
        ax.set_title(title, fontsize=self.style.title_size)
        self.style.apply(fig, ax)
        
        return fig, ax
    
    def plot_success_rate_vs_traces(
        self,
        trace_counts: np.ndarray,
        success_rates: np.ndarray,
        title: str = "Success Rate vs Number of Traces",
        ax: Axes = None
    ) -> Tuple[Figure, Axes]:
        if ax is None:
            fig, ax = plt.subplots(figsize=self.style.figsize, dpi=self.style.dpi)
        else:
            fig = ax.figure
        
        ax.plot(trace_counts, success_rates, 'o-', 
               linewidth=self.style.line_width,
               markersize=self.style.marker_size)
        
        ax.set_xlabel("Number of Traces", fontsize=self.style.font_size)
        ax.set_ylabel("Success Rate", fontsize=self.style.font_size)
        ax.set_title(title, fontsize=self.style.title_size)
        ax.set_ylim(0, 1.05)
        self.style.apply(fig, ax)
        
        return fig, ax
    
    def plot_rank_evolution(
        self,
        trace_counts: np.ndarray,
        ranks: np.ndarray,
        title: str = "Key Rank Evolution",
        ax: Axes = None
    ) -> Tuple[Figure, Axes]:
        if ax is None:
            fig, ax = plt.subplots(figsize=self.style.figsize, dpi=self.style.dpi)
        else:
            fig = ax.figure
        
        ax.semilogy(trace_counts, ranks + 1, 'o-',
                   linewidth=self.style.line_width,
                   markersize=self.style.marker_size)
        
        ax.axhline(y=1, color='r', linestyle='--', label='Rank 1')
        
        ax.set_xlabel("Number of Traces", fontsize=self.style.font_size)
        ax.set_ylabel("Key Rank", fontsize=self.style.font_size)
        ax.set_title(title, fontsize=self.style.title_size)
        ax.legend()
        self.style.apply(fig, ax)
        
        return fig, ax
    
    def plot_differential_trace(
        self,
        diff_trace: np.ndarray,
        title: str = "Differential Trace",
        ax: Axes = None
    ) -> Tuple[Figure, Axes]:
        if ax is None:
            fig, ax = plt.subplots(figsize=self.style.figsize, dpi=self.style.dpi)
        else:
            fig = ax.figure
        
        ax.plot(diff_trace, linewidth=self.style.line_width)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        ax.set_xlabel("Sample", fontsize=self.style.font_size)
        ax.set_ylabel("Differential", fontsize=self.style.font_size)
        ax.set_title(title, fontsize=self.style.title_size)
        self.style.apply(fig, ax)
        
        return fig, ax


class TVLAPlotter:
    
    def __init__(self, style: PlotStyle = None):
        require_matplotlib()
        self.style = style or PlotStyle()
    
    def plot_t_statistic(
        self,
        t_statistic: np.ndarray,
        threshold: float = 4.5,
        title: str = "TVLA t-statistic",
        ax: Axes = None
    ) -> Tuple[Figure, Axes]:
        if ax is None:
            fig, ax = plt.subplots(figsize=self.style.figsize, dpi=self.style.dpi)
        else:
            fig = ax.figure
        
        ax.plot(t_statistic, linewidth=self.style.line_width)
        ax.axhline(y=threshold, color='r', linestyle='--', label=f'+{threshold}')
        ax.axhline(y=-threshold, color='r', linestyle='--', label=f'-{threshold}')
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        
        leaking = np.abs(t_statistic) > threshold
        if np.any(leaking):
            leaking_indices = np.where(leaking)[0]
            ax.scatter(leaking_indices, t_statistic[leaking_indices], 
                      c='red', s=self.style.marker_size * 2, zorder=5,
                      label='Leaking points')
        
        ax.set_xlabel("Sample", fontsize=self.style.font_size)
        ax.set_ylabel("t-statistic", fontsize=self.style.font_size)
        ax.set_title(title, fontsize=self.style.title_size)
        ax.legend()
        self.style.apply(fig, ax)
        
        return fig, ax
    
    def plot_t_statistic_histogram(
        self,
        t_statistic: np.ndarray,
        bins: int = 50,
        title: str = "t-statistic Distribution",
        ax: Axes = None
    ) -> Tuple[Figure, Axes]:
        if ax is None:
            fig, ax = plt.subplots(figsize=self.style.figsize, dpi=self.style.dpi)
        else:
            fig = ax.figure
        
        ax.hist(t_statistic, bins=bins, density=True, alpha=0.7)
        
        x = np.linspace(-6, 6, 100)
        from scipy import stats
        ax.plot(x, stats.norm.pdf(x), 'r-', linewidth=2, label='Normal distribution')
        
        ax.axvline(x=4.5, color='g', linestyle='--', label='±4.5 threshold')
        ax.axvline(x=-4.5, color='g', linestyle='--')
        
        ax.set_xlabel("t-statistic", fontsize=self.style.font_size)
        ax.set_ylabel("Density", fontsize=self.style.font_size)
        ax.set_title(title, fontsize=self.style.title_size)
        ax.legend()
        self.style.apply(fig, ax)
        
        return fig, ax
    
    def plot_leakage_over_time(
        self,
        t_statistics: List[np.ndarray],
        trace_counts: List[int],
        sample_index: int,
        title: str = "Leakage Evolution",
        ax: Axes = None
    ) -> Tuple[Figure, Axes]:
        if ax is None:
            fig, ax = plt.subplots(figsize=self.style.figsize, dpi=self.style.dpi)
        else:
            fig = ax.figure
        
        t_values = [t[sample_index] for t in t_statistics]
        
        ax.plot(trace_counts, t_values, 'o-',
               linewidth=self.style.line_width,
               markersize=self.style.marker_size)
        
        ax.axhline(y=4.5, color='r', linestyle='--', alpha=0.5)
        ax.axhline(y=-4.5, color='r', linestyle='--', alpha=0.5)
        
        ax.set_xlabel("Number of Traces", fontsize=self.style.font_size)
        ax.set_ylabel("t-statistic", fontsize=self.style.font_size)
        ax.set_title(f"{title} (Sample {sample_index})", fontsize=self.style.title_size)
        self.style.apply(fig, ax)
        
        return fig, ax


class SNRPlotter:
    
    def __init__(self, style: PlotStyle = None):
        require_matplotlib()
        self.style = style or PlotStyle()
    
    def plot_snr(
        self,
        snr: np.ndarray,
        title: str = "Signal-to-Noise Ratio",
        ax: Axes = None
    ) -> Tuple[Figure, Axes]:
        if ax is None:
            fig, ax = plt.subplots(figsize=self.style.figsize, dpi=self.style.dpi)
        else:
            fig = ax.figure
        
        ax.plot(snr, linewidth=self.style.line_width)
        
        max_idx = np.argmax(snr)
        ax.scatter([max_idx], [snr[max_idx]], c='red', s=50, zorder=5,
                  label=f'Max SNR: {snr[max_idx]:.4f} at {max_idx}')
        
        ax.set_xlabel("Sample", fontsize=self.style.font_size)
        ax.set_ylabel("SNR", fontsize=self.style.font_size)
        ax.set_title(title, fontsize=self.style.title_size)
        ax.legend()
        self.style.apply(fig, ax)
        
        return fig, ax
    
    def plot_snr_by_byte(
        self,
        snr_by_byte: Dict[int, np.ndarray],
        title: str = "SNR by Key Byte",
        ax: Axes = None
    ) -> Tuple[Figure, Axes]:
        if ax is None:
            fig, ax = plt.subplots(figsize=self.style.figsize, dpi=self.style.dpi)
        else:
            fig = ax.figure
        
        for byte_idx, snr in snr_by_byte.items():
            ax.plot(snr, label=f'Byte {byte_idx}', alpha=0.7)
        
        ax.set_xlabel("Sample", fontsize=self.style.font_size)
        ax.set_ylabel("SNR", fontsize=self.style.font_size)
        ax.set_title(title, fontsize=self.style.title_size)
        ax.legend(loc='upper right', ncol=2)
        self.style.apply(fig, ax)
        
        return fig, ax


class ResultsPlotter:
    
    def __init__(self, style: PlotStyle = None):
        require_matplotlib()
        self.style = style or PlotStyle()
    
    def plot_comparison(
        self,
        data: Dict[str, np.ndarray],
        xlabel: str = "X",
        ylabel: str = "Y",
        title: str = "Comparison",
        ax: Axes = None
    ) -> Tuple[Figure, Axes]:
        if ax is None:
            fig, ax = plt.subplots(figsize=self.style.figsize, dpi=self.style.dpi)
        else:
            fig = ax.figure
        
        for label, values in data.items():
            ax.plot(values, label=label, linewidth=self.style.line_width)
        
        ax.set_xlabel(xlabel, fontsize=self.style.font_size)
        ax.set_ylabel(ylabel, fontsize=self.style.font_size)
        ax.set_title(title, fontsize=self.style.title_size)
        ax.legend()
        self.style.apply(fig, ax)
        
        return fig, ax
    
    def plot_bar_comparison(
        self,
        labels: List[str],
        values: List[float],
        title: str = "Comparison",
        ylabel: str = "Value",
        ax: Axes = None
    ) -> Tuple[Figure, Axes]:
        if ax is None:
            fig, ax = plt.subplots(figsize=self.style.figsize, dpi=self.style.dpi)
        else:
            fig = ax.figure
        
        x = np.arange(len(labels))
        ax.bar(x, values, color='steelblue')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        
        ax.set_ylabel(ylabel, fontsize=self.style.font_size)
        ax.set_title(title, fontsize=self.style.title_size)
        self.style.apply(fig, ax)
        
        fig.tight_layout()
        
        return fig, ax
    
    def create_summary_figure(
        self,
        traces: np.ndarray,
        snr: np.ndarray,
        t_statistic: np.ndarray,
        correlations: np.ndarray,
        correct_key: Optional[int] = None
    ) -> Figure:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=self.style.dpi)
        
        trace_plotter = TracePlotter(self.style)
        trace_plotter.plot_mean_trace_with_std(traces, ax=axes[0, 0])
        
        snr_plotter = SNRPlotter(self.style)
        snr_plotter.plot_snr(snr, ax=axes[0, 1])
        
        tvla_plotter = TVLAPlotter(self.style)
        tvla_plotter.plot_t_statistic(t_statistic, ax=axes[1, 0])
        
        attack_plotter = AttackPlotter(self.style)
        attack_plotter.plot_correlation_traces(correlations, correct_key, ax=axes[1, 1])
        
        fig.tight_layout()
        
        return fig


def save_figure(fig: Figure, path: Union[str, Path], **kwargs):
    require_matplotlib()
    fig.savefig(path, **kwargs)


def show_figure():
    require_matplotlib()
    plt.show()
