from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Union, Any
from pathlib import Path
import json
import warnings

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    go = None
    px = None


def require_plotly():
    if not HAS_PLOTLY:
        raise ImportError("plotly is required for interactive plotting. Install with: pip install plotly")


@dataclass 
class InteractiveStyle:
    width: int = 900
    height: int = 600
    template: str = "plotly_white"
    colorscale: str = "Viridis"
    line_width: float = 1.5
    marker_size: int = 8
    font_size: int = 12
    title_size: int = 16


class InteractiveTracePlotter:
    
    def __init__(self, style: InteractiveStyle = None):
        require_plotly()
        self.style = style or InteractiveStyle()
    
    def plot_single_trace(
        self,
        trace: np.ndarray,
        title: str = "Power Trace",
        xlabel: str = "Sample",
        ylabel: str = "Power"
    ) -> go.Figure:
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            y=trace,
            mode='lines',
            name='Trace',
            line=dict(width=self.style.line_width)
        ))
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=self.style.title_size)),
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            width=self.style.width,
            height=self.style.height,
            template=self.style.template,
            font=dict(size=self.style.font_size)
        )
        
        return fig
    
    def plot_multiple_traces(
        self,
        traces: np.ndarray,
        num_traces: int = 10,
        title: str = "Power Traces"
    ) -> go.Figure:
        fig = go.Figure()
        
        n = min(num_traces, len(traces))
        for i in range(n):
            fig.add_trace(go.Scatter(
                y=traces[i],
                mode='lines',
                name=f'Trace {i}',
                opacity=0.6,
                line=dict(width=self.style.line_width)
            ))
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=self.style.title_size)),
            xaxis_title="Sample",
            yaxis_title="Power",
            width=self.style.width,
            height=self.style.height,
            template=self.style.template,
            font=dict(size=self.style.font_size),
            showlegend=True
        )
        
        return fig
    
    def plot_mean_with_std(
        self,
        traces: np.ndarray,
        title: str = "Mean Trace with Standard Deviation"
    ) -> go.Figure:
        mean = np.mean(traces, axis=0)
        std = np.std(traces, axis=0)
        x = np.arange(len(mean))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=x, y=mean + std,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=x, y=mean - std,
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(68, 68, 68, 0.3)',
            name='±1 std'
        ))
        
        fig.add_trace(go.Scatter(
            x=x, y=mean,
            mode='lines',
            name='Mean',
            line=dict(width=self.style.line_width, color='blue')
        ))
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=self.style.title_size)),
            xaxis_title="Sample",
            yaxis_title="Power",
            width=self.style.width,
            height=self.style.height,
            template=self.style.template,
            font=dict(size=self.style.font_size)
        )
        
        return fig
    
    def plot_trace_heatmap(
        self,
        traces: np.ndarray,
        title: str = "Trace Heatmap"
    ) -> go.Figure:
        fig = go.Figure(data=go.Heatmap(
            z=traces,
            colorscale=self.style.colorscale,
            colorbar=dict(title="Power")
        ))
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=self.style.title_size)),
            xaxis_title="Sample",
            yaxis_title="Trace Index",
            width=self.style.width,
            height=self.style.height,
            template=self.style.template,
            font=dict(size=self.style.font_size)
        )
        
        return fig


class InteractiveAttackPlotter:
    
    def __init__(self, style: InteractiveStyle = None):
        require_plotly()
        self.style = style or InteractiveStyle()
    
    def plot_correlation_traces(
        self,
        correlations: np.ndarray,
        correct_key: Optional[int] = None,
        title: str = "Correlation Traces"
    ) -> go.Figure:
        fig = go.Figure()
        
        n_keys = correlations.shape[0]
        
        for i in range(n_keys):
            if correct_key is not None and i == correct_key:
                fig.add_trace(go.Scatter(
                    y=correlations[i],
                    mode='lines',
                    name=f'Key {i} (correct)',
                    line=dict(width=self.style.line_width * 2, color='red')
                ))
            else:
                fig.add_trace(go.Scatter(
                    y=correlations[i],
                    mode='lines',
                    name=f'Key {i}',
                    opacity=0.3,
                    line=dict(width=self.style.line_width, color='gray'),
                    showlegend=False
                ))
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=self.style.title_size)),
            xaxis_title="Sample",
            yaxis_title="Correlation",
            width=self.style.width,
            height=self.style.height,
            template=self.style.template,
            font=dict(size=self.style.font_size)
        )
        
        return fig
    
    def plot_correlation_heatmap(
        self,
        correlations: np.ndarray,
        title: str = "Correlation Matrix"
    ) -> go.Figure:
        max_val = np.max(np.abs(correlations))
        
        fig = go.Figure(data=go.Heatmap(
            z=correlations,
            colorscale='RdBu_r',
            zmin=-max_val,
            zmax=max_val,
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=self.style.title_size)),
            xaxis_title="Sample",
            yaxis_title="Key Hypothesis",
            width=self.style.width,
            height=self.style.height,
            template=self.style.template,
            font=dict(size=self.style.font_size)
        )
        
        return fig
    
    def plot_key_ranking(
        self,
        correlations: np.ndarray,
        correct_key: Optional[int] = None,
        title: str = "Key Ranking"
    ) -> go.Figure:
        max_corr = np.max(np.abs(correlations), axis=1)
        sorted_indices = np.argsort(max_corr)[::-1]
        
        colors = ['red' if correct_key is not None and i == correct_key else 'steelblue' 
                  for i in sorted_indices]
        
        fig = go.Figure(data=go.Bar(
            x=list(range(len(max_corr))),
            y=max_corr[sorted_indices],
            marker_color=colors,
            text=[f'Key {i}' for i in sorted_indices],
            hovertemplate='%{text}<br>Correlation: %{y:.4f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=self.style.title_size)),
            xaxis_title="Rank",
            yaxis_title="Max Correlation",
            width=self.style.width,
            height=self.style.height,
            template=self.style.template,
            font=dict(size=self.style.font_size)
        )
        
        return fig
    
    def plot_success_rate_vs_traces(
        self,
        trace_counts: np.ndarray,
        success_rates: np.ndarray,
        title: str = "Success Rate vs Number of Traces"
    ) -> go.Figure:
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=trace_counts,
            y=success_rates,
            mode='lines+markers',
            marker=dict(size=self.style.marker_size),
            line=dict(width=self.style.line_width)
        ))
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=self.style.title_size)),
            xaxis_title="Number of Traces",
            yaxis_title="Success Rate",
            yaxis=dict(range=[0, 1.05]),
            width=self.style.width,
            height=self.style.height,
            template=self.style.template,
            font=dict(size=self.style.font_size)
        )
        
        return fig
    
    def plot_rank_evolution(
        self,
        trace_counts: np.ndarray,
        ranks: np.ndarray,
        title: str = "Key Rank Evolution"
    ) -> go.Figure:
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=trace_counts,
            y=ranks + 1,
            mode='lines+markers',
            marker=dict(size=self.style.marker_size),
            line=dict(width=self.style.line_width)
        ))
        
        fig.add_hline(y=1, line_dash="dash", line_color="red", 
                      annotation_text="Rank 1")
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=self.style.title_size)),
            xaxis_title="Number of Traces",
            yaxis_title="Key Rank",
            yaxis_type="log",
            width=self.style.width,
            height=self.style.height,
            template=self.style.template,
            font=dict(size=self.style.font_size)
        )
        
        return fig


class InteractiveTVLAPlotter:
    
    def __init__(self, style: InteractiveStyle = None):
        require_plotly()
        self.style = style or InteractiveStyle()
    
    def plot_t_statistic(
        self,
        t_statistic: np.ndarray,
        threshold: float = 4.5,
        title: str = "TVLA t-statistic"
    ) -> go.Figure:
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            y=t_statistic,
            mode='lines',
            name='t-statistic',
            line=dict(width=self.style.line_width)
        ))
        
        fig.add_hline(y=threshold, line_dash="dash", line_color="red",
                      annotation_text=f"+{threshold}")
        fig.add_hline(y=-threshold, line_dash="dash", line_color="red",
                      annotation_text=f"-{threshold}")
        fig.add_hline(y=0, line_color="gray", opacity=0.5)
        
        leaking = np.abs(t_statistic) > threshold
        if np.any(leaking):
            leaking_indices = np.where(leaking)[0]
            fig.add_trace(go.Scatter(
                x=leaking_indices,
                y=t_statistic[leaking_indices],
                mode='markers',
                name='Leaking points',
                marker=dict(color='red', size=self.style.marker_size)
            ))
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=self.style.title_size)),
            xaxis_title="Sample",
            yaxis_title="t-statistic",
            width=self.style.width,
            height=self.style.height,
            template=self.style.template,
            font=dict(size=self.style.font_size)
        )
        
        return fig
    
    def plot_t_statistic_3d(
        self,
        t_statistics: List[np.ndarray],
        trace_counts: List[int],
        title: str = "TVLA Evolution 3D"
    ) -> go.Figure:
        z_data = np.array(t_statistics)
        x = np.arange(z_data.shape[1])
        y = np.array(trace_counts)
        
        fig = go.Figure(data=go.Surface(
            x=x, y=y, z=z_data,
            colorscale='RdBu_r',
            colorbar=dict(title="t-statistic")
        ))
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=self.style.title_size)),
            scene=dict(
                xaxis_title="Sample",
                yaxis_title="Number of Traces",
                zaxis_title="t-statistic"
            ),
            width=self.style.width,
            height=self.style.height,
            template=self.style.template,
            font=dict(size=self.style.font_size)
        )
        
        return fig


class InteractiveSNRPlotter:
    
    def __init__(self, style: InteractiveStyle = None):
        require_plotly()
        self.style = style or InteractiveStyle()
    
    def plot_snr(
        self,
        snr: np.ndarray,
        title: str = "Signal-to-Noise Ratio"
    ) -> go.Figure:
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            y=snr,
            mode='lines',
            name='SNR',
            line=dict(width=self.style.line_width)
        ))
        
        max_idx = np.argmax(snr)
        fig.add_trace(go.Scatter(
            x=[max_idx],
            y=[snr[max_idx]],
            mode='markers',
            name=f'Max: {snr[max_idx]:.4f}',
            marker=dict(color='red', size=self.style.marker_size * 1.5)
        ))
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=self.style.title_size)),
            xaxis_title="Sample",
            yaxis_title="SNR",
            width=self.style.width,
            height=self.style.height,
            template=self.style.template,
            font=dict(size=self.style.font_size)
        )
        
        return fig
    
    def plot_snr_by_byte(
        self,
        snr_by_byte: Dict[int, np.ndarray],
        title: str = "SNR by Key Byte"
    ) -> go.Figure:
        fig = go.Figure()
        
        for byte_idx, snr in snr_by_byte.items():
            fig.add_trace(go.Scatter(
                y=snr,
                mode='lines',
                name=f'Byte {byte_idx}',
                opacity=0.7,
                line=dict(width=self.style.line_width)
            ))
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=self.style.title_size)),
            xaxis_title="Sample",
            yaxis_title="SNR",
            width=self.style.width,
            height=self.style.height,
            template=self.style.template,
            font=dict(size=self.style.font_size)
        )
        
        return fig


class InteractiveDashboard:
    
    def __init__(self, style: InteractiveStyle = None):
        require_plotly()
        self.style = style or InteractiveStyle()
    
    def create_analysis_dashboard(
        self,
        traces: np.ndarray,
        snr: np.ndarray,
        t_statistic: np.ndarray,
        correlations: np.ndarray,
        correct_key: Optional[int] = None
    ) -> go.Figure:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Mean Trace with Std',
                'Signal-to-Noise Ratio',
                'TVLA t-statistic',
                'Correlation Traces'
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        mean = np.mean(traces, axis=0)
        std = np.std(traces, axis=0)
        x = np.arange(len(mean))
        
        fig.add_trace(go.Scatter(x=x, y=mean + std, mode='lines', line=dict(width=0),
                                 showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=mean - std, mode='lines', line=dict(width=0),
                                 fill='tonexty', fillcolor='rgba(68,68,68,0.3)',
                                 name='±1 std'), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=mean, mode='lines', name='Mean',
                                 line=dict(color='blue')), row=1, col=1)
        
        fig.add_trace(go.Scatter(y=snr, mode='lines', name='SNR',
                                 line=dict(color='green')), row=1, col=2)
        max_idx = np.argmax(snr)
        fig.add_trace(go.Scatter(x=[max_idx], y=[snr[max_idx]], mode='markers',
                                 name=f'Max SNR', marker=dict(color='red', size=10)),
                     row=1, col=2)
        
        fig.add_trace(go.Scatter(y=t_statistic, mode='lines', name='t-stat',
                                 line=dict(color='purple')), row=2, col=1)
        fig.add_hline(y=4.5, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=-4.5, line_dash="dash", line_color="red", row=2, col=1)
        
        n_keys = correlations.shape[0]
        for i in range(n_keys):
            if correct_key is not None and i == correct_key:
                fig.add_trace(go.Scatter(y=correlations[i], mode='lines',
                                         name=f'Key {i}', line=dict(color='red', width=2)),
                             row=2, col=2)
            else:
                fig.add_trace(go.Scatter(y=correlations[i], mode='lines',
                                         showlegend=False, opacity=0.2,
                                         line=dict(color='gray')), row=2, col=2)
        
        fig.update_layout(
            height=800,
            width=1200,
            title_text="Side-Channel Analysis Dashboard",
            template=self.style.template,
            font=dict(size=self.style.font_size)
        )
        
        return fig
    
    def create_attack_comparison_dashboard(
        self,
        attack_results: Dict[str, Dict],
        title: str = "Attack Comparison"
    ) -> go.Figure:
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Success Rate', 'Key Rank'),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        attack_names = list(attack_results.keys())
        success_rates = [r.get('success_rate', 0) for r in attack_results.values()]
        key_ranks = [r.get('key_rank', 256) for r in attack_results.values()]
        
        fig.add_trace(go.Bar(
            x=attack_names,
            y=success_rates,
            name='Success Rate',
            marker_color='steelblue'
        ), row=1, col=1)
        
        fig.add_trace(go.Bar(
            x=attack_names,
            y=key_ranks,
            name='Key Rank',
            marker_color='coral'
        ), row=1, col=2)
        
        fig.update_layout(
            title_text=title,
            height=500,
            width=1000,
            template=self.style.template,
            font=dict(size=self.style.font_size),
            showlegend=False
        )
        
        return fig


def save_interactive_figure(fig: go.Figure, path: Union[str, Path], **kwargs):
    require_plotly()
    path = Path(path)
    
    if path.suffix == '.html':
        fig.write_html(str(path), **kwargs)
    elif path.suffix == '.json':
        fig.write_json(str(path), **kwargs)
    elif path.suffix in ['.png', '.jpg', '.jpeg', '.svg', '.pdf', '.webp']:
        fig.write_image(str(path), **kwargs)
    else:
        fig.write_html(str(path) + '.html', **kwargs)


def show_interactive_figure(fig: go.Figure):
    require_plotly()
    fig.show()
