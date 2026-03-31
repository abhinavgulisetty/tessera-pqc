from .metrics import (
    MetricType,
    MetricResult,
    SNRCalculator,
    TVLAMetrics,
    SuccessRateMetrics,
    GuessingEntropy,
    CorrelationMetrics,
    MutualInformation,
    AttackPerformanceMetrics,
    MetricsAggregator,
)

from .export import (
    ExportMetadata,
    CSVExporter,
    LaTeXExporter,
    JSONExporter,
    NPZExporter,
    ReportGenerator,
)

try:
    from .export import DataFrameExporter
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

try:
    from .plotting import (
        PlotStyle,
        TracePlotter,
        AttackPlotter,
        TVLAPlotter,
        SNRPlotter,
        ResultsPlotter,
        save_figure,
        show_figure,
    )
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

try:
    from .interactive import (
        InteractiveStyle,
        InteractiveTracePlotter,
        InteractiveAttackPlotter,
        InteractiveTVLAPlotter,
        InteractiveSNRPlotter,
        InteractiveDashboard,
        save_interactive_figure,
        show_interactive_figure,
    )
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False


__all__ = [
    "MetricType",
    "MetricResult",
    "SNRCalculator",
    "TVLAMetrics",
    "SuccessRateMetrics",
    "GuessingEntropy",
    "CorrelationMetrics",
    "MutualInformation",
    "AttackPerformanceMetrics",
    "MetricsAggregator",
    "ExportMetadata",
    "CSVExporter",
    "LaTeXExporter",
    "JSONExporter",
    "NPZExporter",
    "ReportGenerator",
]

if _HAS_PANDAS:
    __all__.append("DataFrameExporter")

if _HAS_MATPLOTLIB:
    __all__.extend([
        "PlotStyle",
        "TracePlotter",
        "AttackPlotter",
        "TVLAPlotter",
        "SNRPlotter",
        "ResultsPlotter",
        "save_figure",
        "show_figure",
    ])

if _HAS_PLOTLY:
    __all__.extend([
        "InteractiveStyle",
        "InteractiveTracePlotter",
        "InteractiveAttackPlotter",
        "InteractiveTVLAPlotter",
        "InteractiveSNRPlotter",
        "InteractiveDashboard",
        "save_interactive_figure",
        "show_interactive_figure",
    ])
