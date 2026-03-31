import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import tempfile
import os
from pathlib import Path

from tessera.analysis import (
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
    ExportMetadata,
    CSVExporter,
    LaTeXExporter,
    JSONExporter,
    NPZExporter,
    ReportGenerator,
)


class TestMetricResult:

    def test_scalar_result(self):
        result = MetricResult(MetricType.SUCCESS_RATE, 0.95)
        assert result.metric_type == MetricType.SUCCESS_RATE
        assert result.value == 0.95
        assert "0.95" in repr(result)

    def test_array_result(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = MetricResult(MetricType.SNR, arr)
        assert result.metric_type == MetricType.SNR
        assert_array_equal(result.value, arr)
        assert "shape=" in repr(result)

    def test_with_metadata(self):
        result = MetricResult(
            MetricType.CORRELATION, 
            0.8,
            {"max_index": 42, "sample_count": 1000}
        )
        assert result.metadata["max_index"] == 42


class TestSNRCalculator:

    def test_compute_snr_binary_labels(self):
        calc = SNRCalculator()
        
        rng = np.random.default_rng(42)
        group0 = rng.normal(0, 0.1, (50, 100))
        group1 = rng.normal(1, 0.1, (50, 100))
        traces = np.vstack([group0, group1])
        labels = np.array([0] * 50 + [1] * 50)
        
        snr = calc.compute_snr(traces, labels)
        
        assert len(snr) == 100
        assert np.mean(snr) > 10

    def test_compute_snr_multiclass(self):
        calc = SNRCalculator()
        
        rng = np.random.default_rng(42)
        traces = []
        labels = []
        for i in range(4):
            traces.append(rng.normal(i, 0.1, (25, 50)))
            labels.extend([i] * 25)
        
        traces = np.vstack(traces)
        labels = np.array(labels)
        
        snr = calc.compute_snr(traces, labels)
        
        assert len(snr) == 50
        assert np.all(snr >= 0)

    def test_compute_snr_multivariate(self):
        calc = SNRCalculator()
        
        rng = np.random.default_rng(42)
        traces = rng.standard_normal((100, 200))
        labels = rng.integers(0, 4, 100)
        
        top_snr, top_indices = calc.compute_snr_multivariate(traces, labels, num_components=5)
        
        assert len(top_snr) == 5
        assert len(top_indices) == 5

    def test_compute_snr_by_byte(self):
        calc = SNRCalculator()
        
        rng = np.random.default_rng(42)
        traces = rng.standard_normal((100, 50))
        data = rng.integers(0, 256, (100, 16), dtype=np.uint8)
        
        snr = calc.compute_snr_by_byte(traces, data, byte_index=0)
        
        assert len(snr) == 50


class TestTVLAMetrics:

    def test_welch_t_test_no_leakage(self):
        tvla = TVLAMetrics()
        
        rng = np.random.default_rng(42)
        group1 = rng.normal(0, 1, (500, 100))
        group2 = rng.normal(0, 1, (500, 100))
        
        t_stat = tvla.welch_t_test(group1, group2)
        
        assert len(t_stat) == 100
        assert np.max(np.abs(t_stat)) < 4.5

    def test_welch_t_test_with_leakage(self):
        tvla = TVLAMetrics()
        
        rng = np.random.default_rng(42)
        group1 = rng.normal(0, 0.1, (500, 100))
        group2 = rng.normal(0, 0.1, (500, 100))
        group1[:, 50] += 0.5
        
        t_stat = tvla.welch_t_test(group1, group2)
        
        has_leakage, points = tvla.detect_leakage(t_stat)
        
        assert has_leakage
        assert points[50]

    def test_leakage_summary(self):
        tvla = TVLAMetrics()
        
        t_stat = np.zeros(100)
        t_stat[25] = 5.0
        t_stat[75] = -6.0
        
        summary = tvla.leakage_summary(t_stat)
        
        assert summary["max_t_statistic"] == 6.0
        assert summary["leakage_at_4.5"] == True
        assert summary["num_leaking_points_4.5"] == 2
        assert 25 in summary["leaking_indices_4.5"]
        assert 75 in summary["leaking_indices_4.5"]

    def test_higher_order_tvla(self):
        tvla = TVLAMetrics()
        
        rng = np.random.default_rng(42)
        traces = rng.standard_normal((100, 50))
        labels = np.array([0] * 50 + [1] * 50)
        
        t_stat_order2 = tvla.compute_higher_order_tvla(traces, labels, order=2)
        
        assert len(t_stat_order2) == 50


class TestSuccessRateMetrics:

    def test_compute_success_rate_perfect(self):
        metrics = SuccessRateMetrics()
        
        predicted = np.array([0, 1, 2, 3, 4])
        actual = np.array([0, 1, 2, 3, 4])
        
        sr = metrics.compute_success_rate(predicted, actual)
        
        assert sr == 1.0

    def test_compute_success_rate_partial(self):
        metrics = SuccessRateMetrics()
        
        predicted = np.array([0, 1, 2, 3, 4])
        actual = np.array([0, 1, 9, 3, 9])
        
        sr = metrics.compute_success_rate(predicted, actual)
        
        assert sr == 0.6

    def test_compute_partial_success_rate(self):
        metrics = SuccessRateMetrics()
        
        rankings = np.array([5, 3, 0, 2, 1, 4])
        actual_key = 2
        
        psr = metrics.compute_partial_success_rate(rankings, actual_key, rank_threshold=1)
        
        assert psr == 1.0

    def test_compute_rank_over_time(self):
        metrics = SuccessRateMetrics()
        
        correlations_over_time = [
            np.array([0.1, 0.8, 0.3, 0.5]),
            np.array([0.2, 0.7, 0.5, 0.4]),
            np.array([0.3, 0.6, 0.8, 0.2]),
        ]
        
        ranks = metrics.compute_rank_over_time(correlations_over_time, correct_key=2)
        
        assert len(ranks) == 3
        assert ranks[2] == 0


class TestGuessingEntropy:

    def test_compute_guessing_entropy_correct_first(self):
        ge = GuessingEntropy()
        
        probs = np.array([0.1, 0.5, 0.3, 0.1])
        actual_key = 1
        
        entropy = ge.compute_guessing_entropy(probs, actual_key)
        
        assert entropy == 0.0

    def test_compute_guessing_entropy_correct_last(self):
        ge = GuessingEntropy()
        
        probs = np.array([0.5, 0.3, 0.15, 0.05])
        actual_key = 3
        
        entropy = ge.compute_guessing_entropy(probs, actual_key)
        
        assert entropy == np.log2(4)

    def test_compute_guessing_entropy_from_correlations(self):
        ge = GuessingEntropy()
        
        correlations = np.array([0.1, -0.9, 0.3, 0.5])
        actual_key = 1
        
        entropy = ge.compute_guessing_entropy_from_correlations(correlations, actual_key)
        
        assert entropy == 0.0


class TestCorrelationMetrics:

    def test_pearson_correlation_perfect(self):
        metrics = CorrelationMetrics()
        
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])
        
        corr = metrics.pearson_correlation(x, y)
        
        assert_allclose(corr, 1.0, atol=1e-10)

    def test_pearson_correlation_negative(self):
        metrics = CorrelationMetrics()
        
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([10, 8, 6, 4, 2])
        
        corr = metrics.pearson_correlation(x, y)
        
        assert_allclose(corr, -1.0, atol=1e-10)

    def test_correlation_matrix(self):
        metrics = CorrelationMetrics()
        
        rng = np.random.default_rng(42)
        traces = rng.standard_normal((100, 50))
        hypotheses = rng.standard_normal((100, 4))
        
        corr_matrix = metrics.correlation_matrix(traces, hypotheses)
        
        assert corr_matrix.shape == (4, 50)

    def test_max_correlation_by_key(self):
        metrics = CorrelationMetrics()
        
        corr_matrix = np.array([
            [0.1, 0.5, 0.2],
            [0.3, 0.2, 0.8],
            [0.4, 0.1, 0.3]
        ])
        
        max_corr, max_idx = metrics.max_correlation_by_key(corr_matrix)
        
        assert_array_equal(max_corr, [0.5, 0.8, 0.4])
        assert_array_equal(max_idx, [1, 2, 0])


class TestMutualInformation:

    def test_compute_mi_perfect_dependency(self):
        mi = MutualInformation(num_bins=16)
        
        x = np.repeat(np.arange(16), 10)
        y = x.copy()
        
        mi_value = mi.compute_mi(x, y)
        
        assert mi_value > 0

    def test_compute_mi_independent(self):
        mi = MutualInformation(num_bins=16)
        
        rng = np.random.default_rng(42)
        x = rng.integers(0, 16, 1000)
        y = rng.integers(0, 16, 1000)
        
        mi_value = mi.compute_mi(x, y)
        
        assert mi_value < 0.5

    def test_mi_by_sample(self):
        mi = MutualInformation(num_bins=16)
        
        rng = np.random.default_rng(42)
        traces = rng.standard_normal((100, 50))
        labels = rng.integers(0, 4, 100)
        
        mi_values = mi.mi_by_sample(traces, labels)
        
        assert len(mi_values) == 50


class TestAttackPerformanceMetrics:

    def test_to_dict(self):
        metrics = AttackPerformanceMetrics(
            success_rate=0.95,
            guessing_entropy=0.5,
            key_rank=1,
            num_traces_used=1000,
            execution_time_seconds=5.5,
            snr_max=10.5,
            correlation_max=0.9
        )
        
        d = metrics.to_dict()
        
        assert d["success_rate"] == 0.95
        assert d["key_rank"] == 1
        assert d["correlation_max"] == 0.9


class TestMetricsAggregator:

    def test_compute_all_metrics(self):
        agg = MetricsAggregator()
        
        rng = np.random.default_rng(42)
        traces = rng.standard_normal((100, 50))
        labels = np.array([0] * 50 + [1] * 50)
        
        results = agg.compute_all_metrics(traces, labels)
        
        assert "snr" in results
        assert "tvla" in results
        assert "mutual_information" in results

    def test_generate_report(self):
        agg = MetricsAggregator()
        
        metrics = {
            "snr": MetricResult(MetricType.SNR, np.array([1.0, 2.0]), {"max_snr": 2.0, "max_snr_index": 1}),
            "tvla": MetricResult(MetricType.TVLA, np.array([3.0, 4.0]), 
                                {"max_t_statistic": 4.0, "leakage_at_4.5": False, "num_leaking_points_4.5": 0}),
            "success_rate": MetricResult(MetricType.SUCCESS_RATE, 0.95)
        }
        
        report = agg.generate_report(metrics)
        
        assert "SNR" in report
        assert "TVLA" in report
        assert "95" in report


class TestCSVExporter:

    def test_export_traces(self):
        exporter = CSVExporter()
        
        traces = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            path = f.name
        
        try:
            exporter.export_traces(traces, path)
            
            with open(path, 'r') as f:
                content = f.read()
            
            assert "s0" in content
            assert "1.000000" in content
        finally:
            os.unlink(path)

    def test_export_correlations(self):
        exporter = CSVExporter()
        
        correlations = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            path = f.name
        
        try:
            exporter.export_correlations(correlations, path)
            
            with open(path, 'r') as f:
                content = f.read()
            
            assert "key_0" in content
        finally:
            os.unlink(path)

    def test_export_metrics(self):
        exporter = CSVExporter()
        
        metrics = {"snr_max": 10.5, "success_rate": 0.95}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            path = f.name
        
        try:
            exporter.export_metrics(metrics, path)
            
            with open(path, 'r') as f:
                content = f.read()
            
            assert "snr_max" in content
            assert "10.5" in content
        finally:
            os.unlink(path)


class TestLaTeXExporter:

    def test_export_metrics_table(self):
        exporter = LaTeXExporter()
        
        metrics = {"SNR Max": 10.5, "Success Rate": 0.95}
        
        latex = exporter.export_metrics_table(metrics)
        
        assert "\\begin{table}" in latex
        assert "SNR Max" in latex
        assert "10.5" in latex
        assert "\\end{table}" in latex

    def test_export_attack_results_table(self):
        exporter = LaTeXExporter()
        
        results = [
            {"attack": "CPA", "success_rate": 0.95},
            {"attack": "DPA", "success_rate": 0.85}
        ]
        
        latex = exporter.export_attack_results_table(results)
        
        assert "CPA" in latex
        assert "DPA" in latex
        assert "0.95" in latex

    def test_export_comparison_table(self):
        exporter = LaTeXExporter()
        
        data = {
            "Method A": {"accuracy": 0.95, "time": 1.5},
            "Method B": {"accuracy": 0.90, "time": 0.5}
        }
        
        latex = exporter.export_comparison_table(data)
        
        assert "Method A" in latex
        assert "accuracy" in latex


class TestJSONExporter:

    def test_export_and_load(self):
        exporter = JSONExporter()
        
        data = {"snr": [1.0, 2.0, 3.0], "key_rank": 1}
        metadata = ExportMetadata(name="test", description="Test export")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            path = f.name
        
        try:
            exporter.export(data, path, metadata)
            loaded_data, loaded_metadata = exporter.load(path)
            
            assert loaded_data["snr"] == [1.0, 2.0, 3.0]
            assert loaded_data["key_rank"] == 1
            assert loaded_metadata.name == "test"
        finally:
            os.unlink(path)

    def test_export_traces(self):
        exporter = JSONExporter()
        
        traces = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            path = f.name
        
        try:
            exporter.export_traces(traces, path)
            loaded, _ = exporter.load(path)
            
            assert loaded["shape"] == [2, 2]
            assert len(loaded["traces"]) == 2
        finally:
            os.unlink(path)


class TestNPZExporter:

    def test_export_and_load_traces(self):
        exporter = NPZExporter()
        
        traces = np.array([[1.0, 2.0], [3.0, 4.0]])
        labels = np.array([0, 1])
        
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            path = f.name
        
        try:
            exporter.export_traces(path, traces, labels=labels)
            loaded = exporter.load_traces(path)
            
            assert_array_equal(loaded["traces"], traces)
            assert_array_equal(loaded["labels"], labels)
        finally:
            os.unlink(path)

    def test_export_attack_state(self):
        exporter = NPZExporter()
        
        correlations = np.array([[0.1, 0.2], [0.3, 0.4]])
        hypotheses = np.array([[1, 2], [3, 4]])
        
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            path = f.name
        
        try:
            exporter.export_attack_state(path, correlations, hypotheses, 100)
            loaded = exporter.load_traces(path)
            
            assert_array_equal(loaded["correlations"], correlations)
            assert loaded["num_traces_processed"][0] == 100
        finally:
            os.unlink(path)


class TestReportGenerator:

    def test_generate_text_report(self):
        generator = ReportGenerator()
        
        metrics = {"SNR Max": 10.5, "t-statistic Max": 5.0}
        attack_results = {"success_rate": 0.95, "key_rank": 1}
        
        report = generator.generate_text_report(metrics, attack_results)
        
        assert "METRICS SUMMARY" in report
        assert "ATTACK RESULTS" in report
        assert "SNR Max" in report

    def test_export_full_report(self):
        generator = ReportGenerator()
        
        traces = np.random.randn(10, 20)
        metrics = {"snr_max": 10.5, "success_rate": 0.95}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            generator.export_full_report(
                tmpdir,
                traces=traces,
                metrics=metrics,
                name="test"
            )
            
            files = os.listdir(tmpdir)
            assert "test_traces.csv" in files
            assert "test_metrics.csv" in files
            assert "test_metrics.tex" in files
            assert "test_report.txt" in files


class TestExportMetadata:

    def test_default_timestamp(self):
        metadata = ExportMetadata(name="test")
        
        assert metadata.name == "test"
        assert metadata.timestamp is not None
        assert metadata.version == "1.0"

    def test_custom_fields(self):
        metadata = ExportMetadata(
            name="analysis",
            description="Test analysis",
            custom={"algorithm": "Kyber", "traces": 1000}
        )
        
        assert metadata.custom["algorithm"] == "Kyber"
