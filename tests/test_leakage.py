import pytest
import numpy as np
import tempfile
from pathlib import Path

from tessera.leakage import (
    LeakageType,
    HammingWeightModel,
    HammingDistanceModel,
    IdentityModel,
    LinearModel,
    BitLeakageModel,
    CombinedLeakageModel,
    create_leakage_model,
    TraceMetadata,
    Trace,
    TraceSet,
    TraceGenerator,
    TraceStorage,
    generate_random_traces,
    NoiseGenerator,
    SNRController,
    ElectronicNoise,
    RealisticNoiseModel,
    TraceAligner,
    TraceFilter,
    TraceNormalizer,
    TraceCompressor,
    PointOfInterestSelector,
    preprocess_traces,
)


class TestHammingWeightModel:
    
    def test_hw8_zero(self):
        hw = HammingWeightModel(8)
        assert hw(0) == 0
    
    def test_hw8_all_ones(self):
        hw = HammingWeightModel(8)
        assert hw(0xFF) == 8
    
    def test_hw8_single_bit(self):
        hw = HammingWeightModel(8)
        for i in range(8):
            assert hw(1 << i) == 1
    
    def test_hw16(self):
        hw = HammingWeightModel(16)
        assert hw(0xFFFF) == 16
        assert hw(0xAAAA) == 8
    
    def test_hw32(self):
        hw = HammingWeightModel(32)
        assert hw(0xFFFFFFFF) == 32
    
    def test_hw_array(self):
        hw = HammingWeightModel(8)
        data = np.array([0, 1, 0xFF, 0x0F], dtype=np.uint8)
        result = hw(data)
        expected = np.array([0, 1, 8, 4], dtype=np.float64)
        np.testing.assert_array_equal(result, expected)
    
    def test_leakage_type(self):
        hw = HammingWeightModel(8)
        assert hw.leakage_type == LeakageType.HAMMING_WEIGHT
    
    def test_name(self):
        hw = HammingWeightModel(8)
        assert hw.name == "HW8"


class TestHammingDistanceModel:
    
    def test_hd_same(self):
        hd = HammingDistanceModel(8)
        assert hd(0x55, 0x55) == 0
    
    def test_hd_opposite(self):
        hd = HammingDistanceModel(8)
        assert hd(0x00, 0xFF) == 8
    
    def test_hd_single_bit(self):
        hd = HammingDistanceModel(8)
        assert hd(0x00, 0x01) == 1
        assert hd(0x00, 0x80) == 1
    
    def test_hd_with_reference(self):
        hd = HammingDistanceModel(8, reference=0xAA)
        assert hd(0xAA) == 0
        assert hd(0x55) == 8
    
    def test_hd_array(self):
        hd = HammingDistanceModel(8)
        data = np.array([0x00, 0xFF, 0x0F], dtype=np.uint8)
        prev = np.array([0xFF, 0xFF, 0x00], dtype=np.uint8)
        result = hd(data, prev)
        expected = np.array([8, 0, 4], dtype=np.float64)
        np.testing.assert_array_equal(result, expected)


class TestIdentityModel:
    
    def test_identity_scalar(self):
        model = IdentityModel()
        assert model(42) == 42.0
    
    def test_identity_with_scale(self):
        model = IdentityModel(scale=2.0)
        assert model(10) == 20.0
    
    def test_identity_with_offset(self):
        model = IdentityModel(offset=5.0)
        assert model(10) == 15.0
    
    def test_identity_array(self):
        model = IdentityModel(scale=0.5, offset=1.0)
        data = np.array([0, 2, 4, 6])
        expected = np.array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_equal(model(data), expected)


class TestLinearModel:
    
    def test_linear_scalar(self):
        model = LinearModel(coefficients=np.array([1, 2, 3]))
        data = np.array([1, 1, 1])
        assert model(data) == 6.0
    
    def test_linear_with_intercept(self):
        model = LinearModel(coefficients=np.array([1, 1]), intercept=10.0)
        data = np.array([2, 3])
        assert model(data) == 15.0


class TestBitLeakageModel:
    
    def test_bit0(self):
        model = BitLeakageModel(0, 8)
        assert model(0b00000001) == 1.0
        assert model(0b00000000) == 0.0
    
    def test_bit7(self):
        model = BitLeakageModel(7, 8)
        assert model(0b10000000) == 1.0
        assert model(0b01111111) == 0.0
    
    def test_bit_array(self):
        model = BitLeakageModel(0, 8)
        data = np.array([0, 1, 2, 3], dtype=np.uint8)
        expected = np.array([0, 1, 0, 1], dtype=np.float64)
        np.testing.assert_array_equal(model(data), expected)


class TestCombinedLeakageModel:
    
    def test_combined_equal_weights(self):
        hw = HammingWeightModel(8)
        identity = IdentityModel()
        combined = CombinedLeakageModel([hw, identity])
        
        result = combined(0xFF)
        assert result == 8.0 + 255.0
    
    def test_combined_with_weights(self):
        hw = HammingWeightModel(8)
        identity = IdentityModel()
        combined = CombinedLeakageModel([hw, identity], weights=[2.0, 0.5])
        
        result = combined(0xFF)
        assert result == 2.0 * 8.0 + 0.5 * 255.0


class TestCreateLeakageModel:
    
    def test_create_hw(self):
        model = create_leakage_model("hamming_weight", bit_width=16)
        assert isinstance(model, HammingWeightModel)
        assert model.bit_width == 16
    
    def test_create_hd(self):
        model = create_leakage_model(LeakageType.HAMMING_DISTANCE, bit_width=8, reference=0x55)
        assert isinstance(model, HammingDistanceModel)
        assert model.reference == 0x55
    
    def test_create_identity(self):
        model = create_leakage_model("identity", scale=2.0, offset=1.0)
        assert isinstance(model, IdentityModel)
        assert model.scale == 2.0
        assert model.offset == 1.0


class TestTrace:
    
    def test_trace_creation(self):
        samples = np.random.randn(100)
        plaintext = np.array([1, 2, 3, 4], dtype=np.uint8)
        
        trace = Trace(samples=samples, plaintext=plaintext)
        
        assert trace.num_samples == 100
        assert len(trace) == 100
        np.testing.assert_array_equal(trace.plaintext, plaintext)
    
    def test_trace_slicing(self):
        samples = np.arange(100, dtype=np.float64)
        trace = Trace(samples=samples)
        
        assert trace[0] == 0.0
        np.testing.assert_array_equal(trace[10:20], np.arange(10, 20))
    
    def test_trace_copy(self):
        samples = np.random.randn(50)
        trace = Trace(samples=samples)
        trace_copy = trace.copy()
        
        trace_copy.samples[0] = 9999.0
        assert trace.samples[0] != 9999.0


class TestTraceSet:
    
    def test_traceset_creation(self):
        traces = [Trace(samples=np.random.randn(100)) for _ in range(10)]
        traceset = TraceSet(traces=traces)
        
        assert traceset.num_traces == 10
        assert traceset.num_samples == 100
    
    def test_traceset_samples_matrix(self):
        samples = [np.arange(50, dtype=np.float64) + i for i in range(5)]
        traces = [Trace(samples=s) for s in samples]
        traceset = TraceSet(traces=traces)
        
        matrix = traceset.get_samples_matrix()
        assert matrix.shape == (5, 50)
    
    def test_traceset_subset(self):
        traces = [Trace(samples=np.random.randn(100)) for _ in range(10)]
        traceset = TraceSet(traces=traces)
        
        subset = traceset.subset([0, 2, 4])
        assert subset.num_traces == 3
    
    def test_traceset_mean_trace(self):
        samples = [np.ones(10) * i for i in range(5)]
        traces = [Trace(samples=s) for s in samples]
        traceset = TraceSet(traces=traces)
        
        mean = traceset.mean_trace()
        np.testing.assert_array_almost_equal(mean, np.ones(10) * 2.0)


class TestTraceGenerator:
    
    def test_basic_generation(self):
        hw = HammingWeightModel(8)
        gen = TraceGenerator(leakage_model=hw, num_samples_per_op=5)
        
        gen.record_operation("test", 0xFF)
        gen.record_operation("test2", 0x00)
        
        trace = gen.generate_trace()
        assert trace.num_samples == 10
    
    def test_generation_with_noise(self):
        hw = HammingWeightModel(8)
        gen = TraceGenerator(leakage_model=hw, num_samples_per_op=100, noise_std=0.1)
        
        gen.record_operation("test", 0xFF)
        trace = gen.generate_trace()
        
        assert np.std(trace.samples) > 0


class TestTraceStorage:
    
    def test_save_load_numpy(self):
        traces = [Trace(
            samples=np.random.randn(100),
            plaintext=np.random.randint(0, 256, 16, dtype=np.uint8)
        ) for _ in range(5)]
        traceset = TraceSet(traces=traces)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "traces.npz"
            TraceStorage.save_numpy(traceset, path)
            
            loaded = TraceStorage.load_numpy(path)
            
            assert loaded.num_traces == 5
            assert loaded.num_samples == 100
            np.testing.assert_array_almost_equal(
                traceset.get_samples_matrix(),
                loaded.get_samples_matrix()
            )


class TestGenerateRandomTraces:
    
    def test_basic_generation(self):
        traceset = generate_random_traces(
            num_traces=10,
            num_samples=100,
            noise_std=0.5
        )
        
        assert traceset.num_traces == 10
        assert traceset.num_samples == 100
    
    def test_with_poi(self):
        poi = [0, 5, 10, 15]
        traceset = generate_random_traces(
            num_traces=20,
            num_samples=50,
            poi_indices=poi,
            noise_std=0.1
        )
        
        samples = traceset.get_samples_matrix()
        variance_at_poi = np.var(samples[:, poi], axis=0)
        variance_outside = np.var(samples[:, [1, 2, 3]], axis=0)
        
        assert np.mean(variance_at_poi) > np.mean(variance_outside)


class TestNoiseGenerator:
    
    def test_gaussian_shape(self):
        gen = NoiseGenerator(seed=42)
        noise = gen.gaussian((10, 20))
        assert noise.shape == (10, 20)
    
    def test_gaussian_reproducible(self):
        gen1 = NoiseGenerator(seed=42)
        gen2 = NoiseGenerator(seed=42)
        
        noise1 = gen1.gaussian(100)
        noise2 = gen2.gaussian(100)
        
        np.testing.assert_array_equal(noise1, noise2)
    
    def test_uniform_range(self):
        gen = NoiseGenerator(seed=42)
        noise = gen.uniform(1000, low=-5, high=5)
        
        assert np.min(noise) >= -5
        assert np.max(noise) <= 5


class TestSNRController:
    
    def test_calculate_snr(self):
        signal = np.ones(100)
        noise = np.zeros(100)
        
        snr = SNRController.calculate_snr(signal, noise)
        assert snr == float('inf')
    
    def test_add_noise_for_snr(self):
        snr_ctrl = SNRController(target_snr_db=20.0)
        signal = np.ones(1000) * 10.0
        
        noisy, noise_std = snr_ctrl.add_noise_for_snr(signal)
        
        assert noisy.shape == signal.shape
        assert noise_std > 0


class TestElectronicNoise:
    
    def test_thermal_noise(self):
        noise_model = ElectronicNoise(thermal_std=0.1)
        noise = noise_model.thermal_noise(1000)
        
        assert len(noise) == 1000
        assert np.abs(np.std(noise) - 0.1) < 0.05
    
    def test_apply_all(self):
        noise_model = ElectronicNoise()
        signal = np.ones(100)
        noisy = noise_model.apply_all(signal)
        
        assert noisy.shape == signal.shape
        assert not np.allclose(noisy, signal)


class TestTraceAligner:
    
    def test_correlation_alignment(self):
        reference = np.zeros(100)
        reference[40:60] = 1.0
        
        shifted = np.zeros(100)
        shifted[45:65] = 1.0
        
        aligner = TraceAligner(reference=reference, method="correlation")
        aligned, shift = aligner.align(shifted)
        
        assert shift != 0
    
    def test_align_traces(self):
        reference = np.zeros(100)
        reference[50] = 1.0
        
        traces = np.zeros((5, 100))
        for i in range(5):
            traces[i, 50 + i] = 1.0
        
        aligner = TraceAligner(reference=reference, method="correlation")
        aligned, shifts = aligner.align_traces(traces)
        
        assert aligned.shape == traces.shape


class TestTraceFilter:
    
    def test_lowpass(self):
        filt = TraceFilter(sample_rate=1e6)
        
        noise = np.random.randn(1000) * 0.5
        filtered = filt.lowpass(noise, cutoff=100000)
        
        assert np.std(filtered) < np.std(noise)
    
    def test_moving_average(self):
        filt = TraceFilter()
        signal = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        
        smoothed = filt.moving_average(signal, window=3)
        assert len(smoothed) == len(signal)


class TestTraceNormalizer:
    
    def test_z_score(self):
        trace = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        normalized = TraceNormalizer.z_score(trace)
        
        assert np.abs(np.mean(normalized)) < 1e-10
        assert np.abs(np.std(normalized) - 1.0) < 1e-10
    
    def test_min_max(self):
        trace = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        normalized = TraceNormalizer.min_max(trace)
        
        assert np.min(normalized) == 0.0
        assert np.max(normalized) == 1.0


class TestTraceCompressor:
    
    def test_downsample(self):
        trace = np.arange(100, dtype=np.float64)
        downsampled = TraceCompressor.downsample(trace, factor=5)
        
        assert len(downsampled) == 20
        np.testing.assert_array_equal(downsampled, np.arange(0, 100, 5))
    
    def test_average_downsample(self):
        trace = np.arange(100, dtype=np.float64)
        downsampled = TraceCompressor.average_downsample(trace, factor=5)
        
        assert len(downsampled) == 20
    
    def test_pca(self):
        traces = np.random.randn(50, 100)
        projected, components, eigenvalues = TraceCompressor.principal_components(traces, n_components=5)
        
        assert projected.shape == (50, 5)
        assert components.shape == (100, 5)
        assert len(eigenvalues) == 5


class TestPointOfInterestSelector:
    
    def test_select_by_variance(self):
        np.random.seed(42)
        traces = np.random.randn(100, 50) * 0.1
        traces[:, [10, 20, 30]] += np.random.randn(100, 3) * 10.0
        
        selector = PointOfInterestSelector(num_poi=3)
        poi = selector.select_by_variance(traces)
        
        assert len(poi) == 3
        assert 10 in poi and 20 in poi and 30 in poi
    
    def test_select_by_snr(self):
        traces = np.random.randn(100, 50) * 0.1
        labels = np.array([0] * 50 + [1] * 50)
        
        traces[:50, 10] += 1.0
        traces[50:, 10] += 5.0
        
        selector = PointOfInterestSelector(num_poi=5)
        poi = selector.select_by_snr(traces, labels)
        
        assert 10 in poi


class TestPreprocessTraces:
    
    def test_preprocess_normalize(self):
        traces = np.random.randn(10, 100) * 10 + 50
        
        processed = preprocess_traces(traces, normalize=True)
        
        for i in range(len(processed)):
            assert np.abs(np.mean(processed[i])) < 0.5
    
    def test_preprocess_compress(self):
        traces = np.random.randn(10, 100)
        
        processed = preprocess_traces(traces, normalize=False, compress_factor=5)
        
        assert processed.shape == (10, 20)
    
    def test_preprocess_poi(self):
        traces = np.random.randn(10, 100)
        poi = np.array([0, 10, 20, 30])
        
        processed = preprocess_traces(traces, normalize=False, poi_indices=poi)
        
        assert processed.shape == (10, 4)
