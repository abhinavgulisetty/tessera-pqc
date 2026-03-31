import pytest
import numpy as np

from tessera.leakage import (
    HammingWeightModel,
    Trace,
    TraceSet,
    TraceMetadata,
)

from tessera.attacks import (
    AttackType,
    AttackResult,
    LeakageAssessmentResult,
    pearson_correlation,
    welch_t_statistic,
    differential_means,
    snr_distinguisher,
    WelchTTest,
    FixedVsRandomTVLA,
    IncrementalTTest,
    CPA,
    IncrementalCPA,
    HigherOrderCPA,
    SBOX,
    DPA,
    MultiBitDPA,
    IncrementalDPA,
    TemplateAttack,
    TemplateKeyRecovery,
    TemplateProfile,
    NTTCoefficientAttack,
    NTTButterflyAttack,
    SecretKeyRecoveryAttack,
    BeliefPropagationAttack,
    LatticeLeakageAssessment,
    LatticeOperation,
    ButterflyLeakageModel,
)


def create_test_traceset(num_traces: int = 100,
                         num_samples: int = 50,
                         num_key_bytes: int = 16,
                         seed: int = 42) -> TraceSet:
    np.random.seed(seed)
    
    key = np.random.randint(0, 256, size=num_key_bytes, dtype=np.uint8)
    hw_model = HammingWeightModel(8)
    
    traces = []
    for _ in range(num_traces):
        plaintext = np.random.randint(0, 256, size=num_key_bytes, dtype=np.uint8)
        
        samples = np.random.randn(num_samples).astype(np.float32)
        
        for byte_idx in range(min(4, num_key_bytes)):
            intermediate = SBOX[plaintext[byte_idx] ^ key[byte_idx]]
            leakage = hw_model(intermediate)
            poi = byte_idx * 10 + 5
            if poi < num_samples:
                samples[poi] += leakage * 2.0
        
        traces.append(Trace(
            samples=samples,
            plaintext=plaintext,
            key=key.copy(),
        ))
    
    return TraceSet(traces=traces)


class TestStatisticalFunctions:
    
    def test_pearson_correlation_perfect(self):
        x = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        y = x * 2 + 1
        corr = pearson_correlation(x, y)
        assert np.abs(corr - 1.0) < 1e-10
    
    def test_pearson_correlation_negative(self):
        x = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        y = -x
        corr = pearson_correlation(x, y)
        assert np.abs(corr + 1.0) < 1e-10
    
    def test_pearson_correlation_uncorrelated(self):
        np.random.seed(42)
        x = np.random.randn(1000)
        y = np.random.randn(1000)
        corr = pearson_correlation(x, y)
        assert np.abs(corr) < 0.1
    
    def test_pearson_correlation_matrix(self):
        x = np.random.randn(100, 5)
        y = np.random.randn(100, 10)
        corr = pearson_correlation(x, y)
        assert corr.shape == (5, 10)
    
    def test_welch_t_statistic_same_mean(self):
        np.random.seed(42)
        group1 = np.random.randn(100, 10)
        group2 = np.random.randn(100, 10)
        t_stat = welch_t_statistic(group1, group2)
        assert t_stat.shape == (10,)
        assert np.all(np.abs(t_stat) < 3)
    
    def test_welch_t_statistic_different_mean(self):
        np.random.seed(42)
        group1 = np.random.randn(100, 10)
        group2 = np.random.randn(100, 10) + 5.0
        t_stat = welch_t_statistic(group1, group2)
        assert np.all(np.abs(t_stat) > 10)
    
    def test_differential_means(self):
        group1 = np.array([[1, 2, 3], [2, 3, 4]], dtype=np.float64)
        group2 = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float64)
        diff = differential_means(group1, group2)
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(diff, expected)
    
    def test_snr_distinguisher(self):
        np.random.seed(42)
        traces = np.zeros((100, 20))
        labels = np.array([0] * 50 + [1] * 50)
        traces[:50, 10] = np.random.randn(50) + 0
        traces[50:, 10] = np.random.randn(50) + 5
        traces += np.random.randn(100, 20) * 0.1
        snr = snr_distinguisher(traces, labels)
        assert snr.shape == (20,)
        assert np.argmax(snr) == 10


class TestWelchTTest:
    
    def test_ttest_no_leakage(self):
        np.random.seed(42)
        group1 = np.random.randn(100, 50).astype(np.float32)
        group2 = np.random.randn(100, 50).astype(np.float32)
        ttest = WelchTTest()
        result = ttest.assess(group1, group2)
        assert isinstance(result, LeakageAssessmentResult)
        assert result.max_statistic < 4.5
    
    def test_ttest_with_leakage(self):
        np.random.seed(42)
        group1 = np.random.randn(100, 50).astype(np.float32)
        group2 = np.random.randn(100, 50).astype(np.float32)
        group2[:, 25] += 3.0
        ttest = WelchTTest()
        result = ttest.assess(group1, group2)
        assert result.leakage_detected
        assert 25 in result.poi_indices
    
    def test_ttest_threshold(self):
        ttest = WelchTTest(confidence_level=0.99)
        assert ttest.confidence_level == 0.99


class TestFixedVsRandomTVLA:
    
    def test_fixed_vs_random_creation(self):
        tvla = FixedVsRandomTVLA()
        assert tvla.order == 1
    
    def test_fixed_vs_random_higher_order(self):
        tvla = FixedVsRandomTVLA(order=2)
        assert tvla.order == 2


class TestIncrementalTTest:
    
    def test_incremental_update(self):
        np.random.seed(42)
        ttest = IncrementalTTest()
        
        for _ in range(5):
            group1_batch = np.random.randn(10, 20).astype(np.float32)
            group2_batch = np.random.randn(10, 20).astype(np.float32)
            ttest.update_group1(group1_batch)
            ttest.update_group2(group2_batch)
        
        result = ttest.assess()
        assert isinstance(result, LeakageAssessmentResult)
    
    def test_incremental_finalize(self):
        np.random.seed(42)
        ttest = IncrementalTTest()
        
        for _ in range(10):
            group1_batch = np.random.randn(10, 30).astype(np.float32)
            group2_batch = np.random.randn(10, 30).astype(np.float32) + 2.0
            group2_batch[:, 15] += 5.0
            ttest.update_group1(group1_batch)
            ttest.update_group2(group2_batch)
        
        result = ttest.assess()
        assert isinstance(result, LeakageAssessmentResult)


class TestCPA:
    
    def test_cpa_creation(self):
        cpa = CPA()
        assert cpa.attack_type == AttackType.CPA
        assert len(cpa.key_byte_indices) == 16
    
    def test_cpa_custom_leakage_model(self):
        hw = HammingWeightModel(8)
        cpa = CPA(leakage_model=hw, key_byte_indices=[0, 1])
        assert len(cpa.key_byte_indices) == 2
    
    def test_cpa_attack_byte(self):
        traceset = create_test_traceset(num_traces=200, num_samples=50)
        cpa = CPA(key_byte_indices=[0])
        
        samples = traceset.get_samples_matrix()
        plaintexts = traceset.get_plaintexts()
        
        key_byte, confidence, correlations = cpa.attack_byte(samples, plaintexts, 0)
        
        assert 0 <= key_byte < 256
        assert correlations.shape == (50,)
    
    def test_cpa_run(self):
        traceset = create_test_traceset(num_traces=500, num_samples=50)
        cpa = CPA(key_byte_indices=[0])
        result = cpa.run(traceset)
        
        assert isinstance(result, AttackResult)
        assert result.attack_type == AttackType.CPA
        assert result.recovered_bytes is not None
        assert 0 in result.recovered_bytes


class TestIncrementalCPA:
    
    def test_incremental_cpa_update(self):
        cpa = IncrementalCPA(key_byte_indices=[0])
        
        np.random.seed(42)
        for _ in range(5):
            traces = np.random.randn(20, 30).astype(np.float32)
            plaintexts = np.random.randint(0, 256, (20, 16), dtype=np.uint8)
            cpa.update(traces, plaintexts)
        
        assert cpa.trace_count == 100
    
    def test_incremental_cpa_finalize(self):
        cpa = IncrementalCPA(key_byte_indices=[0])
        
        np.random.seed(42)
        for _ in range(10):
            traces = np.random.randn(20, 30).astype(np.float32)
            plaintexts = np.random.randint(0, 256, (20, 16), dtype=np.uint8)
            cpa.update(traces, plaintexts)
        
        result = cpa.finalize()
        assert isinstance(result, AttackResult)
        assert result.attack_type == AttackType.CPA


class TestHigherOrderCPA:
    
    def test_higher_order_creation(self):
        ho_cpa = HigherOrderCPA(order=2)
        assert ho_cpa.order == 2
    
    def test_higher_order_preprocess(self):
        ho_cpa = HigherOrderCPA(order=2)
        traces = np.random.randn(100, 50).astype(np.float32)
        processed = ho_cpa.preprocess_traces(traces, [10, 20, 30])
        assert processed.shape[0] == 100


class TestDPA:
    
    def test_dpa_creation(self):
        dpa = DPA(target_bit=3)
        assert dpa.attack_type == AttackType.DPA
        assert dpa.target_bit == 3
    
    def test_dpa_target_bit_range(self):
        dpa = DPA()
        with pytest.raises(ValueError):
            dpa.target_bit = 8
        with pytest.raises(ValueError):
            dpa.target_bit = -1
    
    def test_dpa_attack_byte(self):
        traceset = create_test_traceset(num_traces=200, num_samples=50)
        dpa = DPA(key_byte_indices=[0], target_bit=0)
        
        samples = traceset.get_samples_matrix()
        plaintexts = traceset.get_plaintexts()
        
        key_byte, confidence, differential = dpa.attack_byte(samples, plaintexts, 0)
        
        assert 0 <= key_byte < 256
        assert differential.shape == (50,)
    
    def test_dpa_run(self):
        traceset = create_test_traceset(num_traces=500, num_samples=50)
        dpa = DPA(key_byte_indices=[0])
        result = dpa.run(traceset)
        
        assert isinstance(result, AttackResult)
        assert result.attack_type == AttackType.DPA


class TestMultiBitDPA:
    
    def test_multibit_dpa_creation(self):
        dpa = MultiBitDPA(target_bits=[0, 1, 2, 3])
        assert len(dpa.target_bits) == 4
    
    def test_multibit_dpa_all_bits(self):
        dpa = MultiBitDPA()
        assert len(dpa.target_bits) == 8
    
    def test_multibit_dpa_run(self):
        traceset = create_test_traceset(num_traces=300, num_samples=50)
        dpa = MultiBitDPA(key_byte_indices=[0], target_bits=[0, 1])
        result = dpa.run(traceset)
        
        assert isinstance(result, AttackResult)
        assert result.attack_type == AttackType.DPA


class TestIncrementalDPA:
    
    def test_incremental_dpa_update(self):
        dpa = IncrementalDPA(key_byte_indices=[0])
        
        np.random.seed(42)
        for _ in range(5):
            traces = np.random.randn(20, 30).astype(np.float32)
            plaintexts = np.random.randint(0, 256, (20, 16), dtype=np.uint8)
            dpa.update(traces, plaintexts)
        
        assert dpa.trace_count == 100
    
    def test_incremental_dpa_finalize(self):
        dpa = IncrementalDPA(key_byte_indices=[0])
        
        np.random.seed(42)
        for _ in range(10):
            traces = np.random.randn(20, 30).astype(np.float32)
            plaintexts = np.random.randint(0, 256, (20, 16), dtype=np.uint8)
            dpa.update(traces, plaintexts)
        
        result = dpa.finalize()
        assert isinstance(result, AttackResult)
        assert result.metrics["target_bit"] == 0


class TestTemplateAttack:
    
    def test_template_creation(self):
        template = TemplateAttack(num_poi=5)
        assert template.num_poi == 5
        assert template.attack_type == AttackType.TEMPLATE
    
    def test_template_poi_selection_snr(self):
        template = TemplateAttack(poi_selection="snr", num_poi=3)
        assert template.poi_selection == "snr"
    
    def test_template_build(self):
        np.random.seed(42)
        
        traces_list = []
        labels = []
        for label in range(4):
            for _ in range(25):
                samples = np.random.randn(30).astype(np.float32)
                samples[10 + label] += label * 2.0
                traces_list.append(Trace(
                    samples=samples,
                    plaintext=np.array([label] * 16, dtype=np.uint8),
                    key=np.zeros(16, dtype=np.uint8),
                ))
                labels.append(label)
        
        traceset = TraceSet(traces=traces_list)
        labels = np.array(labels)
        
        template = TemplateAttack(num_poi=3)
        templates = template.build_templates(traceset, labels)
        
        assert len(templates) == 4
        assert template.poi_indices is not None
    
    def test_template_attack_trace(self):
        np.random.seed(42)
        
        traces_list = []
        labels = []
        for label in range(4):
            for _ in range(30):
                samples = np.random.randn(30).astype(np.float32)
                samples[10 + label] += label * 3.0
                traces_list.append(Trace(
                    samples=samples,
                    plaintext=np.array([label] * 16, dtype=np.uint8),
                    key=np.zeros(16, dtype=np.uint8),
                ))
                labels.append(label)
        
        traceset = TraceSet(traces=traces_list)
        labels = np.array(labels)
        
        template = TemplateAttack(num_poi=4)
        template.build_templates(traceset, labels)
        
        test_trace = np.random.randn(30).astype(np.float32)
        test_trace[12] += 6.0
        
        pred, probs = template.attack_trace(test_trace)
        assert 0 <= pred < 256
        assert len(probs) == 256


class TestTemplateProfile:
    
    def test_profile_creation(self):
        means = np.array([1.0, 2.0, 3.0])
        cov = np.eye(3)
        profile = TemplateProfile(means=means, covariance=cov)
        
        assert profile.inv_covariance is not None
        np.testing.assert_array_almost_equal(profile.inv_covariance, np.eye(3))


class TestNTTCoefficientAttack:
    
    def test_ntt_attack_creation(self):
        attack = NTTCoefficientAttack(q=3329, n=256)
        assert attack.q == 3329
        assert attack.n == 256
        assert attack.attack_type == AttackType.LATTICE
    
    def test_ntt_attack_run(self):
        np.random.seed(42)
        
        traces_list = []
        for _ in range(50):
            samples = np.random.randn(40).astype(np.float32)
            metadata = TraceMetadata()
            traces_list.append(Trace(samples=samples, metadata=metadata))
        
        traceset = TraceSet(traces=traces_list)
        
        attack = NTTCoefficientAttack(
            q=3329,
            target_coefficients=[0, 1, 2],
            coefficient_range=(0, 256),
        )
        result = attack.run(traceset)
        
        assert isinstance(result, AttackResult)
        assert result.attack_type == AttackType.LATTICE


class TestNTTButterflyAttack:
    
    def test_butterfly_attack_creation(self):
        attack = NTTButterflyAttack(q=3329, n=256)
        assert attack.attack_type == AttackType.LATTICE
    
    def test_butterfly_attack_run(self):
        np.random.seed(42)
        
        traces_list = []
        for _ in range(30):
            samples = np.random.randn(40).astype(np.float32)
            metadata = TraceMetadata()
            traces_list.append(Trace(samples=samples, metadata=metadata))
        
        traceset = TraceSet(traces=traces_list)
        
        attack = NTTButterflyAttack(q=3329, n=256)
        result = attack.run(traceset, target_layers=[0], butterflies_per_layer=2)
        
        assert isinstance(result, AttackResult)
        assert "layer_results" in result.metrics


class TestButterflyLeakageModel:
    
    def test_butterfly_model_creation(self):
        model = ButterflyLeakageModel(q=3329)
        assert model.q == 3329
    
    def test_butterfly_output_hw(self):
        model = ButterflyLeakageModel(q=3329, word_size=16)
        hw_a, hw_b = model.butterfly_output_hw(100, 200, 17)
        assert isinstance(hw_a, (int, float, np.integer, np.floating))
        assert isinstance(hw_b, (int, float, np.integer, np.floating))
    
    def test_butterfly_intermediate_hw(self):
        model = ButterflyLeakageModel(q=3329)
        hw = model.butterfly_intermediate_hw(200, 17)
        assert hw >= 0


class TestSecretKeyRecoveryAttack:
    
    def test_secret_recovery_creation(self):
        attack = SecretKeyRecoveryAttack(q=3329, n=256, secret_bound=2)
        assert attack.attack_type == AttackType.LATTICE
    
    def test_secret_recovery_run(self):
        np.random.seed(42)
        
        traces_list = []
        for _ in range(50):
            samples = np.random.randn(40).astype(np.float32)
            metadata = TraceMetadata()
            traces_list.append(Trace(samples=samples, metadata=metadata))
        
        traceset = TraceSet(traces=traces_list)
        
        attack = SecretKeyRecoveryAttack(
            q=3329,
            n=256,
            secret_bound=2,
            target_coefficients=list(range(5)),
        )
        result = attack.run(traceset)
        
        assert isinstance(result, AttackResult)
        assert result.recovered_key is not None


class TestBeliefPropagationAttack:
    
    def test_bp_attack_creation(self):
        attack = BeliefPropagationAttack(q=3329, num_iterations=5)
        assert attack.attack_type == AttackType.LATTICE
    
    def test_bp_attack_run(self):
        np.random.seed(42)
        
        traces_list = []
        for _ in range(30):
            samples = np.random.randn(40).astype(np.float32)
            metadata = TraceMetadata()
            traces_list.append(Trace(samples=samples, metadata=metadata))
        
        traceset = TraceSet(traces=traces_list)
        
        attack = BeliefPropagationAttack(q=3329, num_iterations=3)
        result = attack.run(traceset, num_coefficients=10)
        
        assert isinstance(result, AttackResult)
        assert "num_iterations" in result.metrics


class TestLatticeLeakageAssessment:
    
    def test_lattice_assessment_creation(self):
        assessment = LatticeLeakageAssessment(
            operation=LatticeOperation.NTT
        )
        assert assessment.operation == LatticeOperation.NTT
    
    def test_lattice_assessment_assess(self):
        np.random.seed(42)
        group1 = np.random.randn(50, 30).astype(np.float32)
        group2 = np.random.randn(50, 30).astype(np.float32) + 2.0
        
        assessment = LatticeLeakageAssessment()
        result = assessment.assess(group1, group2)
        
        assert isinstance(result, LeakageAssessmentResult)
        assert result.leakage_detected


class TestAttackResult:
    
    def test_result_to_dict(self):
        result = AttackResult(
            attack_type=AttackType.CPA,
            success=True,
            recovered_key=np.array([1, 2, 3], dtype=np.uint8),
            confidence=0.95,
            num_traces_used=100,
        )
        
        d = result.to_dict()
        assert d["attack_type"] == "cpa"
        assert d["success"] is True
        assert d["confidence"] == 0.95
        assert d["recovered_key"] == [1, 2, 3]


class TestSBOX:
    
    def test_sbox_size(self):
        assert len(SBOX) == 256
    
    def test_sbox_bijective(self):
        assert len(set(SBOX)) == 256
    
    def test_sbox_known_values(self):
        assert SBOX[0x00] == 0x63
        assert SBOX[0x01] == 0x7c
        assert SBOX[0xFF] == 0x16
