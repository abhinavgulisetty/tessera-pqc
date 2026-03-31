import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

from tessera.countermeasures import (
    MaskingScheme,
    MaskedValue,
    BooleanMasking,
    ArithmeticMasking,
    MaskConverter,
    HigherOrderMasking,
    MaskedNTTConfig,
    MaskedNTT,
    MaskedPolynomialOps,
    ShuffleStrategy,
    ShuffleConfig,
    SecurePermutation,
    ShuffledOperation,
    ShuffledNTT,
    ShuffledPolynomialMul,
    RandomDelayGenerator,
    DummyOperationGenerator,
    ShufflingCountermeasure,
    NoiseType,
    HidingTechnique,
    HidingConfig,
    NoiseInjector,
    AmplitudeRandomizer,
    TemporalJitter,
    PowerBalancer,
    DualRailLogic,
    HidingCountermeasure,
    OperationHider,
    TraceHider,
    CombinedHidingConfig,
    CombinedHiding,
)


class TestBooleanMasking:

    def test_mask_unmask_roundtrip(self):
        masking = BooleanMasking(bit_width=16)
        value = np.array([0x1234, 0x5678, 0xABCD], dtype=np.uint16)
        
        masked = masking.mask(value, num_shares=2)
        assert masked.num_shares == 2
        assert masked.scheme == MaskingScheme.BOOLEAN
        
        unmasked = masking.unmask(masked)
        assert_array_equal(unmasked, value)

    def test_mask_with_multiple_shares(self):
        masking = BooleanMasking(bit_width=32)
        value = np.array([0x12345678], dtype=np.uint32)
        
        for num_shares in [2, 3, 4, 5]:
            masked = masking.mask(value, num_shares=num_shares)
            assert masked.num_shares == num_shares
            assert masked.order == num_shares - 1
            
            unmasked = masking.unmask(masked)
            assert_array_equal(unmasked, value)

    def test_refresh_preserves_value(self):
        masking = BooleanMasking(bit_width=16)
        value = np.array([0xFFFF, 0x0000, 0x5555], dtype=np.uint16)
        
        masked = masking.mask(value, num_shares=2)
        original_shares = masked.shares.copy()
        
        refreshed = masking.refresh(masked)
        
        assert_array_equal(masking.unmask(refreshed), value)
        assert not np.array_equal(refreshed.shares, original_shares)

    def test_secure_xor(self):
        masking = BooleanMasking(bit_width=8)
        a = np.array([0x12, 0x34], dtype=np.uint8)
        b = np.array([0x56, 0x78], dtype=np.uint8)
        
        masked_a = masking.mask(a, num_shares=2)
        masked_b = masking.mask(b, num_shares=2)
        
        masked_result = masking.secure_xor(masked_a, masked_b)
        result = masking.unmask(masked_result)
        
        expected = a ^ b
        assert_array_equal(result, expected)

    def test_secure_and(self):
        masking = BooleanMasking(bit_width=8)
        a = np.array([0xFF, 0x0F], dtype=np.uint8)
        b = np.array([0xF0, 0x33], dtype=np.uint8)
        
        masked_a = masking.mask(a, num_shares=2)
        masked_b = masking.mask(b, num_shares=2)
        
        masked_result = masking.secure_and(masked_a, masked_b)
        result = masking.unmask(masked_result)
        
        expected = a & b
        assert_array_equal(result, expected)

    def test_secure_not(self):
        masking = BooleanMasking(bit_width=8)
        a = np.array([0x55, 0xAA], dtype=np.uint8)
        
        masked_a = masking.mask(a, num_shares=2)
        masked_result = masking.secure_not(masked_a)
        result = masking.unmask(masked_result)
        
        expected = ~a & 0xFF
        assert_array_equal(result & 0xFF, expected)

    def test_invalid_share_count(self):
        masking = BooleanMasking()
        value = np.array([1, 2, 3])
        
        with pytest.raises(ValueError):
            masking.mask(value, num_shares=1)


class TestArithmeticMasking:

    def test_mask_unmask_roundtrip(self):
        q = 3329
        masking = ArithmeticMasking(modulus=q)
        value = np.array([100, 200, 3000], dtype=np.int64)
        
        masked = masking.mask(value, num_shares=2)
        assert masked.num_shares == 2
        assert masked.scheme == MaskingScheme.ARITHMETIC
        assert masked.modulus == q
        
        unmasked = masking.unmask(masked)
        assert_array_equal(unmasked, value % q)

    def test_mask_with_multiple_shares(self):
        q = 8380417
        masking = ArithmeticMasking(modulus=q)
        value = np.array([12345, 67890], dtype=np.int64)
        
        for num_shares in [2, 3, 4]:
            masked = masking.mask(value, num_shares=num_shares)
            assert masked.num_shares == num_shares
            
            unmasked = masking.unmask(masked)
            assert_array_equal(unmasked, value % q)

    def test_refresh_preserves_value(self):
        q = 3329
        masking = ArithmeticMasking(modulus=q)
        value = np.array([1000, 2000, 3000], dtype=np.int64)
        
        masked = masking.mask(value, num_shares=2)
        original_shares = masked.shares.copy()
        
        refreshed = masking.refresh(masked)
        
        assert_array_equal(masking.unmask(refreshed), value % q)
        assert not np.array_equal(refreshed.shares, original_shares)

    def test_secure_add(self):
        q = 3329
        masking = ArithmeticMasking(modulus=q)
        a = np.array([100, 200], dtype=np.int64)
        b = np.array([300, 400], dtype=np.int64)
        
        masked_a = masking.mask(a, num_shares=2)
        masked_b = masking.mask(b, num_shares=2)
        
        masked_result = masking.secure_add(masked_a, masked_b)
        result = masking.unmask(masked_result)
        
        expected = (a + b) % q
        assert_array_equal(result, expected)

    def test_secure_sub(self):
        q = 3329
        masking = ArithmeticMasking(modulus=q)
        a = np.array([500, 600], dtype=np.int64)
        b = np.array([100, 200], dtype=np.int64)
        
        masked_a = masking.mask(a, num_shares=2)
        masked_b = masking.mask(b, num_shares=2)
        
        masked_result = masking.secure_sub(masked_a, masked_b)
        result = masking.unmask(masked_result)
        
        expected = (a - b) % q
        assert_array_equal(result, expected)

    def test_secure_scalar_mul(self):
        q = 3329
        masking = ArithmeticMasking(modulus=q)
        a = np.array([100, 200], dtype=np.int64)
        scalar = 17
        
        masked_a = masking.mask(a, num_shares=2)
        masked_result = masking.secure_scalar_mul(masked_a, scalar)
        result = masking.unmask(masked_result)
        
        expected = (a * scalar) % q
        assert_array_equal(result, expected)


class TestMaskConverter:

    def test_bool_to_arith_conversion(self):
        q = 256
        converter = MaskConverter(modulus=q, bit_width=8)
        value = np.array([0x12, 0x34], dtype=np.uint8)
        
        bool_masked = converter.boolean.mask(value, num_shares=2)
        arith_masked = converter.bool_to_arith(bool_masked)
        
        assert arith_masked.scheme == MaskingScheme.ARITHMETIC
        assert arith_masked.modulus == q

    def test_arith_to_bool_conversion(self):
        q = 256
        converter = MaskConverter(modulus=q, bit_width=8)
        value = np.array([100, 200], dtype=np.int64)
        
        arith_masked = converter.arithmetic.mask(value, num_shares=2)
        bool_masked = converter.arith_to_bool(arith_masked)
        
        assert bool_masked.scheme == MaskingScheme.BOOLEAN


class TestHigherOrderMasking:

    def test_boolean_higher_order(self):
        hom = HigherOrderMasking(order=3, bit_width=16)
        value = np.array([0x1234, 0x5678], dtype=np.uint16)
        
        masked = hom.mask_boolean(value)
        assert masked.num_shares == 4

    def test_arithmetic_higher_order(self):
        q = 3329
        hom = HigherOrderMasking(order=2, modulus=q)
        value = np.array([100, 200], dtype=np.int64)
        
        masked = hom.mask_arithmetic(value)
        assert masked.num_shares == 3
        assert masked.modulus == q

    def test_isw_and(self):
        hom = HigherOrderMasking(order=2, bit_width=8)
        a = np.array([0xFF, 0x0F], dtype=np.uint8)
        b = np.array([0xF0, 0x33], dtype=np.uint8)
        
        masked_a = hom.mask_boolean(a)
        masked_b = hom.mask_boolean(b)
        
        masked_result = hom.isw_and(masked_a, masked_b)
        result = masked_result.unmask()
        
        expected = a & b
        assert_array_equal(result, expected)

    def test_isw_refresh(self):
        hom = HigherOrderMasking(order=2, bit_width=16)
        value = np.array([0x1234], dtype=np.uint16)
        
        masked = hom.mask_boolean(value)
        original_shares = masked.shares.copy()
        
        refreshed = hom.isw_refresh(masked)
        
        assert_array_equal(refreshed.unmask(), value)
        assert not np.array_equal(refreshed.shares, original_shares)


class TestMaskedNTT:

    def test_masked_ntt_creates_output(self):
        config = MaskedNTTConfig(n=256, q=3329, num_shares=2)
        ntt = MaskedNTT(config)
        
        poly = np.array([i % 3329 for i in range(256)], dtype=np.int64)
        masked_poly = ntt.masking.mask(poly, num_shares=2)
        
        masked_ntt_result = ntt.masked_ntt(masked_poly)
        
        assert masked_ntt_result.num_shares == 2
        assert masked_ntt_result.scheme == MaskingScheme.ARITHMETIC
        assert len(masked_ntt_result.shares[0]) == 256

    def test_masked_pointwise_mul_structure(self):
        config = MaskedNTTConfig(n=16, q=3329, num_shares=2)
        ntt = MaskedNTT(config)
        
        a = np.array([i for i in range(16)], dtype=np.int64)
        b = np.array([i + 1 for i in range(16)], dtype=np.int64)
        
        masked_a = ntt.masking.mask(a, num_shares=2)
        masked_b = ntt.masking.mask(b, num_shares=2)
        
        masked_result = ntt.masked_pointwise_mul(masked_a, masked_b)
        
        assert masked_result.num_shares == 2
        assert masked_result.scheme == MaskingScheme.ARITHMETIC
        assert len(masked_result.shares[0]) == 16


class TestMaskedPolynomialOps:

    def test_polynomial_add(self):
        ops = MaskedPolynomialOps(n=16, q=3329, num_shares=2)
        
        a = np.array([i for i in range(16)], dtype=np.int64)
        b = np.array([i + 100 for i in range(16)], dtype=np.int64)
        
        masked_a = ops.mask_polynomial(a)
        masked_b = ops.mask_polynomial(b)
        
        masked_result = ops.add(masked_a, masked_b)
        result = ops.unmask_polynomial(masked_result)
        
        expected = (a + b) % 3329
        assert_array_equal(result, expected)

    def test_polynomial_scalar_mul(self):
        ops = MaskedPolynomialOps(n=16, q=3329, num_shares=2)
        
        a = np.array([i for i in range(16)], dtype=np.int64)
        scalar = 17
        
        masked_a = ops.mask_polynomial(a)
        masked_result = ops.scalar_mul(masked_a, scalar)
        result = ops.unmask_polynomial(masked_result)
        
        expected = (a * scalar) % 3329
        assert_array_equal(result, expected)


class TestSecurePermutation:

    def test_fisher_yates_is_permutation(self):
        n = 100
        perm = SecurePermutation.fisher_yates(n)
        
        assert len(perm) == n
        assert set(perm) == set(range(n))

    def test_random_pairs_is_permutation(self):
        n = 50
        perm = SecurePermutation.random_pairs(n)
        
        assert len(perm) == n
        assert set(perm) == set(range(n))

    def test_random_cycles_is_permutation(self):
        n = 64
        perm = SecurePermutation.random_cycles(n)
        
        assert len(perm) == n
        assert set(perm) == set(range(n))

    def test_inverse_permutation(self):
        n = 32
        perm = SecurePermutation.fisher_yates(n)
        inv = SecurePermutation.inverse_permutation(perm)
        
        assert_array_equal(perm[inv], np.arange(n))
        assert_array_equal(inv[perm], np.arange(n))

    def test_generate_with_strategy(self):
        n = 20
        
        for strategy in ShuffleStrategy:
            perm = SecurePermutation.generate(n, strategy)
            assert len(perm) == n
            assert set(perm) == set(range(n))


class TestShuffledOperation:

    def test_shuffle_loop(self):
        shuffler = ShuffledOperation()
        n = 10
        results = []
        
        output = shuffler.shuffle_loop(n, lambda i: i * 2)
        
        assert len(output) == n
        for i in range(n):
            assert output[i] == i * 2

    def test_shuffle_array_operation(self):
        shuffler = ShuffledOperation()
        arr = np.array([1, 2, 3, 4, 5])
        
        result = shuffler.shuffle_array_operation(arr, lambda x: x ** 2)
        
        expected = arr ** 2
        assert_array_equal(result, expected)

    def test_shuffle_pairwise_operation(self):
        shuffler = ShuffledOperation()
        a = np.array([1, 2, 3, 4])
        b = np.array([5, 6, 7, 8])
        
        result = shuffler.shuffle_pairwise_operation(a, b, lambda x, y: x + y)
        
        expected = a + b
        assert_array_equal(result, expected)


class TestShuffledNTT:

    def test_ntt_shuffled_produces_output(self):
        n, q = 16, 3329
        ntt = ShuffledNTT(n, q)
        
        coeffs = np.array([i for i in range(n)], dtype=np.int64)
        
        ntt_result = ntt.ntt_shuffled(coeffs)
        
        assert len(ntt_result) == n
        assert ntt_result.dtype == np.int64

    def test_ntt_with_config(self):
        config = ShuffleConfig(
            strategy=ShuffleStrategy.RANDOM_PAIRS,
            random_delays=False
        )
        ntt = ShuffledNTT(16, 3329, config)
        
        coeffs = np.array([i % 100 for i in range(16)], dtype=np.int64)
        ntt_result = ntt.ntt_shuffled(coeffs)
        
        assert len(ntt_result) == 16


class TestShuffledPolynomialMul:

    def test_pointwise_mul_shuffled(self):
        mul = ShuffledPolynomialMul(16, 3329)
        
        a = np.array([i for i in range(16)], dtype=np.int64)
        b = np.array([i + 1 for i in range(16)], dtype=np.int64)
        
        result = mul.pointwise_mul_shuffled(a, b)
        expected = (a * b) % 3329
        
        assert_array_equal(result, expected)


class TestRandomDelayGenerator:

    def test_delay_creation(self):
        gen = RandomDelayGenerator(min_delay_ns=0, max_delay_ns=100)
        gen.delay()

    def test_random_nop_loop(self):
        gen = RandomDelayGenerator()
        result = gen.random_nop_loop(max_iterations=10)
        assert isinstance(result, int)


class TestDummyOperationGenerator:

    def test_generate_dummy_arithmetic(self):
        gen = DummyOperationGenerator()
        a, b = gen.generate_dummy_arithmetic(10)
        
        assert len(a) == 10
        assert len(b) == 10

    def test_execute_dummy_operations(self):
        gen = DummyOperationGenerator(operation_type="arithmetic")
        gen.execute_dummy_operations(count=5, size=10)
        
        gen = DummyOperationGenerator(operation_type="bitwise")
        gen.execute_dummy_operations(count=5, size=10)


class TestShufflingCountermeasure:

    def test_create_config(self):
        cm = ShufflingCountermeasure()
        config = cm.create_config()
        
        assert config.strategy == ShuffleStrategy.FISHER_YATES
        assert config.random_delays == True
        assert config.dummy_operations == True


class TestNoiseInjector:

    def test_gaussian_noise(self):
        injector = NoiseInjector(NoiseType.GAUSSIAN, seed=42)
        noise = injector.generate_noise((100,), amplitude=1.0)
        
        assert len(noise) == 100
        assert np.abs(np.mean(noise)) < 0.5

    def test_uniform_noise(self):
        injector = NoiseInjector(NoiseType.UNIFORM, seed=42)
        noise = injector.generate_noise((100,), amplitude=1.0)
        
        assert len(noise) == 100
        assert np.all(noise >= -1.0)
        assert np.all(noise <= 1.0)

    def test_laplacian_noise(self):
        injector = NoiseInjector(NoiseType.LAPLACIAN, seed=42)
        noise = injector.generate_noise((100,), amplitude=1.0)
        
        assert len(noise) == 100

    def test_shot_noise(self):
        injector = NoiseInjector(NoiseType.SHOT, seed=42)
        noise = injector.generate_noise((100,), amplitude=5.0)
        
        assert len(noise) == 100

    def test_inject_noise_snr(self):
        injector = NoiseInjector(NoiseType.GAUSSIAN, seed=42)
        signal = np.sin(np.linspace(0, 2 * np.pi, 100))
        
        noisy = injector.inject_noise(signal, snr_db=20)
        
        assert len(noisy) == len(signal)
        assert not np.array_equal(noisy, signal)

    def test_inject_additive_noise(self):
        injector = NoiseInjector(NoiseType.GAUSSIAN, seed=42)
        signal = np.ones(100)
        
        noisy = injector.inject_additive_noise(signal, amplitude=0.1)
        
        assert len(noisy) == 100
        assert np.std(noisy) > 0

    def test_inject_multiplicative_noise(self):
        injector = NoiseInjector(NoiseType.GAUSSIAN, seed=42)
        signal = np.ones(100) * 10
        
        noisy = injector.inject_multiplicative_noise(signal, amplitude=0.1)
        
        assert len(noisy) == 100
        assert np.abs(np.mean(noisy) - 10) < 2


class TestAmplitudeRandomizer:

    def test_randomize_amplitude(self):
        randomizer = AmplitudeRandomizer(seed=42)
        signal = np.ones(100)
        
        randomized = randomizer.randomize_amplitude(signal, 0.5, 1.5)
        
        scale = randomized[0]
        assert 0.5 <= scale <= 1.5

    def test_randomize_dc_offset(self):
        randomizer = AmplitudeRandomizer(seed=42)
        signal = np.zeros(100)
        
        randomized = randomizer.randomize_dc_offset(signal, max_offset=1.0)
        
        offset = randomized[0]
        assert -1.0 <= offset <= 1.0

    def test_randomize_per_sample(self):
        randomizer = AmplitudeRandomizer(seed=42)
        signal = np.ones(100)
        
        randomized = randomizer.randomize_per_sample(signal, amplitude=0.1)
        
        assert np.std(randomized) > 0


class TestTemporalJitter:

    def test_random_shift(self):
        jitter = TemporalJitter(max_jitter_samples=5, seed=42)
        signal = np.arange(100)
        
        shifted = jitter.apply_random_shift(signal)
        
        assert len(shifted) == len(signal)

    def test_random_stretch(self):
        jitter = TemporalJitter(seed=42)
        signal = np.sin(np.linspace(0, 2 * np.pi, 100))
        
        stretched = jitter.apply_random_stretch(signal, 0.9, 1.1)
        
        assert len(stretched) == len(signal)

    def test_local_jitter(self):
        jitter = TemporalJitter(max_jitter_samples=3, seed=42)
        signal = np.arange(200)
        
        jittered = jitter.apply_local_jitter(signal, segment_size=50)
        
        assert len(jittered) == len(signal)


class TestPowerBalancer:

    def test_balance_hamming_weight(self):
        balancer = PowerBalancer()
        
        value, complement = balancer.balance_hamming_weight(0x55, bit_width=8)
        
        assert value == 0x55
        assert complement == 0xAA

    def test_balance_array(self):
        balancer = PowerBalancer()
        values = np.array([0x0F, 0xF0], dtype=np.uint8)
        
        balanced, complements = balancer.balance_array(values, bit_width=8)
        
        assert_array_equal(balanced, values)
        assert_array_equal(complements, np.array([0xF0, 0x0F], dtype=np.uint8))

    def test_balanced_encoding_decoding(self):
        balancer = PowerBalancer()
        
        for value in [0, 1, 127, 255]:
            encoded = balancer.create_balanced_encoding(value, bit_width=8)
            decoded = balancer.decode_balanced(encoded)
            
            assert decoded == value
            assert len(encoded) == 16


class TestDualRailLogic:

    def test_encode_decode_roundtrip(self):
        dual = DualRailLogic(bit_width=8)
        values = np.array([0x12, 0x34, 0xFF], dtype=np.uint8)
        
        true_rail, false_rail = dual.encode(values)
        decoded = dual.decode(true_rail, false_rail)
        
        assert_array_equal(decoded, values)

    def test_dual_and(self):
        dual = DualRailLogic(bit_width=8)
        a = np.array([0xFF, 0x0F], dtype=np.uint8)
        b = np.array([0xF0, 0x33], dtype=np.uint8)
        
        a_enc = dual.encode(a)
        b_enc = dual.encode(b)
        
        result_enc = dual.dual_and(a_enc, b_enc)
        result = dual.decode(*result_enc)
        
        expected = a & b
        assert_array_equal(result, expected)

    def test_dual_or(self):
        dual = DualRailLogic(bit_width=8)
        a = np.array([0x0F, 0x33], dtype=np.uint8)
        b = np.array([0xF0, 0xCC], dtype=np.uint8)
        
        a_enc = dual.encode(a)
        b_enc = dual.encode(b)
        
        result_enc = dual.dual_or(a_enc, b_enc)
        result = dual.decode(*result_enc)
        
        expected = a | b
        assert_array_equal(result, expected)

    def test_dual_xor(self):
        dual = DualRailLogic(bit_width=8)
        a = np.array([0x55, 0xAA], dtype=np.uint8)
        b = np.array([0xFF, 0x00], dtype=np.uint8)
        
        a_enc = dual.encode(a)
        b_enc = dual.encode(b)
        
        result_enc = dual.dual_xor(a_enc, b_enc)
        result = dual.decode(*result_enc)
        
        expected = a ^ b
        assert_array_equal(result, expected)

    def test_dual_not(self):
        dual = DualRailLogic(bit_width=8)
        a = np.array([0x55, 0xAA], dtype=np.uint8)
        
        a_enc = dual.encode(a)
        result_enc = dual.dual_not(a_enc)
        
        assert_array_equal(result_enc[0], a_enc[1])
        assert_array_equal(result_enc[1], a_enc[0])


class TestHidingCountermeasure:

    def test_noise_injection(self):
        config = HidingConfig(
            technique=HidingTechnique.NOISE_INJECTION,
            noise_amplitude=0.1
        )
        hiding = HidingCountermeasure(config)
        
        signal = np.sin(np.linspace(0, 2 * np.pi, 100))
        hidden = hiding.apply(signal)
        
        assert len(hidden) == len(signal)
        assert not np.array_equal(hidden, signal)

    def test_amplitude_randomization(self):
        config = HidingConfig(technique=HidingTechnique.AMPLITUDE_RANDOMIZATION)
        hiding = HidingCountermeasure(config)
        
        signal = np.ones(100)
        hidden = hiding.apply(signal)
        
        assert len(hidden) == len(signal)

    def test_temporal_jitter(self):
        config = HidingConfig(
            technique=HidingTechnique.TEMPORAL_JITTER,
            temporal_jitter_samples=5
        )
        hiding = HidingCountermeasure(config)
        
        signal = np.arange(100, dtype=float)
        hidden = hiding.apply(signal)
        
        assert len(hidden) == len(signal)

    def test_apply_all(self):
        config = HidingConfig(noise_amplitude=0.05)
        hiding = HidingCountermeasure(config)
        
        signal = np.sin(np.linspace(0, 2 * np.pi, 100))
        hidden = hiding.apply_all(signal)
        
        assert len(hidden) == len(signal)
        assert not np.array_equal(hidden, signal)


class TestOperationHider:

    def test_hide_memory_access(self):
        hider = OperationHider()
        array = np.arange(100)
        
        result = hider.hide_memory_access(array, index=50, num_dummy_accesses=4)
        
        assert result == 50

    def test_hide_conditional(self):
        hider = OperationHider()
        true_val = np.array([1, 2, 3])
        false_val = np.array([4, 5, 6])
        
        result_true = hider.hide_conditional(True, true_val, false_val)
        result_false = hider.hide_conditional(False, true_val, false_val)
        
        assert_array_equal(result_true, true_val)
        assert_array_equal(result_false, false_val)


class TestTraceHider:

    def test_hide_trace(self):
        hider = TraceHider()
        trace = np.sin(np.linspace(0, 2 * np.pi, 100))
        
        hidden = hider.hide_trace(trace)
        
        assert len(hidden) == len(trace)

    def test_hide_traces(self):
        hider = TraceHider()
        traces = np.random.randn(10, 100)
        
        hidden = hider.hide_traces(traces)
        
        assert hidden.shape == traces.shape

    def test_add_dummy_traces(self):
        hider = TraceHider()
        traces = np.random.randn(10, 100)
        
        combined, real_indices = hider.add_dummy_traces(traces, num_dummy=5)
        
        assert combined.shape == (15, 100)
        assert len(real_indices) == 10


class TestCombinedHiding:

    def test_apply(self):
        config = CombinedHidingConfig(
            noise_amplitude=0.1,
            amplitude_randomization=True,
            temporal_jitter=True
        )
        hiding = CombinedHiding(config)
        
        signal = np.sin(np.linspace(0, 2 * np.pi, 100))
        hidden = hiding.apply(signal)
        
        assert len(hidden) == len(signal)

    def test_apply_batch(self):
        config = CombinedHidingConfig(noise_amplitude=0.05)
        hiding = CombinedHiding(config)
        
        signals = np.random.randn(10, 100)
        hidden = hiding.apply_batch(signals)
        
        assert hidden.shape == signals.shape

    def test_config_defaults(self):
        config = CombinedHidingConfig()
        
        assert config.noise_amplitude == 0.1
        assert config.noise_type == NoiseType.GAUSSIAN
        assert config.amplitude_randomization == True
        assert config.temporal_jitter == True
        assert config.use_dual_rail == False
