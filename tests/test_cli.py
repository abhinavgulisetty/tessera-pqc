import pytest
import subprocess
import sys
import json
import tempfile
from pathlib import Path
import numpy as np


class TestCLIHelp:
    
    def test_main_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "tessera.cli", "--help"],
            capture_output=True, text=True
        )
        assert result.returncode == 0
        assert "Tessera-PQC" in result.stdout
        assert "kyber" in result.stdout
        assert "dilithium" in result.stdout
        assert "attack" in result.stdout
        assert "benchmark" in result.stdout
    
    def test_version(self):
        result = subprocess.run(
            [sys.executable, "-m", "tessera.cli", "--version"],
            capture_output=True, text=True
        )
        assert result.returncode == 0
        assert "0.2.0" in result.stdout
    
    def test_kyber_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "tessera.cli", "kyber", "--help"],
            capture_output=True, text=True
        )
        assert result.returncode == 0
        assert "--variant" in result.stdout
        assert "512" in result.stdout
        assert "768" in result.stdout
        assert "1024" in result.stdout
    
    def test_dilithium_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "tessera.cli", "dilithium", "--help"],
            capture_output=True, text=True
        )
        assert result.returncode == 0
        assert "--variant" in result.stdout
        assert "--message" in result.stdout
    
    def test_attack_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "tessera.cli", "attack", "--help"],
            capture_output=True, text=True
        )
        assert result.returncode == 0
        assert "cpa" in result.stdout
        assert "dpa" in result.stdout
        assert "tvla" in result.stdout
        assert "template" in result.stdout
    
    def test_benchmark_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "tessera.cli", "benchmark", "--help"],
            capture_output=True, text=True
        )
        assert result.returncode == 0
        assert "--ntt" in result.stdout
        assert "--kyber" in result.stdout
        assert "--dilithium" in result.stdout


class TestCLIKyber:
    
    def test_kyber_512(self):
        result = subprocess.run(
            [sys.executable, "-m", "tessera.cli", "kyber", "--variant", "512"],
            capture_output=True, text=True, timeout=60
        )
        assert result.returncode == 0
        assert "SUCCESS" in result.stdout
        assert "shared secrets match" in result.stdout
    
    def test_kyber_768(self):
        result = subprocess.run(
            [sys.executable, "-m", "tessera.cli", "kyber", "--variant", "768"],
            capture_output=True, text=True, timeout=60
        )
        assert result.returncode == 0
        assert "SUCCESS" in result.stdout
    
    def test_kyber_1024(self):
        result = subprocess.run(
            [sys.executable, "-m", "tessera.cli", "kyber", "--variant", "1024"],
            capture_output=True, text=True, timeout=60
        )
        assert result.returncode == 0
        assert "SUCCESS" in result.stdout


class TestCLIDilithium:
    
    def test_dilithium_2(self):
        result = subprocess.run(
            [sys.executable, "-m", "tessera.cli", "dilithium", "--variant", "2"],
            capture_output=True, text=True, timeout=120
        )
        assert result.returncode == 0
        assert "SUCCESS" in result.stdout
        assert "signature verified" in result.stdout
    
    def test_dilithium_with_message(self):
        result = subprocess.run(
            [sys.executable, "-m", "tessera.cli", "dilithium", "--variant", "2", 
             "--message", "Hello World"],
            capture_output=True, text=True, timeout=120
        )
        assert result.returncode == 0
        assert "SUCCESS" in result.stdout


class TestCLIAttack:
    
    def test_cpa_attack(self):
        result = subprocess.run(
            [sys.executable, "-m", "tessera.cli", "attack", 
             "--type", "cpa", "--traces", "200", "--seed", "42"],
            capture_output=True, text=True, timeout=60
        )
        assert result.returncode == 0
        assert "CPA" in result.stdout
        assert "Recovered key" in result.stdout
    
    def test_dpa_attack(self):
        result = subprocess.run(
            [sys.executable, "-m", "tessera.cli", "attack",
             "--type", "dpa", "--traces", "200", "--seed", "42"],
            capture_output=True, text=True, timeout=60
        )
        assert result.returncode == 0
        assert "DPA" in result.stdout
        assert "Recovered key" in result.stdout
    
    def test_tvla_attack(self):
        result = subprocess.run(
            [sys.executable, "-m", "tessera.cli", "attack",
             "--type", "tvla", "--traces", "200", "--seed", "42"],
            capture_output=True, text=True, timeout=60
        )
        assert result.returncode == 0
        assert "TVLA" in result.stdout
        assert "t-value" in result.stdout


class TestCLIBenchmark:
    
    def test_benchmark_ntt(self):
        result = subprocess.run(
            [sys.executable, "-m", "tessera.cli", "benchmark", "--ntt", "-n", "10"],
            capture_output=True, text=True, timeout=60
        )
        assert result.returncode == 0
        assert "NTT" in result.stdout
        assert "Kyber" in result.stdout
        assert "Dilithium" in result.stdout
    
    def test_benchmark_kyber(self):
        result = subprocess.run(
            [sys.executable, "-m", "tessera.cli", "benchmark", "--kyber", "-n", "2"],
            capture_output=True, text=True, timeout=120
        )
        assert result.returncode == 0
        assert "Kyber-512" in result.stdout
        assert "Kyber-768" in result.stdout
        assert "Kyber-1024" in result.stdout
    
    def test_benchmark_output_json(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = f.name
        
        try:
            result = subprocess.run(
                [sys.executable, "-m", "tessera.cli", "benchmark", 
                 "--ntt", "-n", "5", "--output", output_path],
                capture_output=True, text=True, timeout=60
            )
            assert result.returncode == 0
            
            with open(output_path) as f:
                data = json.load(f)
            
            assert "ntt_kyber" in data
            assert "ntt_dilithium" in data
            assert "mean_us" in data["ntt_kyber"]
        finally:
            Path(output_path).unlink(missing_ok=True)


class TestCLIVerify:
    
    def test_verify_ntt(self):
        result = subprocess.run(
            [sys.executable, "-m", "tessera.cli", "verify", "--count", "3"],
            capture_output=True, text=True, timeout=30
        )
        assert result.returncode == 0
        assert "PASS" in result.stdout
        assert "tests PASSED" in result.stdout


class TestHighLevelAPI:
    
    def test_kyber_roundtrip(self):
        import tessera as t
        pk, sk = t.kyber_keygen("512")
        ct, ss1 = t.kyber_encaps(pk, "512")
        ss2 = t.kyber_decaps(sk, ct, "512")
        assert ss1 == ss2
    
    def test_dilithium_roundtrip(self):
        import tessera as t
        pk, sk = t.dilithium_keygen("2")
        sig = t.dilithium_sign(sk, b"test message", "2")
        valid = t.dilithium_verify(pk, b"test message", sig, "2")
        assert valid
    
    def test_generate_traces(self):
        import tessera as t
        data = t.generate_traces(50, 16, key_bytes=4, seed=42)
        assert data["traces"].shape == (50, 16)
        assert data["plaintexts"].shape == (50, 4)
        assert len(data["key"]) == 4
    
    def test_run_cpa(self):
        import tessera as t
        data = t.generate_traces(500, 16, key_bytes=4, noise_level=0.1, seed=42)
        result = t.run_cpa(data["traces"], data["plaintexts"], key_bytes=[0])
        assert len(result["recovered_key"]) == 1
    
    def test_run_tvla(self):
        import tessera as t
        rng = np.random.default_rng(42)
        traces1 = rng.standard_normal((50, 32))
        traces2 = rng.standard_normal((50, 32))
        result = t.run_tvla(traces1, traces2)
        assert "leakage_detected" in result
        assert "max_t_value" in result
        assert "t_values" in result
    
    def test_compute_snr(self):
        import tessera as t
        rng = np.random.default_rng(42)
        traces = rng.standard_normal((100, 32))
        labels = rng.integers(0, 4, 100)
        snr = t.compute_snr(traces, labels)
        assert snr.shape == (32,)
    
    def test_apply_masking(self):
        import tessera as t
        data = np.array([0x12, 0x34, 0x56, 0x78], dtype=np.uint64)
        result = t.apply_masking(data, num_shares=2, mask_type="boolean")
        assert "shares" in result
        assert len(result["shares"]) == 2
    
    def test_export_traces_npz(self):
        import tessera as t
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            output_path = f.name
        
        try:
            traces = np.random.randn(10, 32)
            t.export_traces_npz(output_path, traces, key=np.array([1, 2, 3]))
            
            loaded = t.load_traces_npz(output_path)
            assert "traces" in loaded
            assert "key" in loaded
            np.testing.assert_array_almost_equal(loaded["traces"], traces)
        finally:
            Path(output_path).unlink(missing_ok=True)
    
    def test_benchmark_kyber(self):
        import tessera as t
        result = t.benchmark_kyber("512", iterations=2)
        assert "keygen_ms" in result
        assert "encaps_ms" in result
        assert "decaps_ms" in result
        assert result["keygen_ms"] > 0
    
    def test_benchmark_dilithium(self):
        import tessera as t
        result = t.benchmark_dilithium("2", iterations=2)
        assert "keygen_ms" in result
        assert "sign_ms" in result
        assert "verify_ms" in result
        assert result["keygen_ms"] > 0


class TestAPIErrors:
    
    def test_invalid_kyber_variant(self):
        import tessera as t
        with pytest.raises(ValueError, match="Unknown variant"):
            t.kyber_keygen("256")
    
    def test_invalid_dilithium_variant(self):
        import tessera as t
        with pytest.raises(ValueError, match="Unknown variant"):
            t.dilithium_keygen("1")
    
    def test_invalid_masking_type(self):
        import tessera as t
        with pytest.raises(ValueError, match="Unknown mask_type"):
            t.apply_masking(np.array([1, 2, 3]), mask_type="invalid")
