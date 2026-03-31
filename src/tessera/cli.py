from __future__ import annotations

import argparse
import sys
import time
import json
from pathlib import Path
from typing import Optional

import numpy as np


def _get_kyber(variant: str):
    from tessera.algorithms.kyber import kyber512, kyber768, kyber1024
    variants = {
        "512": kyber512,
        "768": kyber768,
        "1024": kyber1024,
    }
    if variant not in variants:
        print(f"[Error] Unknown Kyber variant: {variant}")
        print(f"        Available: {list(variants.keys())}")
        sys.exit(1)
    return variants[variant]()


def _get_dilithium(variant: str):
    from tessera.algorithms.dilithium import dilithium2, dilithium3, dilithium5
    variants = {
        "2": dilithium2,
        "3": dilithium3,
        "5": dilithium5,
    }
    if variant not in variants:
        print(f"[Error] Unknown Dilithium variant: {variant}")
        print(f"        Available: {list(variants.keys())}")
        sys.exit(1)
    return variants[variant]()


def cmd_kyber(args) -> None:
    print("=" * 60)
    print(f" Tessera - Kyber-{args.variant} KEM")
    print("=" * 60)

    kyber = _get_kyber(args.variant)

    print(f"\n[Kyber-{args.variant}] Generating key pair...")
    t0 = time.perf_counter()
    pk, sk = kyber.keygen()
    t_keygen = time.perf_counter() - t0
    print(f"  Public key:  {len(pk):,} bytes")
    print(f"  Secret key:  {len(sk):,} bytes")
    print(f"  Time: {t_keygen*1000:.2f} ms")

    print(f"\n[Kyber-{args.variant}] Encapsulating...")
    t0 = time.perf_counter()
    ct, ss_enc = kyber.encaps(pk)
    t_encaps = time.perf_counter() - t0
    print(f"  Ciphertext:  {len(ct):,} bytes")
    print(f"  Shared secret: {ss_enc[:16].hex()}...")
    print(f"  Time: {t_encaps*1000:.2f} ms")

    print(f"\n[Kyber-{args.variant}] Decapsulating...")
    t0 = time.perf_counter()
    ss_dec = kyber.decaps(sk, ct)
    t_decaps = time.perf_counter() - t0
    print(f"  Shared secret: {ss_dec[:16].hex()}...")
    print(f"  Time: {t_decaps*1000:.2f} ms")

    if ss_enc == ss_dec:
        print(f"\n[Kyber-{args.variant}] SUCCESS - shared secrets match!")
    else:
        print(f"\n[Kyber-{args.variant}] FAILURE - shared secrets do NOT match!")
        sys.exit(1)

    if args.benchmark:
        print(f"\n[Benchmark] Running {args.iterations} iterations...")
        times_kg, times_enc, times_dec = [], [], []
        for _ in range(args.iterations):
            t0 = time.perf_counter()
            pk, sk = kyber.keygen()
            times_kg.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            ct, ss = kyber.encaps(pk)
            times_enc.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            kyber.decaps(sk, ct)
            times_dec.append(time.perf_counter() - t0)

        print(f"  KeyGen:  {np.mean(times_kg)*1000:.2f} ms (std: {np.std(times_kg)*1000:.2f})")
        print(f"  Encaps:  {np.mean(times_enc)*1000:.2f} ms (std: {np.std(times_enc)*1000:.2f})")
        print(f"  Decaps:  {np.mean(times_dec)*1000:.2f} ms (std: {np.std(times_dec)*1000:.2f})")


def cmd_dilithium(args) -> None:
    print("=" * 60)
    print(f" Tessera - Dilithium{args.variant} Signature")
    print("=" * 60)

    dil = _get_dilithium(args.variant)
    message = args.message.encode() if args.message else b"Test message for Dilithium signature"

    print(f"\n[Dilithium{args.variant}] Generating key pair...")
    t0 = time.perf_counter()
    pk, sk = dil.keygen()
    t_keygen = time.perf_counter() - t0
    print(f"  Public key:  {len(pk):,} bytes")
    print(f"  Secret key:  {len(sk):,} bytes")
    print(f"  Time: {t_keygen*1000:.2f} ms")

    print(f"\n[Dilithium{args.variant}] Signing message ({len(message)} bytes)...")
    t0 = time.perf_counter()
    sig = dil.sign(sk, message)
    t_sign = time.perf_counter() - t0
    print(f"  Signature:   {len(sig):,} bytes")
    print(f"  Time: {t_sign*1000:.2f} ms")

    print(f"\n[Dilithium{args.variant}] Verifying signature...")
    t0 = time.perf_counter()
    valid = dil.verify(pk, message, sig)
    t_verify = time.perf_counter() - t0
    print(f"  Valid: {valid}")
    print(f"  Time: {t_verify*1000:.2f} ms")

    if valid:
        print(f"\n[Dilithium{args.variant}] SUCCESS - signature verified!")
    else:
        print(f"\n[Dilithium{args.variant}] FAILURE - signature invalid!")
        sys.exit(1)

    if args.benchmark:
        print(f"\n[Benchmark] Running {args.iterations} iterations...")
        times_kg, times_sign, times_ver = [], [], []
        for _ in range(args.iterations):
            t0 = time.perf_counter()
            pk, sk = dil.keygen()
            times_kg.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            sig = dil.sign(sk, message)
            times_sign.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            dil.verify(pk, message, sig)
            times_ver.append(time.perf_counter() - t0)

        print(f"  KeyGen: {np.mean(times_kg)*1000:.2f} ms (std: {np.std(times_kg)*1000:.2f})")
        print(f"  Sign:   {np.mean(times_sign)*1000:.2f} ms (std: {np.std(times_sign)*1000:.2f})")
        print(f"  Verify: {np.mean(times_ver)*1000:.2f} ms (std: {np.std(times_ver)*1000:.2f})")


def cmd_attack(args) -> None:
    print("=" * 60)
    print(f" Tessera - Side-Channel Attack Simulation")
    print("=" * 60)

    from tessera.leakage import HammingWeightModel
    from tessera.attacks import CPA, DPA, WelchTTest, TemplateAttack

    n_traces = args.traces
    trace_len = args.trace_length
    key_bytes = args.key_bytes

    print(f"\n[Setup] Generating {n_traces} traces (length {trace_len})...")
    rng = np.random.default_rng(args.seed)
    true_key = rng.integers(0, 256, key_bytes, dtype=np.uint8)
    print(f"  True key: {true_key[:8].tolist()}{'...' if key_bytes > 8 else ''}")

    model = HammingWeightModel(8)
    traces = []
    plaintexts = []

    for _ in range(n_traces):
        pt = rng.integers(0, 256, key_bytes, dtype=np.uint8)
        plaintexts.append(pt)
        intermediate = pt ^ true_key
        leakage = np.array([model(int(v)) for v in intermediate], dtype=np.float64)
        noise = rng.standard_normal(trace_len) * args.noise
        trace = np.zeros(trace_len)
        trace[:key_bytes] = leakage
        trace += noise
        traces.append(trace)

    traces = np.array(traces)
    plaintexts = np.array(plaintexts)

    if args.attack_type == "cpa":
        print(f"\n[CPA] Running Correlation Power Analysis...")
        
        def xor_intermediate(pt_byte, key_guess):
            return pt_byte ^ key_guess
        
        attack = CPA(intermediate_func=xor_intermediate)
        t0 = time.perf_counter()
        
        recovered = []
        for byte_idx in range(key_bytes):
            best_key, confidence, _ = attack.attack_byte(traces, plaintexts, byte_idx)
            recovered.append(best_key)
        
        t_attack = time.perf_counter() - t0
        recovered = np.array(recovered, dtype=np.uint8)
        correct = np.sum(recovered == true_key)
        print(f"  Recovered key: {recovered[:8].tolist()}{'...' if key_bytes > 8 else ''}")
        print(f"  Correct bytes: {correct}/{key_bytes}")
        print(f"  Time: {t_attack*1000:.2f} ms")

    elif args.attack_type == "dpa":
        print(f"\n[DPA] Running Differential Power Analysis...")
        
        def xor_intermediate(pt_byte, key_guess):
            return pt_byte ^ key_guess
        
        attack = DPA(intermediate_func=xor_intermediate)
        t0 = time.perf_counter()
        
        recovered = []
        for byte_idx in range(key_bytes):
            best_key, confidence, _ = attack.attack_byte(traces, plaintexts, byte_idx)
            recovered.append(best_key)
        
        t_attack = time.perf_counter() - t0
        recovered = np.array(recovered, dtype=np.uint8)
        correct = np.sum(recovered == true_key)
        print(f"  Recovered key: {recovered[:8].tolist()}{'...' if key_bytes > 8 else ''}")
        print(f"  Correct bytes: {correct}/{key_bytes}")
        print(f"  Time: {t_attack*1000:.2f} ms")

    elif args.attack_type == "tvla":
        print(f"\n[TVLA] Running Test Vector Leakage Assessment...")
        analyzer = WelchTTest()

        fixed_traces = traces[:n_traces//2]
        random_traces = traces[n_traces//2:]

        t0 = time.perf_counter()
        result = analyzer.assess(fixed_traces, random_traces)
        t_attack = time.perf_counter() - t0

        print(f"  Max |t-value|: {result.max_statistic:.2f}")
        print(f"  Threshold: {result.threshold:.2f}")
        print(f"  Leakage detected: {result.leakage_detected}")
        print(f"  Time: {t_attack*1000:.2f} ms")

    elif args.attack_type == "template":
        print(f"\n[Template] Running Template Attack...")
        
        n_profiling = n_traces * 2 // 3
        profiling_traces = traces[:n_profiling]
        profiling_labels = plaintexts[:n_profiling, 0] ^ true_key[0]

        attack_traces = traces[n_profiling:]
        attack_plaintexts = plaintexts[n_profiling:]

        t0 = time.perf_counter()
        
        from tessera.leakage import TraceSet
        profiling_set = TraceSet(profiling_traces, labels=profiling_labels)
        
        attack = TemplateAttack(num_pois=min(5, trace_len))
        attack.build_templates(profiling_set, profiling_labels)
        
        predictions = []
        for i in range(len(attack_traces)):
            probs = attack.attack_trace(attack_traces[i])
            pred_inter = int(np.argmax(probs))
            pred_key = pred_inter ^ int(attack_plaintexts[i, 0])
            predictions.append(pred_key)
        
        t_attack = time.perf_counter() - t0

        from collections import Counter
        key_votes = Counter(predictions)
        if key_votes:
            recovered_key = key_votes.most_common(1)[0][0]
            print(f"  Recovered key byte 0: {recovered_key} (true: {true_key[0]})")
            print(f"  Match: {recovered_key == true_key[0]}")
        print(f"  Time: {t_attack*1000:.2f} ms")


def cmd_analyze(args) -> None:
    print("=" * 60)
    print(" Tessera - Trace Analysis")
    print("=" * 60)

    if not args.input:
        print("[Error] --input file required")
        sys.exit(1)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[Error] File not found: {input_path}")
        sys.exit(1)

    print(f"\n[Load] Reading traces from {input_path}...")
    data = np.load(input_path)
    traces = data["traces"] if "traces" in data else data[data.files[0]]
    print(f"  Shape: {traces.shape}")
    print(f"  Dtype: {traces.dtype}")

    from tessera.analysis import SNRCalculator

    if "labels" in data:
        labels = data["labels"]
        print(f"\n[SNR] Computing Signal-to-Noise Ratio...")
        snr_calc = SNRCalculator()
        snr = snr_calc.compute_snr(traces, labels)
        print(f"  Max SNR: {np.max(snr):.4f}")
        print(f"  Mean SNR: {np.mean(snr):.4f}")
        print(f"  Points with SNR > 0.1: {np.sum(snr > 0.1)}")

    if args.output:
        output_path = Path(args.output)
        from tessera.analysis import CSVExporter, JSONExporter

        if output_path.suffix == ".csv":
            exporter = CSVExporter()
            stats = np.column_stack([np.mean(traces, axis=0), np.std(traces, axis=0)])
            exporter.export_traces(stats, output_path)
            print(f"\n[Export] Saved to {output_path}")
        elif output_path.suffix == ".json":
            exporter = JSONExporter()
            stats = {
                "n_traces": int(traces.shape[0]),
                "trace_length": int(traces.shape[1]),
                "mean": float(np.mean(traces)),
                "std": float(np.std(traces)),
            }
            exporter.export_stats(stats, output_path)
            print(f"\n[Export] Saved to {output_path}")


def cmd_benchmark(args) -> None:
    print("=" * 60)
    print(" Tessera - Performance Benchmark")
    print("=" * 60)

    results = {}
    iterations = args.iterations

    if args.ntt or args.all:
        print(f"\n[NTT] Benchmarking NTT operations...")
        from tessera.core.math_fast import ntt_kyber, intt_kyber, ntt_dilithium, intt_dilithium

        rng = np.random.default_rng(42)
        poly_kyber = rng.integers(0, 3329, 256, dtype=np.int64)
        poly_dil = rng.integers(0, 8380417, 256, dtype=np.int64)

        ntt_kyber(poly_kyber.copy())
        ntt_dilithium(poly_dil.copy())

        times = []
        for _ in range(iterations):
            p = poly_kyber.copy()
            t0 = time.perf_counter()
            ntt_kyber(p)
            times.append(time.perf_counter() - t0)
        results["ntt_kyber"] = {"mean_us": np.mean(times)*1e6, "std_us": np.std(times)*1e6}
        print(f"  NTT (Kyber):     {results['ntt_kyber']['mean_us']:.2f} us")

        times = []
        for _ in range(iterations):
            p = poly_dil.copy()
            t0 = time.perf_counter()
            ntt_dilithium(p)
            times.append(time.perf_counter() - t0)
        results["ntt_dilithium"] = {"mean_us": np.mean(times)*1e6, "std_us": np.std(times)*1e6}
        print(f"  NTT (Dilithium): {results['ntt_dilithium']['mean_us']:.2f} us")

    if args.kyber or args.all:
        print(f"\n[Kyber] Benchmarking Kyber variants...")
        for variant in ["512", "768", "1024"]:
            kyber = _get_kyber(variant)
            times_kg, times_enc, times_dec = [], [], []
            for _ in range(iterations):
                t0 = time.perf_counter()
                pk, sk = kyber.keygen()
                times_kg.append(time.perf_counter() - t0)
                t0 = time.perf_counter()
                ct, ss = kyber.encaps(pk)
                times_enc.append(time.perf_counter() - t0)
                t0 = time.perf_counter()
                kyber.decaps(sk, ct)
                times_dec.append(time.perf_counter() - t0)

            results[f"kyber_{variant}"] = {
                "keygen_ms": np.mean(times_kg)*1000,
                "encaps_ms": np.mean(times_enc)*1000,
                "decaps_ms": np.mean(times_dec)*1000,
            }
            print(f"  Kyber-{variant}: keygen={results[f'kyber_{variant}']['keygen_ms']:.2f}ms "
                  f"encaps={results[f'kyber_{variant}']['encaps_ms']:.2f}ms "
                  f"decaps={results[f'kyber_{variant}']['decaps_ms']:.2f}ms")

    if args.dilithium or args.all:
        print(f"\n[Dilithium] Benchmarking Dilithium variants...")
        msg = b"Benchmark message"
        for variant in ["2", "3", "5"]:
            dil = _get_dilithium(variant)
            times_kg, times_sign, times_ver = [], [], []
            for _ in range(iterations):
                t0 = time.perf_counter()
                pk, sk = dil.keygen()
                times_kg.append(time.perf_counter() - t0)
                t0 = time.perf_counter()
                sig = dil.sign(sk, msg)
                times_sign.append(time.perf_counter() - t0)
                t0 = time.perf_counter()
                dil.verify(pk, msg, sig)
                times_ver.append(time.perf_counter() - t0)

            results[f"dilithium_{variant}"] = {
                "keygen_ms": np.mean(times_kg)*1000,
                "sign_ms": np.mean(times_sign)*1000,
                "verify_ms": np.mean(times_ver)*1000,
            }
            print(f"  Dilithium{variant}: keygen={results[f'dilithium_{variant}']['keygen_ms']:.2f}ms "
                  f"sign={results[f'dilithium_{variant}']['sign_ms']:.2f}ms "
                  f"verify={results[f'dilithium_{variant}']['verify_ms']:.2f}ms")

    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[Export] Results saved to {output_path}")


def cmd_demo(args) -> None:
    from tessera.demo import run_demo
    run_demo(duration=args.duration, on_avg=args.on_avg, off_avg=args.off_avg)


def cmd_run(args) -> None:
    import simpy
    from tessera.hardware.power import PowerSource
    from tessera.hardware.memory import NonVolatileMemory
    from tessera.scheduler import AtomicTaskScheduler
    from tessera.core.math import PolynomialRing

    print("=" * 60)
    print(f" Tessera - Atomic NTT Simulation")
    print(f" Duration : {args.duration} time-units")
    print(f" Power    : on_avg={args.on_avg}  off_avg={args.off_avg}")
    print("=" * 60)

    env = simpy.Environment()
    ring = PolynomialRing()
    power = PowerSource(env, on_time_avg=args.on_avg, off_time_avg=args.off_avg)
    nvm = NonVolatileMemory()
    sched = AtomicTaskScheduler(env, power, nvm, ring)

    poly = np.random.randint(0, ring.q, ring.n, dtype=np.int64)
    env.process(sched.run_atomic_ntt(poly_data=poly))
    env.run(until=args.duration)

    print()
    print("=" * 60)
    print(" Simulation Summary")
    print(f"  Completed layers : {sched.completed_layers}")
    print(f"  Power failures   : {sched.power_failures}")
    print(f"  NVM restores     : {sched.restores}")
    print(f"  {nvm.summary()}")
    print("=" * 60)

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("[Warning] matplotlib not installed - skipping plot.")
            return

        times = nvm.times()
        powers = nvm.power_values()
        if times:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(times, powers, marker='o', linewidth=1.2, color='royalblue', markersize=4)
            ax.set_xlabel("Simulation time")
            ax.set_ylabel("Hamming Weight")
            ax.set_title("Tessera - NVM Write Leakage Trace")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig("leakage_trace.png", dpi=150)
            print("[Plot] Saved to leakage_trace.png")
            plt.show()


def cmd_verify(args) -> None:
    from tessera.core.math import PolynomialRing

    print("=" * 60)
    print(" Tessera - NTT Round-Trip Verification")
    print("=" * 60)

    ring = PolynomialRing()
    failures = 0

    for i in range(args.count):
        x = np.random.randint(0, ring.q, ring.n, dtype=np.int64)
        ok = ring.verify_round_trip(x)
        status = "PASS" if ok else "FAIL"
        print(f"  [Test {i+1}] {status}")
        if not ok:
            failures += 1

    print()
    if failures == 0:
        print(f"All {args.count} tests PASSED.")
    else:
        print(f"{failures}/{args.count} tests FAILED.")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tessera",
        description="Tessera-PQC: Post-Quantum Cryptography Research Framework"
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.2.0")
    sub = parser.add_subparsers(dest="command", help="Available commands")

    p_kyber = sub.add_parser("kyber", help="Run Kyber KEM (512/768/1024)")
    p_kyber.add_argument("--variant", "-v", default="768", choices=["512", "768", "1024"],
                         help="Kyber variant (default: 768)")
    p_kyber.add_argument("--benchmark", "-b", action="store_true",
                         help="Run benchmark iterations")
    p_kyber.add_argument("--iterations", "-n", type=int, default=10,
                         help="Benchmark iterations (default: 10)")

    p_dil = sub.add_parser("dilithium", help="Run Dilithium signatures (2/3/5)")
    p_dil.add_argument("--variant", "-v", default="3", choices=["2", "3", "5"],
                       help="Dilithium variant (default: 3)")
    p_dil.add_argument("--message", "-m", type=str, default=None,
                       help="Message to sign")
    p_dil.add_argument("--benchmark", "-b", action="store_true",
                       help="Run benchmark iterations")
    p_dil.add_argument("--iterations", "-n", type=int, default=10,
                       help="Benchmark iterations (default: 10)")

    p_attack = sub.add_parser("attack", help="Run side-channel attack simulation")
    p_attack.add_argument("--type", "-t", dest="attack_type", default="cpa",
                          choices=["cpa", "dpa", "tvla", "template"],
                          help="Attack type (default: cpa)")
    p_attack.add_argument("--traces", type=int, default=1000,
                          help="Number of traces (default: 1000)")
    p_attack.add_argument("--trace-length", type=int, default=32,
                          help="Trace length (default: 32)")
    p_attack.add_argument("--key-bytes", type=int, default=16,
                          help="Key size in bytes (default: 16)")
    p_attack.add_argument("--noise", type=float, default=0.5,
                          help="Noise level (default: 0.5)")
    p_attack.add_argument("--seed", type=int, default=None,
                          help="Random seed for reproducibility")

    p_analyze = sub.add_parser("analyze", help="Analyze trace files")
    p_analyze.add_argument("--input", "-i", type=str, required=False,
                           help="Input file (.npz)")
    p_analyze.add_argument("--output", "-o", type=str,
                           help="Output file (.csv, .json)")

    p_bench = sub.add_parser("benchmark", help="Run performance benchmarks")
    p_bench.add_argument("--iterations", "-n", type=int, default=100,
                         help="Iterations per benchmark (default: 100)")
    p_bench.add_argument("--ntt", action="store_true", help="Benchmark NTT")
    p_bench.add_argument("--kyber", action="store_true", help="Benchmark Kyber")
    p_bench.add_argument("--dilithium", action="store_true", help="Benchmark Dilithium")
    p_bench.add_argument("--all", "-a", action="store_true", help="Run all benchmarks")
    p_bench.add_argument("--output", "-o", type=str, help="Output JSON file")

    p_demo = sub.add_parser("demo", help="Full animated Rich terminal demonstration")
    p_demo.add_argument("--duration", type=int, default=800,
                        help="Simulation duration (default: 800)")
    p_demo.add_argument("--on-avg", dest="on_avg", type=float, default=100,
                        help="Mean powered-on time (default: 100)")
    p_demo.add_argument("--off-avg", dest="off_avg", type=float, default=40,
                        help="Mean powered-off time (default: 40)")

    p_run = sub.add_parser("run", help="Run intermittent Atomic-NTT simulation")
    p_run.add_argument("--duration", type=int, default=1000,
                       help="Simulation duration (default: 1000)")
    p_run.add_argument("--on-avg", dest="on_avg", type=float, default=120,
                       help="Mean powered-on time (default: 120)")
    p_run.add_argument("--off-avg", dest="off_avg", type=float, default=40,
                       help="Mean powered-off time (default: 40)")
    p_run.add_argument("--plot", action="store_true",
                       help="Save and show leakage plot")

    p_verify = sub.add_parser("verify", help="Verify NTT round-trip correctness")
    p_verify.add_argument("--count", type=int, default=5,
                          help="Number of tests (default: 5)")

    args = parser.parse_args()

    commands = {
        "kyber": cmd_kyber,
        "dilithium": cmd_dilithium,
        "attack": cmd_attack,
        "analyze": cmd_analyze,
        "benchmark": cmd_benchmark,
        "demo": cmd_demo,
        "run": cmd_run,
        "verify": cmd_verify,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
