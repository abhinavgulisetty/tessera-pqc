"""
Tessera CLI — entry point for all simulation and analysis commands.

Commands
--------
  tessera demo     Full animated terminal demonstration (Rich).
  tessera run      Run an intermittent Atomic-NTT simulation.
  tessera kem      Exercise the Baby-Kyber KEM (keygen → encaps → decaps).
  tessera verify   Verify the NTT round-trip: inv_ntt(ntt(x)) == x.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import simpy

from tessera.hardware.power  import PowerSource
from tessera.hardware.memory import NonVolatileMemory
from tessera.scheduler       import AtomicTaskScheduler
from tessera.core.math       import PolynomialRing
from tessera.core.primitives import LatticeKEM


def _plot_leakage(nvm: NonVolatileMemory) -> None:
    """Plot the Hamming-Weight power side-channel trace using Matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Warning] matplotlib not installed — skipping plot.")
        return

    times  = nvm.times()
    powers = nvm.power_values()
    if not times:
        print("[Info] No leakage data to plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, powers, marker='o', linewidth=1.2, color='royalblue',
            markersize=4, label='HW leakage')
    ax.set_xlabel("Simulation time (arbitrary units)")
    ax.set_ylabel("Hamming Weight (proxy for power consumption)")
    ax.set_title("Tessera — NVM Write Leakage Trace (Hamming Weight Model)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("leakage_trace.png", dpi=150)
    print("[Plot] Leakage trace saved → leakage_trace.png")
    plt.show()


def cmd_demo(args) -> None:
    """Full Rich animated terminal demo of all three phases."""
    from tessera.demo import run_demo
    run_demo(duration=args.duration, on_avg=args.on_avg, off_avg=args.off_avg)


def cmd_run(args) -> None:
    """Run an intermittent Atomic-NTT power simulation."""
    print("=" * 60)
    print(f" Tessera — Atomic NTT Simulation")
    print(f" Duration : {args.duration} time-units")
    print(f" Power    : on_avg={args.on_avg}  off_avg={args.off_avg}")
    print("=" * 60)

    env    = simpy.Environment()
    ring   = PolynomialRing()
    power  = PowerSource(env,
                         on_time_avg  = args.on_avg,
                         off_time_avg = args.off_avg)
    nvm    = NonVolatileMemory()
    sched  = AtomicTaskScheduler(env, power, nvm, ring)

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
        _plot_leakage(nvm)


def cmd_kem(args) -> None:
    """Run the Baby-Kyber KEM and verify shared secret agreement."""
    print("=" * 60)
    print(" Tessera — Baby-Kyber KEM Demo")
    print("=" * 60)

    ring = PolynomialRing()
    kem  = LatticeKEM(ring)

    print("[KEM] Generating key pair...")
    pk, sk = kem.keygen()
    print(f"      pk length = {len(pk)} bytes")
    print(f"      sk length = {len(sk)} bytes")

    print("[KEM] Encapsulating...")
    ct, ss_enc = kem.encaps(pk)
    print(f"      ciphertext length = {len(ct)} bytes")
    print(f"      shared secret (enc) = {ss_enc.hex()[:32]}...")

    print("[KEM] Decapsulating...")
    ss_dec = kem.decaps(sk, ct)
    print(f"      shared secret (dec) = {ss_dec.hex()[:32]}...")

    if ss_enc == ss_dec:
        print("[KEM] SUCCESS — shared secrets match! ✓")
    else:
        print("[KEM] MISMATCH — decapsulation failed (check parameters).")
        sys.exit(1)


def cmd_verify(args) -> None:
    """Verify the NTT round-trip: inv_ntt(ntt(x)) == x for random inputs."""
    print("=" * 60)
    print(" Tessera — NTT Round-Trip Verification")
    print("=" * 60)

    ring = PolynomialRing()
    n_tests = args.count
    failures = 0

    for i in range(n_tests):
        x = np.random.randint(0, ring.q, ring.n, dtype=np.int64)
        ok = ring.verify_round_trip(x)
        if not ok:
            failures += 1
            print(f"  [Test {i+1}] FAIL")
        else:
            print(f"  [Test {i+1}] PASS")

    print()
    if failures == 0:
        print(f"All {n_tests} round-trip tests PASSED. ✓")
    else:
        print(f"{failures}/{n_tests} tests FAILED.")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tessera",
        description="Tessera-PQC: Intermittent Atomic Post-Quantum Crypto Simulator"
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    p_demo = sub.add_parser("demo", help="Full animated Rich terminal demonstration")
    p_demo.add_argument("--duration", type=int,   default=800,
                        help="Simulation duration for the demo (default=800)")
    p_demo.add_argument("--on-avg",  dest="on_avg",  type=float, default=100,
                        help="Mean powered-on time (default=100)")
    p_demo.add_argument("--off-avg", dest="off_avg", type=float, default=40,
                        help="Mean powered-off time (default=40)")

    p_run = sub.add_parser("run", help="Run an intermittent Atomic-NTT simulation")
    p_run.add_argument("--duration",  type=int,   default=1000,
                       help="Simulation duration (time units, default=1000)")
    p_run.add_argument("--on-avg",   dest="on_avg",  type=float, default=120,
                       help="Mean time device is powered (default=120)")
    p_run.add_argument("--off-avg",  dest="off_avg", type=float, default=40,
                       help="Mean time device is off (default=40)")
    p_run.add_argument("--plot",     action="store_true",
                       help="Show & save leakage trace plot after simulation")

    p_kem = sub.add_parser("kem", help="Run the Baby-Kyber KEM demonstration")
    _ = p_kem

    p_ver = sub.add_parser("verify", help="Verify NTT round-trip correctness")
    p_ver.add_argument("--count", type=int, default=5,
                       help="Number of random polynomials to test (default=5)")

    args = parser.parse_args()

    if args.command == "demo":
        cmd_demo(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "kem":
        cmd_kem(args)
    elif args.command == "verify":
        cmd_verify(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
