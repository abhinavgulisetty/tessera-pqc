import argparse
import simpy
from tessera.hardware.power import PowerSource
from tessera.hardware.memory import NonVolatileMemory
from tessera.scheduler import AtomicTaskScheduler

def run_simulation(args):
    """
    Sets up the SimPy environment and starts the simulation.
    """
    print(f"Initializing Tessera Simulation (Duration: {args.duration})...")
    
    env = simpy.Environment()
    
    # Init Hardware
    power_source = PowerSource(env)
    nvm = NonVolatileMemory()
    
    # Init Scheduler
    scheduler = AtomicTaskScheduler(env, power_source, nvm)
    
    # Start the "main loop" process
    env.process(scheduler.run_atomic_ntt(poly_data=None))
    
    # Run
    env.run(until=args.duration)
    print("Simulation Complete.")

def main():
    parser = argparse.ArgumentParser(description="Tessera-PQC: Intermittent Atomic Crypto Simulator")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # 'run' command
    run_parser = subparsers.add_parser("run", help="Run a simulation")
    run_parser.add_argument("--duration", type=int, default=1000, help="Simulation steps")
    run_parser.add_argument("--scheme", type=str, default="kyber512", help="Target algorithm")

    args = parser.parse_args()

    if args.command == "run":
        run_simulation(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
