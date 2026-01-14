import simpy
from .hardware.power import PowerSource
from .hardware.memory import NonVolatileMemory

class AtomicTaskScheduler:
    """
    Orchestrates the cryptographic operation. 
    It splits big tasks into atomic tiles (Tesserae) that fit within 
    SimPy simulation steps.
    """

    def __init__(self, env: simpy.Environment, power: PowerSource, nvm: NonVolatileMemory):
        self.env = env
        self.power = power
        self.nvm = nvm
        self.current_step = 0

    def run_atomic_ntt(self, poly_data):
        """
        A generator process that tries to compute NTT.
        It must check `self.power.is_powered` before proceeding.
        If power fails, it waits for `self.power.power_restored` and minimizes data loss.
        """
        # TODO:
        # 1. Break NTT into layers (stages).
        # 2. For each stage:
        #    a. Check power. If dead, yield self.power.power_restored.
        #    b. If alive, compute stage (expend time: yield env.timeout(cost)).
        #    c. Write checkpoint to NVM (cost time + leakage).
        
        print(f"[Scheduler] Starting Atomic NTT at {self.env.now}")
        
        # Example scaffolding for the loop
        total_layers = 7 # e.g., for n=256
        for i in range(self.current_step, total_layers):
            if not self.power.is_powered:
                print("[Scheduler] Waiting for power...")
                yield self.power.power_restored
                # On wake up, reload from NVM
                # poly_data = self.nvm.read_checkpoint(...)

            # Simulate computation time
            yield self.env.timeout(10) 
            
            # Checkpoint
            # self.nvm.write_checkpoint(..., poly_data, self.env.now)
            self.current_step = i
            
        print(f"[Scheduler] Finished NTT at {self.env.now}")
