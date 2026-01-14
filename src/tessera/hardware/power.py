import simpy
import random

class PowerSource:
    """
    Simulates an intermittent power harvester (e.g., RF or Solar).
    Acts as a 'Chaos Monkey' that interrupts the CPU.
    """

    def __init__(self, env: simpy.Environment, on_time_avg=100, off_time_avg=50):
        self.env = env
        self.on_time_avg = on_time_avg
        self.off_time_avg = off_time_avg
        # Event triggered when power is lost
        self.power_lost = env.event()
        # Event triggered when power is restored
        self.power_restored = env.event()
        self.is_powered = True
        
        # Start the chaos process
        self.env.process(self._power_cycle_process())

    def _power_cycle_process(self):
        """
        Internal process that toggles power state based on distributions.
        """
        while True:
            # 1. Device runs for some time
            duration = random.expovariate(1.0 / self.on_time_avg)
            yield self.env.timeout(duration)
            
            # 2. Power Failure!
            self.is_powered = False
            self.power_lost.succeed() # Notify listeners
            self.power_lost = self.env.event() # Reset event for next time
            print(f"[HW] Power FAILURE at {self.env.now:.2f}")

            # 3. Device stays dead for some time
            off_duration = random.expovariate(1.0 / self.off_time_avg)
            yield self.env.timeout(off_duration)

            # 4. Power Restore
            self.is_powered = True
            self.power_restored.succeed()
            self.power_restored = self.env.event()
            print(f"[HW] Power RESTORED at {self.env.now:.2f}")
