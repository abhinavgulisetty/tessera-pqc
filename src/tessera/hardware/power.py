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
        self.power_lost = env.event()
        self.power_restored = env.event()
        self.is_powered = True
        
        self.env.process(self._power_cycle_process())

    def _power_cycle_process(self):
        """
        Internal process that toggles power state based on distributions.
        """
        while True:
            duration = random.expovariate(1.0 / self.on_time_avg)
            yield self.env.timeout(duration)
            
            self.is_powered = False
            self.power_lost.succeed()
            self.power_lost = self.env.event()
            print(f"[HW] Power FAILURE at {self.env.now:.2f}")

            off_duration = random.expovariate(1.0 / self.off_time_avg)
            yield self.env.timeout(off_duration)

            self.is_powered = True
            self.power_restored.succeed()
            self.power_restored = self.env.event()
            print(f"[HW] Power RESTORED at {self.env.now:.2f}")
