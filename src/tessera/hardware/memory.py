import numpy as np

class NonVolatileMemory:
    """
    Simulates FRAM/MRAM. 
    Tracks 'leakage' (Hamming Weight) whenever a checkpoint is written.
    """

    def __init__(self):
        self.storage = {}
        self.leakage_trace = [] # Stores (time, power_consumption)

    def write_checkpoint(self, address: int, data: np.ndarray, time: float):
        """
        Writes data to NVM and logs side-channel leakage.
        
        Args:
            address: Logical ID for the data chunk.
            data: The numpy array containing intermediate crypto state.
            time: Current simulation time.
        """
        # Save data (Simulating persistence)
        self.storage[address] = data.copy()

        # TODO: Calculate leakage based on Hamming Weight of the data
        # leakage_val = hamming_weight(data)
        # self.leakage_trace.append((time, leakage_val))
        pass

    def read_checkpoint(self, address: int) -> np.ndarray:
        """
        Retrieves data if it exists.
        """
        return self.storage.get(address, None)
