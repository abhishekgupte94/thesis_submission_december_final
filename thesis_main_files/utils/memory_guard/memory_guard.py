# memory_guard.py

import psutil
import os
import time


class MemoryGuard:
    """
    Simple, reliable macOS-safe memory guard.

    - Checks process RSS memory (resident set size).
    - Optionally checks system-wide available memory.
    - If threshold exceeded → raises MemoryError or returns False.

    Recommended threshold for 16GB RAM:
        8–10 GB for safety.
    """

    def __init__(
        self,
        max_process_gb: float = 8.0,
        min_system_available_gb: float = 2.0,
        throws: bool = True,
    ):
        self.max_process = max_process_gb * (1024**3)
        self.min_system_free = min_system_available_gb * (1024**3)
        self.throws = throws

    def check(self):
        """
        Returns True if safe, otherwise raises or returns False.
        """
        proc = psutil.Process(os.getpid())
        rss = proc.memory_info().rss
        avail = psutil.virtual_memory().available

        if rss > self.max_process:
            msg = f"[MemoryGuard] PROCESS memory exceeded: {rss/1e9:.2f} GB > limit {self.max_process/1e9:.2f} GB"
            if self.throws:
                raise MemoryError(msg)
            print(msg)
            return False

        if avail < self.min_system_free:
            msg = f"[MemoryGuard] SYSTEM available memory dangerously low: {avail/1e9:.2f} GB < minimum {self.min_system_free/1e9:.2f} GB"
            if self.throws:
                raise MemoryError(msg)
            print(msg)
            return False

        return True
