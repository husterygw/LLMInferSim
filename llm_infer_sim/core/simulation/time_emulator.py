"""VirtualTimeEmulator — Phase 1 spike 最小版（time.sleep）。"""
import time


class VirtualTimeEmulator:
    def __init__(self, mode: str = "instant"):
        self.mode = mode

    def simulate(self, latency_seconds: float) -> None:
        if self.mode == "instant":
            return
        time.sleep(latency_seconds)
