# realm/gpu_monitor.py
import csv, os, threading, time

class GPUMonitor:
    """Sample GPU metrics in a background thread."""
    def __init__(self, interval=1.0, device=0):
        import pynvml

        self.nv = pynvml
        self.nv.nvmlInit()
        self.h = self.nv.nvmlDeviceGetHandleByIndex(device)
        self.interval = interval
        self.rows = []
        self.path = None
        self._f = self._w = None
        self._written = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self):
        nv, h = self.nv, self.h
        util = nv.nvmlDeviceGetUtilizationRates(h)
        mem = nv.nvmlDeviceGetMemoryInfo(h)
        return {
            "elapsed_s": time.perf_counter() - self._t0,
            "t": time.time(),
            "gpu_util_pct": util.gpu,
            "mem_bw_util_pct": util.memory,
            "mem_used_mb": mem.used / 2**20,
            "mem_total_mb": mem.total / 2**20,
            "power_w": nv.nvmlDeviceGetPowerUsage(h) / 1000,
            "temp_c": nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU),
            "sm_clock_mhz": nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM),
        }

    def _flush(self):
        n = len(self.rows)
        if self.path is None or self._written == n:
            return
        if self._f is None:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            self._f = open(self.path, "w", newline="")
            self._w = csv.DictWriter(self._f, fieldnames=list(self.rows[0]))
            self._w.writeheader()
        self._w.writerows(self.rows[self._written:n])
        self._f.flush()
        self._written = n

    def _run(self):
        while not self._stop.is_set():
            self.rows.append(self._sample())
            self._flush()
            self._stop.wait(self.interval)

    def start(self):
        self._t0 = time.perf_counter()
        self._thread.start()
        return self

    def set_path(self, path):
        self.path = path

    def stop(self):
        self._stop.set()
        self._thread.join()
        self._flush()
        if self._f:
            self._f.close()
        self.nv.nvmlShutdown()