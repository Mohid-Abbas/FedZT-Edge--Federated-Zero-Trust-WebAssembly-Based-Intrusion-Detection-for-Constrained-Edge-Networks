"""
Layer 5: Measurement Infrastructure and Energy Modeling (FedZT-Edge)
Instruments latency (mean, p95), memory, binary size, Wasm fuel, communication,
and computes state-aware energy (Joules/round) across Raspberry Pi 4, Jetson Xavier NX, and Desktop.
"""

import time
import numpy as np

try:
    from .config import HARDWARE_TIERS
except (ImportError, ValueError):
    from config import HARDWARE_TIERS

class MetricsTracker:
    def __init__(self, hardware_tier="Raspberry_Pi_4"):
        if hardware_tier not in HARDWARE_TIERS:
            raise ValueError(f"Unknown hardware tier: {hardware_tier}")
        self.tier_name = hardware_tier
        self.profile = HARDWARE_TIERS[hardware_tier]
        self.latencies = []
        self.fuel_records = []
        self.bytes_transferred = 0
        self.round_times = []

    def record_inference(self, latency_ms, fuel_consumed):
        self.latencies.append(latency_ms)
        self.fuel_records.append(fuel_consumed)

    def record_network_transfer(self, num_bytes):
        self.bytes_transferred += num_bytes

    def record_round_time(self, round_duration_sec):
        self.round_times.append(round_duration_sec)

    def compute_latency_stats(self):
        if not self.latencies:
            return {"mean_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "max_ms": 0.0}
        arr = np.array(self.latencies)
        return {
            "mean_ms": float(np.mean(arr)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "max_ms": float(np.max(arr))
        }

    def compute_energy_joules(self, active_duration_sec, total_round_sec=None):
        """
        State-aware energy model:
        E = P_idle * T_total + (P_peak - P_idle) * T_active + Fuel * fuel_rate
        """
        if total_round_sec is None:
            total_round_sec = active_duration_sec
            
        p_idle = self.profile["idle_power_w"]
        p_peak = self.profile["peak_power_w"]
        fuel_rate = self.profile["fuel_rate"]
        
        total_fuel = sum(self.fuel_records)
        active_energy = (p_peak - p_idle) * active_duration_sec
        idle_energy = p_idle * total_round_sec
        fuel_energy = total_fuel * fuel_rate
        
        total_joules = idle_energy + active_energy + fuel_energy
        return {
            "tier": self.tier_name,
            "total_joules": float(total_joules),
            "idle_joules": float(idle_energy),
            "active_joules": float(active_energy),
            "fuel_joules": float(fuel_energy),
            "active_sec": float(active_duration_sec)
        }

    def summary(self):
        lat_stats = self.compute_latency_stats()
        avg_round_sec = np.mean(self.round_times) if self.round_times else 1.0
        energy = self.compute_energy_joules(avg_round_sec)
        
        return {
            "tier": self.tier_name,
            "latency": lat_stats,
            "total_fuel": sum(self.fuel_records),
            "mb_transferred": self.bytes_transferred / (1024 * 1024),
            "energy_joules_per_round": energy["total_joules"]
        }
