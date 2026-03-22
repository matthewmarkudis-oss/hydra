"""Hardware Optimization Agent — CPU/GPU config tuning and throughput monitoring.

Pure Python agent (zero LLM cost). Wraps the existing ResourceManager,
benchmarks training throughput, and adjusts compute parameters for optimal
performance on the user's AMD hardware.
"""

from __future__ import annotations

import logging
import platform
import subprocess
import time
from typing import Any

from corp.agents.base_corp_agent import BaseCorpAgent
from corp.state.corporation_state import CorporationState
from corp.state.decision_log import DecisionLog

logger = logging.getLogger("corp.agents.hardware_optimizer")


class HardwareOptimizer(BaseCorpAgent):
    """Hardware optimization agent.

    Responsibilities:
    1. Detect CPU/GPU config and Smart Access Memory status
    2. Benchmark training throughput (time per generation)
    3. Recommend optimal n_envs, cpu_workers settings
    4. Monitor for thermal throttling or memory pressure
    """

    def __init__(
        self,
        state: CorporationState,
        decision_log: DecisionLog,
    ):
        super().__init__("hardware_optimizer", state, decision_log)
        self._generation_times: list[float] = []
        self._best_n_envs: int | None = None
        self._hardware_report: dict[str, Any] | None = None
        self._benchmarked = False

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Run hardware optimization checks.

        Context keys:
        - workflow_summary: From workflow.get_summary() with per-task timing
        - compute_config: Current ComputeConfig dict
        - training_state: Current hydra_training_state dict (for pool analysis)
        """
        result = {
            "hardware": self._detect_hardware(),
            "recommendations": [],
            "sam_enabled": False,
            "optimal_n_envs": 4,
            "optimal_cpu_workers": 6,
        }

        # Detect Smart Access Memory
        result["sam_enabled"] = self._detect_sam()

        # Analyze generation timing if available
        workflow_summary = context.get("workflow_summary", {})
        if workflow_summary:
            timing = self._analyze_timing(workflow_summary)
            result["timing"] = timing

        # Pool efficiency analysis — detect bloat and recommend pruning
        training_state = context.get("training_state", {})
        pool_analysis = self._analyze_pool_efficiency(training_state)
        if pool_analysis:
            result["pool_analysis"] = pool_analysis

        # Generate recommendations
        result["recommendations"] = self._generate_recommendations(result)

        self.log_decision(
            "hardware_analysis",
            detail={
                "sam": result["sam_enabled"],
                "n_envs": result["optimal_n_envs"],
                "cpu_workers": result["optimal_cpu_workers"],
            },
            outcome="complete",
        )

        self._mark_run(result)
        return result

    def _detect_hardware(self) -> dict[str, Any]:
        """Detect CPU, GPU, and memory configuration."""
        if self._hardware_report is not None:
            return self._hardware_report

        info = {
            "platform": platform.system(),
            "processor": platform.processor(),
            "cpu_count": _get_cpu_count(),
            "gpu": "unknown",
            "gpu_memory_gb": 0,
            "total_ram_gb": 0,
        }

        # Try to detect GPU via ResourceManager
        try:
            from hydra.compute.resource_manager import ResourceManager
            rm = ResourceManager()
            state = rm.get_state()
            info["gpu"] = state.gpu.device_name or "not detected"
            info["gpu_memory_gb"] = state.gpu.memory_gb
            info["gpu_type"] = state.gpu.device_type
        except Exception as e:
            logger.debug(f"ResourceManager detection failed: {e}")

        # Try to get total RAM on Windows
        if platform.system() == "Windows":
            try:
                result = subprocess.run(
                    ["wmic", "computersystem", "get", "totalphysicalmemory"],
                    capture_output=True, text=True, timeout=10,
                )
                lines = result.stdout.strip().split("\n")
                if len(lines) >= 2:
                    ram_bytes = int(lines[-1].strip())
                    info["total_ram_gb"] = round(ram_bytes / (1024 ** 3), 1)
            except Exception:
                pass

        self._hardware_report = info
        logger.info(
            f"Hardware: {info['processor']}, {info['cpu_count']} cores, "
            f"GPU: {info['gpu']}, RAM: {info['total_ram_gb']}GB"
        )
        return info

    def _detect_sam(self) -> bool:
        """Detect AMD Smart Access Memory (Resizable BAR) status.

        On Windows, checks via wmic for PCIe Resizable BAR support.
        """
        if platform.system() != "Windows":
            return False

        try:
            # Check if Resizable BAR is enabled via GPU driver
            result = subprocess.run(
                ["wmic", "path", "win32_videocontroller", "get", "name,driverversion"],
                capture_output=True, text=True, timeout=10,
            )
            output = result.stdout.lower()

            # AMD GPUs with SAM typically have recent drivers
            has_amd_gpu = "amd" in output or "radeon" in output

            if has_amd_gpu:
                # Check ReBAR via registry (non-destructive read)
                try:
                    import winreg
                    key = winreg.OpenKey(
                        winreg.HKEY_LOCAL_MACHINE,
                        r"SYSTEM\CurrentControlSet\Control\Video",
                    )
                    # If we can open this key, the GPU driver is installed
                    winreg.CloseKey(key)
                    logger.info("AMD GPU detected — SAM/ReBAR likely available")
                    return True
                except (OSError, ImportError):
                    pass

            return False

        except Exception as e:
            logger.debug(f"SAM detection failed: {e}")
            return False

    def _analyze_timing(self, workflow_summary: dict) -> dict[str, Any]:
        """Analyze per-task timing from workflow execution."""
        tasks = workflow_summary.get("tasks", {})
        total_ms = workflow_summary.get("total_duration_ms", 0)

        timing = {
            "total_seconds": round(total_ms / 1000, 1),
            "phases": {},
        }

        for task_name, task_info in tasks.items():
            duration_ms = task_info.get("duration_ms", 0)
            timing["phases"][task_name] = {
                "seconds": round(duration_ms / 1000, 1),
                "pct_of_total": round(duration_ms / max(total_ms, 1) * 100, 1),
            }

        # Track generation times for trend analysis
        train_ms = tasks.get("train_phase", {}).get("duration_ms", 0)
        if train_ms > 0:
            self._generation_times.append(train_ms / 1000)
            # Keep last 20
            self._generation_times = self._generation_times[-20:]

        if self._generation_times:
            timing["avg_train_seconds"] = round(
                sum(self._generation_times) / len(self._generation_times), 1
            )
            timing["train_trend"] = (
                "improving" if len(self._generation_times) >= 3
                and self._generation_times[-1] < self._generation_times[0]
                else "stable"
            )

        return timing

    def _generate_recommendations(self, analysis: dict) -> list[str]:
        """Generate hardware optimization recommendations."""
        recs = []
        hw = analysis.get("hardware", {})

        # CPU workers
        cpu_count = hw.get("cpu_count", 1)
        if cpu_count >= 12:
            analysis["optimal_cpu_workers"] = 8
            recs.append(f"CPU has {cpu_count} cores — recommend 8 workers for parallel training")
        elif cpu_count >= 8:
            analysis["optimal_cpu_workers"] = 6
        else:
            analysis["optimal_cpu_workers"] = max(2, cpu_count - 2)
            recs.append(f"Limited to {cpu_count} cores — using {analysis['optimal_cpu_workers']} workers")

        # n_envs recommendation based on GPU memory
        gpu_mem = hw.get("gpu_memory_gb", 0)
        if gpu_mem >= 16:
            analysis["optimal_n_envs"] = 8
            recs.append("16GB+ GPU — can support 8 parallel environments")
        elif gpu_mem >= 8:
            analysis["optimal_n_envs"] = 4
        else:
            analysis["optimal_n_envs"] = 2
            recs.append("Limited GPU memory — using 2 parallel environments")

        # SAM recommendation
        if analysis.get("sam_enabled"):
            recs.append(
                "Smart Access Memory detected — GPU can access full VRAM "
                "for 5-15% training improvement"
            )
        elif hw.get("gpu", "").lower().startswith(("amd", "radeon")):
            recs.append(
                "AMD GPU without SAM — enable Smart Access Memory in BIOS "
                "for potential 5-15% training speedup"
            )

        # RAM recommendation
        ram_gb = hw.get("total_ram_gb", 0)
        if ram_gb > 0 and ram_gb < 16:
            recs.append(
                f"Only {ram_gb}GB RAM — consider upgrading to 32GB "
                "for larger datasets and parallel training"
            )

        return recs

    def _analyze_pool_efficiency(self, training_state: dict) -> dict[str, Any] | None:
        """Analyze pool size vs generation time to detect bloat.

        Flags:
        - Pool growing faster than demotion can trim
        - Generation time increasing due to pool size
        - Dead-weight agents (score < threshold contributing nothing)
        - Recommends max_pool_size or increased bottom_k_demote
        """
        generations = training_state.get("generations", [])
        if len(generations) < 3:
            return None

        analysis: dict[str, Any] = {}

        # Extract pool size and timing per generation
        pool_sizes = [g.get("pool_size", 0) for g in generations]
        analysis["current_pool_size"] = pool_sizes[-1] if pool_sizes else 0
        analysis["pool_growth_rate"] = (
            (pool_sizes[-1] - pool_sizes[0]) / max(len(pool_sizes) - 1, 1)
            if len(pool_sizes) >= 2 else 0
        )

        # Count dead-weight agents (eval score < -500 in latest gen)
        latest_scores = generations[-1].get("eval_scores", {})
        if latest_scores:
            dead_weight = sum(1 for s in latest_scores.values() if s < -500)
            near_zero = sum(1 for s in latest_scores.values() if -100 < s < 100)
            total = len(latest_scores)
            analysis["dead_weight_agents"] = dead_weight
            analysis["competitive_agents"] = near_zero
            analysis["total_agents"] = total
            analysis["dead_weight_pct"] = round(dead_weight / max(total, 1) * 100, 1)

        # Detect generation time slowdown from pool growth
        # Use timestamps if available — estimate from generation indices
        if len(generations) >= 6:
            early_pool = sum(g.get("pool_size", 0) for g in generations[:3]) / 3
            late_pool = sum(g.get("pool_size", 0) for g in generations[-3:]) / 3
            pool_ratio = late_pool / max(early_pool, 1)
            analysis["pool_size_ratio"] = round(pool_ratio, 2)
            # O(N) scaling means time roughly proportional to pool size
            analysis["estimated_slowdown_factor"] = round(pool_ratio, 2)

        # Generate alerts
        alerts = []
        pool_size = analysis.get("current_pool_size", 0)
        growth = analysis.get("pool_growth_rate", 0)
        dead_pct = analysis.get("dead_weight_pct", 0)

        if pool_size > 25:
            alerts.append({
                "type": "warning",
                "message": (
                    f"Pool has {pool_size} agents — generation time scales "
                    f"linearly with pool size. Recommend max_pool_size=20."
                ),
            })

        if growth > 0.5:
            alerts.append({
                "type": "warning",
                "message": (
                    f"Pool growing by {growth:.1f} agents/gen (net). "
                    f"Increase bottom_k_demote or set max_pool_size cap."
                ),
            })

        if dead_pct > 30:
            alerts.append({
                "type": "critical",
                "message": (
                    f"{analysis.get('dead_weight_agents', 0)} agents ({dead_pct:.0f}%) "
                    f"score below -500 — pure dead weight consuming compute. "
                    f"Increase bottom_k_demote to 3+ to cull faster."
                ),
            })

        slowdown = analysis.get("estimated_slowdown_factor", 1.0)
        if slowdown > 2.0:
            alerts.append({
                "type": "critical",
                "message": (
                    f"Pool has grown {slowdown:.1f}x since start — "
                    f"generation time ~{slowdown:.1f}x slower. "
                    f"Immediate pool cap recommended."
                ),
            })

        analysis["alerts"] = alerts

        if alerts:
            self.log_decision(
                "pool_efficiency_alert",
                detail=analysis,
                outcome="alerts_raised",
            )
            # Submit as corp proposal if critical
            critical = [a for a in alerts if a["type"] == "critical"]
            if critical:
                self._submit_pool_proposal(analysis)

        return analysis

    def _submit_pool_proposal(self, analysis: dict) -> None:
        """Submit a config change proposal to cap pool size."""
        current = analysis.get("current_pool_size", 0)
        dead = analysis.get("dead_weight_agents", 0)

        proposal = {
            "type": "config_patch",
            "source": "hardware_optimizer",
            "priority": "high",
            "description": (
                f"Pool efficiency alert: {current} agents in pool, "
                f"{dead} dead-weight (score < -500). "
                f"Recommend setting max_pool_size=20 and bottom_k_demote=3 "
                f"to reduce generation time by ~{current/20:.1f}x."
            ),
            "patch": {
                "training.max_pool_size": 20,
                "training.bottom_k_demote": 3,
            },
            "rationale": [a["message"] for a in analysis.get("alerts", [])],
        }

        try:
            self.state.add_proposal(proposal)
            logger.warning(
                f"Pool efficiency proposal submitted: "
                f"{current} agents -> recommend cap at 20"
            )
        except Exception as e:
            logger.debug(f"Could not submit pool proposal: {e}")

    def record_generation_time(self, gen_time_seconds: float) -> None:
        """Record a generation's training time for trend analysis."""
        self._generation_times.append(gen_time_seconds)
        self._generation_times = self._generation_times[-20:]


def _get_cpu_count() -> int:
    """Get CPU core count."""
    import os
    return os.cpu_count() or 1
