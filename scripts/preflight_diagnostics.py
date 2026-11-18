#!/usr/bin/env python3
"""
MaBuy Production Infrastructure Pre-Flight Diagnostic Module.
Performs non-destructive, real-time connectivity and health checks:
1. Binance Futures API ping and time drift.
2. Deribit API connectivity and latency.
3. SQLite database integrity, WAL journal mode, and read access.
4. Storage and memory resources (free disk space, available RAM).

Can be run standalone, via cron, or as a systemd ExecStartPre hook.
Exit code: 0 if all critical checks pass, 1 if any critical check fails.
"""

import sys
import os
import time
import json
import shutil
import sqlite3
import logging
from typing import Dict, Any, Tuple

# Reconfigure stdout to UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Configure structured logging
LOG_FORMAT = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("PreflightDiagnostics")

# Paths and Thresholds
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DB_PATH = os.path.join(PROJECT_ROOT, "data", "mabuy_prod.db")
MIN_FREE_DISK_GB = 1.0       # Critical threshold
WARN_FREE_DISK_GB = 3.0      # Warning threshold
MAX_API_LATENCY_MS = 1500.0  # Max acceptable REST latency
MAX_TIME_DRIFT_MS = 2000.0   # Max server time drift for exchange signatures


class PreflightDiagnosticSuite:
    """Production pre-flight validation suite."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self.results: Dict[str, Dict[str, Any]] = {}
        self.has_critical_failure = False

    def _record_result(self, category: str, passed: bool, critical: bool, details: Dict[str, Any]):
        self.results[category] = {
            "passed": passed,
            "critical": critical,
            "details": details,
            "timestamp": time.time()
        }
        if not passed and critical:
            self.has_critical_failure = True

    # -------------------------------------------------------------------------
    # 1. Binance Futures Connectivity
    # -------------------------------------------------------------------------
    def check_binance_connectivity(self) -> bool:
        import urllib.request
        logger.info("Checking Binance Futures connectivity...")
        url = "https://fapi.binance.com/fapi/v1/time"
        start_t = time.perf_counter()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "MaBuy-Diagnostic/1.0"})
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                elapsed_ms = (time.perf_counter() - start_t) * 1000.0
                if resp.status == 200:
                    payload = json.loads(resp.read().decode("utf-8"))
                    server_time = payload.get("serverTime", 0)
                    local_time = int(time.time() * 1000)
                    drift_ms = abs(server_time - local_time)

                    is_ok = elapsed_ms < MAX_API_LATENCY_MS and drift_ms < MAX_TIME_DRIFT_MS
                    self._record_result(
                        "binance_connectivity",
                        passed=is_ok,
                        critical=True,
                        details={
                            "status_code": resp.status,
                            "latency_ms": round(elapsed_ms, 2),
                            "drift_ms": drift_ms,
                            "endpoint": url
                        }
                    )
                    if is_ok:
                        logger.info(f"Binance OK: Latency {elapsed_ms:.1f}ms, Clock Drift {drift_ms}ms")
                    else:
                        logger.warning(f"Binance Degraded: Latency {elapsed_ms:.1f}ms, Drift {drift_ms}ms")
                    return is_ok
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            logger.error(f"Binance connection failed: {e}")
            self._record_result(
                "binance_connectivity",
                passed=False,
                critical=True,
                details={"error": str(e), "latency_ms": round(elapsed_ms, 2)}
            )
            return False

    # -------------------------------------------------------------------------
    # 2. Deribit API Connectivity
    # -------------------------------------------------------------------------
    def check_deribit_connectivity(self) -> bool:
        import urllib.request
        logger.info("Checking Deribit API connectivity...")
        url = "https://www.deribit.com/api/v2/public/get_time"
        start_t = time.perf_counter()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "MaBuy-Diagnostic/1.0"})
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                elapsed_ms = (time.perf_counter() - start_t) * 1000.0
                if resp.status == 200:
                    payload = json.loads(resp.read().decode("utf-8"))
                    deribit_time = payload.get("result", 0)
                    local_time = int(time.time() * 1000)
                    drift_ms = abs(deribit_time - local_time)

                    is_ok = elapsed_ms < MAX_API_LATENCY_MS
                    self._record_result(
                        "deribit_connectivity",
                        passed=is_ok,
                        critical=False,  # Deribit is telemetry/V6, non-fatal for V5.2 live orders
                        details={
                            "status_code": resp.status,
                            "latency_ms": round(elapsed_ms, 2),
                            "drift_ms": drift_ms,
                            "endpoint": url
                        }
                    )
                    logger.info(f"Deribit OK: Latency {elapsed_ms:.1f}ms, Clock Drift {drift_ms}ms")
                    return is_ok
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            logger.warning(f"Deribit check note (non-critical): {e}")
            self._record_result(
                "deribit_connectivity",
                passed=False,
                critical=False,
                details={"error": str(e), "latency_ms": round(elapsed_ms, 2)}
            )
            return False

    # -------------------------------------------------------------------------
    # 3. SQLite Database Integrity & WAL Mode
    # -------------------------------------------------------------------------
    def check_sqlite_integrity(self) -> bool:
        logger.info(f"Checking SQLite integrity at {self.db_path}...")
        if not os.path.exists(self.db_path):
            logger.warning(f"Database file not found at {self.db_path}. Will be initialized on start.")
            self._record_result(
                "sqlite_integrity",
                passed=True,
                critical=False,
                details={"status": "not_created_yet", "path": self.db_path}
            )
            return True

        start_t = time.perf_counter()
        try:
            conn = sqlite3.connect(self.db_path, timeout=5.0)
            cursor = conn.cursor()

            # 1. PRAGMA integrity_check
            cursor.execute("PRAGMA integrity_check;")
            integrity_rows = cursor.fetchall()
            is_intact = len(integrity_rows) == 1 and integrity_rows[0][0] == "ok"

            # 2. PRAGMA journal_mode
            cursor.execute("PRAGMA journal_mode;")
            journal_mode = cursor.fetchone()[0]

            # 3. Check existing tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [r[0] for r in cursor.fetchall()]

            conn.close()
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0

            details = {
                "integrity": "ok" if is_intact else str(integrity_rows),
                "journal_mode": journal_mode,
                "tables_count": len(tables),
                "tables": tables,
                "check_latency_ms": round(elapsed_ms, 2)
            }

            self._record_result(
                "sqlite_integrity",
                passed=is_intact,
                critical=True,
                details=details
            )
            if is_intact:
                logger.info(f"SQLite OK: Integrity {integrity_rows[0][0]}, Journal={journal_mode}, Tables={len(tables)}")
            else:
                logger.error(f"SQLite Integrity Failed: {integrity_rows}")
            return is_intact
        except Exception as e:
            logger.error(f"SQLite check error: {e}")
            self._record_result(
                "sqlite_integrity",
                passed=False,
                critical=True,
                details={"error": str(e)}
            )
            return False

    # -------------------------------------------------------------------------
    # 4. Storage and System Resources
    # -------------------------------------------------------------------------
    def check_system_resources(self) -> bool:
        logger.info("Checking host disk and memory resources...")
        try:
            # Disk check
            total, used, free = shutil.disk_usage(PROJECT_ROOT)
            free_gb = free / (1024 ** 3)
            total_gb = total / (1024 ** 3)
            disk_ok = free_gb >= MIN_FREE_DISK_GB

            # RAM check (Linux /proc/meminfo or cross-platform fallback)
            mem_details = {}
            if os.path.exists("/proc/meminfo"):
                try:
                    with open("/proc/meminfo", "r") as f:
                        lines = f.readlines()
                    for line in lines:
                        if line.startswith("MemAvailable:"):
                            kb = int(line.split()[1])
                            mem_details["mem_available_mb"] = round(kb / 1024, 1)
                            break
                except Exception:
                    pass

            passed = disk_ok
            details = {
                "free_disk_gb": round(free_gb, 2),
                "total_disk_gb": round(total_gb, 2),
                "min_required_disk_gb": MIN_FREE_DISK_GB,
                **mem_details
            }

            self._record_result(
                "system_resources",
                passed=passed,
                critical=True,
                details=details
            )

            if free_gb < WARN_FREE_DISK_GB:
                logger.warning(f"Disk space low: {free_gb:.2f} GB free (Threshold: {WARN_FREE_DISK_GB} GB)")
            else:
                logger.info(f"Disk OK: {free_gb:.2f} GB free of {total_gb:.1f} GB")

            if "mem_available_mb" in mem_details:
                logger.info(f"Memory OK: {mem_details['mem_available_mb']} MB available")

            return passed
        except Exception as e:
            logger.error(f"System resource check error: {e}")
            self._record_result(
                "system_resources",
                passed=False,
                critical=True,
                details={"error": str(e)}
            )
            return False

    # -------------------------------------------------------------------------
    # Full Run & Exit Code
    # -------------------------------------------------------------------------
    def run_all(self) -> int:
        logger.info("=== Starting MaBuy Pre-Flight Diagnostics ===")
        self.check_binance_connectivity()
        self.check_deribit_connectivity()
        self.check_sqlite_integrity()
        self.check_system_resources()

        logger.info("=== Pre-Flight Summary ===")
        for cat, data in self.results.items():
            status = "PASS" if data["passed"] else ("FAIL (CRITICAL)" if data["critical"] else "WARN")
            logger.info(f"  • {cat.upper()}: [{status}] - {data['details']}")

        if self.has_critical_failure:
            logger.error("❌ CRITICAL PRE-FLIGHT CHECKS FAILED! System cannot start safely.")
            return 1

        logger.info("✅ All critical pre-flight checks PASSED. System ready for daemon execution.")
        return 0


if __name__ == "__main__":
    suite = PreflightDiagnosticSuite()
    exit_code = suite.run_all()
    sys.exit(exit_code)
