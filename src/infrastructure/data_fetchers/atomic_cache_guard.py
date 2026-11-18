"""
MaBuy V4 — Data Engineering Department
AtomicCacheGuard: Cross-Platform Inter-Process File Lock & Thundering Herd Shield.
Guarantees single-flight cache updates across multiple OS processes (API, Bot, Daemon).
"""
import os
import time
import errno
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class LockAcquisitionTimeout(Exception):
    """Raised when a process fails to acquire the atomic lock within the specified timeout."""
    pass


class AtomicCacheGuard:
    """
    Cross-platform, multi-process file lock context manager with stale lock auto-reclamation.
    """

    def __init__(
        self,
        target_file_path: str,
        timeout: float = 5.0,
        stale_lock_max_age: float = 10.0,
        poll_interval: float = 0.05
    ):
        self.target_file_path = target_file_path
        self.lock_file_path = target_file_path + ".lock"
        self.timeout = timeout
        self.stale_lock_max_age = stale_lock_max_age
        self.poll_interval = poll_interval
        self._lock_fd: Optional[int] = None
        self._is_locked = False

    def _is_stale(self) -> bool:
        """Checks if the existing lock file is older than stale_lock_max_age."""
        try:
            mtime = os.path.getmtime(self.lock_file_path)
            return (time.time() - mtime) > self.stale_lock_max_age
        except OSError:
            return False

    def _break_stale_lock(self):
        """Removes a stale lock left by a dead process."""
        try:
            if os.path.exists(self.lock_file_path):
                os.remove(self.lock_file_path)
                logger.warning(f"Reclaimed stale lock for {self.target_file_path}")
        except OSError as e:
            logger.debug(f"Could not remove stale lock: {e}")

    def acquire(self) -> bool:
        start_time = time.time()
        
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(os.path.abspath(self.lock_file_path)), exist_ok=True)

        while True:
            try:
                # O_CREAT | O_EXCL is an atomic kernel-level operation on both POSIX and Windows
                self._lock_fd = os.open(
                    self.lock_file_path,
                    os.O_CREAT | os.O_EXCL | os.O_RDWR
                )
                # Write current PID into lock file for debugging
                try:
                    os.write(self._lock_fd, f"{os.getpid()}:{time.time()}".encode("utf-8"))
                except OSError:
                    pass
                self._is_locked = True
                return True
            except OSError as e:
                if e.errno in (errno.EEXIST, errno.EACCES, 13, 17, 80) or isinstance(e, PermissionError): # File exists / Lock collision / Windows EACCES
                    # Check for stale lock
                    if self._is_stale():
                        self._break_stale_lock()

                    if (time.time() - start_time) >= self.timeout:
                        raise LockAcquisitionTimeout(
                            f"Timed out after {self.timeout}s waiting for lock: {self.lock_file_path}"
                        )
                    time.sleep(self.poll_interval)
                else:
                    raise e

    def release(self):
        if self._is_locked:
            if self._lock_fd is not None:
                try:
                    os.close(self._lock_fd)
                except OSError:
                    pass
                self._lock_fd = None

            try:
                if os.path.exists(self.lock_file_path):
                    os.remove(self.lock_file_path)
            except OSError:
                pass
            self._is_locked = False

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
