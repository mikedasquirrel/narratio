"""
Timeout Monitoring System

Monitors processing time and provides:
- Configurable timeouts per domain
- Default timeout calculation based on sample size
- Warning at threshold (80%)
- Graceful shutdown on timeout
- Partial results preservation
- Detailed timeout reports

Author: Narrative Optimization Framework
Date: November 2025
"""

import time
import logging
from typing import Dict, Optional, Callable
from datetime import datetime
from pathlib import Path


class TimeoutMonitor:
    """
    Monitor processing time and enforce timeouts.
    
    Features:
    - Configurable timeout per domain
    - Smart defaults based on sample size
    - Warning callbacks at thresholds
    - Graceful timeout handling
    - Detailed timeout reports
    """
    
    # Default timeout configurations (minutes per 1000 samples)
    DEFAULT_TIMEOUTS = {
        'low_pi': 10,      # π < 0.3 (simple domains)
        'medium_pi': 15,   # π 0.3-0.7 (moderate domains)
        'high_pi': 25,     # π > 0.7 (complex domains)
        'minimum': 5,      # Absolute minimum
        'maximum': 120     # Absolute maximum (2 hours)
    }
    
    WARNING_THRESHOLDS = [0.5, 0.8, 0.9, 0.95]  # Warning at 50%, 80%, 90%, 95%
    
    def __init__(
        self,
        timeout_minutes: Optional[int] = None,
        warning_callback: Optional[Callable] = None,
        timeout_callback: Optional[Callable] = None
    ):
        """
        Initialize timeout monitor.
        
        Parameters
        ----------
        timeout_minutes : int, optional
            Timeout in minutes (None = auto-calculate)
        warning_callback : callable, optional
            Called when approaching timeout: callback(percent, message)
        timeout_callback : callable, optional
            Called when timeout reached: callback(report)
        """
        self.timeout_minutes = timeout_minutes
        self.timeout_seconds = timeout_minutes * 60 if timeout_minutes else None
        self.warning_callback = warning_callback
        self.timeout_callback = timeout_callback
        
        self.start_time = None
        self.warnings_sent = set()
        self.timed_out = False
        self.current_step = None
        
        self.logger = logging.getLogger(__name__)
    
    def start(self):
        """Start monitoring."""
        self.start_time = time.time()
        self.warnings_sent = set()
        self.timed_out = False
        self.logger.info(f"Timeout monitor started (limit: {self.timeout_minutes}min)")
    
    def check(self, current_step: Optional[str] = None) -> bool:
        """
        Check if timeout reached.
        
        Parameters
        ----------
        current_step : str, optional
            Current processing step
        
        Returns
        -------
        continue_processing : bool
            True if should continue, False if timeout
        """
        if not self.start_time or not self.timeout_seconds:
            return True  # No timeout set
        
        if self.timed_out:
            return False  # Already timed out
        
        self.current_step = current_step or self.current_step
        
        elapsed = time.time() - self.start_time
        percent_elapsed = elapsed / self.timeout_seconds
        
        # Check for warnings
        for threshold in self.WARNING_THRESHOLDS:
            if percent_elapsed >= threshold and threshold not in self.warnings_sent:
                self._send_warning(threshold, elapsed)
                self.warnings_sent.add(threshold)
        
        # Check for timeout
        if elapsed >= self.timeout_seconds:
            self._handle_timeout(elapsed)
            return False
        
        return True
    
    def _send_warning(self, threshold: float, elapsed: float):
        """Send warning at threshold."""
        percent = int(threshold * 100)
        remaining = self.timeout_seconds - elapsed
        
        message = (f"Timeout warning: {percent}% of time elapsed "
                  f"({self._format_time(elapsed)} / {self._format_time(self.timeout_seconds)}), "
                  f"{self._format_time(remaining)} remaining")
        
        self.logger.warning(message)
        
        if self.warning_callback:
            self.warning_callback(threshold, message)
    
    def _handle_timeout(self, elapsed: float):
        """Handle timeout."""
        self.timed_out = True
        
        report = self.get_timeout_report(elapsed)
        
        self.logger.error(f"TIMEOUT: Processing exceeded limit of {self.timeout_minutes}min")
        self.logger.error(f"Current step when timeout: {self.current_step or 'unknown'}")
        
        if self.timeout_callback:
            self.timeout_callback(report)
    
    def get_elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if not self.start_time:
            return 0.0
        return time.time() - self.start_time
    
    def get_remaining(self) -> Optional[float]:
        """Get remaining time in seconds (None if no timeout)."""
        if not self.timeout_seconds:
            return None
        
        elapsed = self.get_elapsed()
        return max(0, self.timeout_seconds - elapsed)
    
    def get_percent_elapsed(self) -> Optional[float]:
        """Get percent of timeout elapsed (None if no timeout)."""
        if not self.timeout_seconds:
            return None
        
        elapsed = self.get_elapsed()
        return min(1.0, elapsed / self.timeout_seconds)
    
    def get_timeout_report(self, elapsed: Optional[float] = None) -> Dict:
        """
        Get detailed timeout report.
        
        Returns
        -------
        report : dict
            Timeout information
        """
        if elapsed is None:
            elapsed = self.get_elapsed()
        
        return {
            'timed_out': self.timed_out,
            'timeout_minutes': self.timeout_minutes,
            'elapsed_seconds': elapsed,
            'elapsed_formatted': self._format_time(elapsed),
            'current_step': self.current_step,
            'warnings_sent': sorted(list(self.warnings_sent)),
            'percent_elapsed': self.get_percent_elapsed(),
            'timestamp': datetime.now().isoformat()
        }
    
    @classmethod
    def calculate_timeout(
        cls,
        sample_size: int,
        domain_pi: float = 0.5,
        custom_timeout: Optional[int] = None
    ) -> int:
        """
        Calculate appropriate timeout in minutes.
        
        Parameters
        ----------
        sample_size : int
            Number of samples to process
        domain_pi : float
            Domain narrativity (affects complexity)
        custom_timeout : int, optional
            Override with custom timeout
        
        Returns
        -------
        timeout_minutes : int
            Calculated timeout in minutes
        """
        if custom_timeout:
            return custom_timeout
        
        # Determine base timeout by π
        if domain_pi < 0.3:
            base_per_1000 = cls.DEFAULT_TIMEOUTS['low_pi']
        elif domain_pi < 0.7:
            base_per_1000 = cls.DEFAULT_TIMEOUTS['medium_pi']
        else:
            base_per_1000 = cls.DEFAULT_TIMEOUTS['high_pi']
        
        # Scale by sample size
        timeout = int((sample_size / 1000) * base_per_1000)
        
        # Apply bounds
        timeout = max(cls.DEFAULT_TIMEOUTS['minimum'], timeout)
        timeout = min(cls.DEFAULT_TIMEOUTS['maximum'], timeout)
        
        return timeout
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds as human-readable time."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}min"
        else:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            return f"{hours}h {minutes}min"


class TimeoutConfig:
    """
    Configuration for domain-specific timeouts.
    
    Can be stored in domain config files.
    """
    
    def __init__(self, config_file: Optional[Path] = None):
        """
        Initialize timeout configuration.
        
        Parameters
        ----------
        config_file : Path, optional
            Path to JSON config file
        """
        self.timeouts = {}
        self.config_file = config_file
        
        if config_file and config_file.exists():
            self.load(config_file)
    
    def set_timeout(self, domain: str, timeout_minutes: int):
        """Set timeout for specific domain."""
        self.timeouts[domain] = timeout_minutes
    
    def get_timeout(self, domain: str, default: Optional[int] = None) -> Optional[int]:
        """Get timeout for domain."""
        return self.timeouts.get(domain, default)
    
    def load(self, config_file: Path):
        """Load timeouts from JSON file."""
        import json
        with open(config_file) as f:
            data = json.load(f)
            self.timeouts = data.get('timeouts', {})
    
    def save(self, config_file: Path):
        """Save timeouts to JSON file."""
        import json
        with open(config_file, 'w') as f:
            json.dump({'timeouts': self.timeouts}, f, indent=2)


if __name__ == '__main__':
    # Test timeout monitor
    print("Testing Timeout Monitor\n")
    
    # Test 1: Calculate timeouts
    print("1. Timeout Calculations:")
    for pi, pi_name in [(0.2, 'Low'), (0.5, 'Medium'), (0.8, 'High')]:
        for samples in [100, 1000, 5000]:
            timeout = TimeoutMonitor.calculate_timeout(samples, pi)
            print(f"  {pi_name} π ({pi:.1f}), {samples} samples: {timeout}min")
    
    # Test 2: Monitor with warnings
    print("\n2. Warning System:")
    
    def warning_handler(threshold, message):
        print(f"  WARNING: {message}")
    
    def timeout_handler(report):
        print(f"  TIMEOUT: {report['elapsed_formatted']} at step '{report['current_step']}'")
    
    monitor = TimeoutMonitor(
        timeout_minutes=0.05,  # 3 seconds for testing
        warning_callback=warning_handler,
        timeout_callback=timeout_handler
    )
    
    monitor.start()
    
    # Simulate processing
    steps = ['loading', 'transforming', 'validating', 'saving']
    for i, step in enumerate(steps):
        time.sleep(1)
        if monitor.check(step):
            print(f"  Step {i+1}/4: {step} ({monitor.get_elapsed():.1f}s elapsed)")
        else:
            print(f"  Processing stopped due to timeout")
            break
    
    print("\n3. Final Report:")
    report = monitor.get_timeout_report()
    for key, value in report.items():
        print(f"  {key}: {value}")

