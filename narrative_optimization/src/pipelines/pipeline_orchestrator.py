"""
Pipeline Orchestrator with Comprehensive Monitoring

Wraps UniversalDomainProcessor with:
- Real-time progress tracking
- Error capture and reporting  
- Timeout monitoring
- Detailed logging (file + memory buffer)
- Graceful failure modes
- Status persistence

Author: Narrative Optimization Framework
Date: November 2025
"""

# FIX TENSORFLOW MUTEX DEADLOCK ON MACOS
# Must be set BEFORE any TensorFlow/transformers imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import sys
import time
import traceback
import logging
from pathlib import Path
from typing import Dict, Optional, Callable, List
from datetime import datetime
from threading import Thread, Event
from queue import Queue
import json

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from universal_domain_processor import UniversalDomainProcessor
from domain_registry import get_domain, DOMAINS


class LogBuffer(logging.Handler):
    """Custom logging handler that buffers logs in memory for web display."""
    
    def __init__(self, max_size=1000):
        super().__init__()
        self.logs = []
        self.max_size = max_size
    
    def emit(self, record):
        """Store log record in memory buffer."""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'message': self.format(record),
            'step': getattr(record, 'step', 'general'),
            'domain': getattr(record, 'domain', 'unknown')
        }
        self.logs.append(log_entry)
        
        # Keep buffer size manageable
        if len(self.logs) > self.max_size:
            self.logs = self.logs[-self.max_size:]
    
    def get_logs(self, since_index=0):
        """Get logs since index."""
        return self.logs[since_index:]
    
    def clear(self):
        """Clear log buffer."""
        self.logs = []


class PipelineOrchestrator:
    """
    Orchestrates domain processing with comprehensive monitoring.
    
    Features:
    - Real-time progress tracking
    - Error capture with full context
    - Timeout monitoring with warnings
    - Dual logging (file + memory buffer)
    - Graceful cancellation
    - State persistence
    """
    
    def __init__(
        self,
        results_dir='narrative_optimization/results/domains',
        logs_dir='logs/processing',
        fail_fast=False
    ):
        """
        Initialize orchestrator.
        
        Parameters
        ----------
        results_dir : str
            Directory for saving results
        logs_dir : str
            Directory for log files
        fail_fast : bool
            If True, stop on first error. If False, continue processing.
        """
        self.results_dir = Path(results_dir)
        self.logs_dir = Path(logs_dir)
        self.fail_fast = fail_fast
        
        # Create directories
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Progress tracking
        self.current_step = ""
        self.current_progress = 0.0
        self.total_steps = 0
        self.completed_steps = 0
        self.start_time = None
        self.end_time = None
        self.status = "idle"  # idle, running, completed, failed, cancelled, timeout
        
        # Error tracking
        self.errors = []
        self.warnings = []
        
        # Log buffer
        self.log_buffer = LogBuffer(max_size=2000)
        
        # Cancellation
        self.cancel_event = Event()
        
        # Results
        self.results = None
        
        # Setup logging
        self.logger = self._setup_logger()
    
    def _setup_logger(self, domain_name=None):
        """Setup dual logging (file + buffer)."""
        logger = logging.getLogger(f'orchestrator_{id(self)}')
        logger.setLevel(logging.DEBUG)
        logger.handlers = []  # Clear existing handlers
        
        # File handler (if domain specified)
        if domain_name:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = self.logs_dir / f'{domain_name}_{timestamp}.log'
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter(
                '[%(asctime)s] [%(levelname)s] [%(step)s] [%(domain)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        # Memory buffer handler
        buffer_handler = self.log_buffer
        buffer_handler.setLevel(logging.INFO)
        buffer_formatter = logging.Formatter('[%(levelname)s] %(message)s')
        buffer_handler.setFormatter(buffer_formatter)
        logger.addHandler(buffer_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('[%(levelname)s] %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def _log(self, level, message, step='general', domain='unknown'):
        """Log with extra context."""
        extra = {'step': step, 'domain': domain}
        self.logger.log(level, message, extra=extra)
    
    def process_domain_with_monitoring(
        self,
        domain_name: str,
        sample_size: Optional[int] = None,
        timeout_minutes: Optional[int] = None,
        progress_callback: Optional[Callable] = None
    ) -> Dict:
        """
        Process domain with full monitoring.
        
        Parameters
        ----------
        domain_name : str
            Domain to process
        sample_size : int, optional
            Number of samples to process
        timeout_minutes : int, optional
            Timeout in minutes (None = no timeout)
        progress_callback : callable, optional
            Callback for progress updates: callback(progress_dict)
        
        Returns
        -------
        result : dict
            Processing results with status, errors, metrics
        """
        # Reset state
        self.current_step = "initializing"
        self.current_progress = 0.0
        self.start_time = time.time()
        self.end_time = None
        self.status = "running"
        self.errors = []
        self.warnings = []
        self.results = None
        self.cancel_event.clear()
        
        # Setup domain-specific logger
        self.logger = self._setup_logger(domain_name)
        
        self._log(logging.INFO, f"Starting domain processing: {domain_name}", 
                  step='init', domain=domain_name)
        
        # Get domain config
        config = get_domain(domain_name)
        if not config:
            error_msg = f"Domain '{domain_name}' not registered"
            self._log(logging.ERROR, error_msg, step='init', domain=domain_name)
            self.status = "failed"
            self.errors.append({
                'step': 'init',
                'error': error_msg,
                'timestamp': datetime.now().isoformat()
            })
            return self._build_result()
        
        # Estimate timeout if not provided
        if timeout_minutes is None:
            timeout_minutes = self._estimate_timeout(sample_size or 1000)
        
        timeout_seconds = timeout_minutes * 60 if timeout_minutes else None
        
        self._log(logging.INFO, 
                  f"Configuration: sample_size={sample_size}, timeout={timeout_minutes}min, fail_fast={self.fail_fast}",
                  step='init', domain=domain_name)
        
        try:
            # Create processor
            processor = UniversalDomainProcessor(
                results_dir=str(self.results_dir),
                use_transformers=True,
                fast_mode=False
            )
            
            # Process with monitoring
            result = self._process_with_timeout(
                processor,
                domain_name,
                sample_size,
                timeout_seconds,
                progress_callback
            )
            
            self.results = result
            
            if self.cancel_event.is_set():
                self.status = "cancelled"
                self._log(logging.WARNING, "Processing cancelled by user", 
                          step='complete', domain=domain_name)
            elif self.status != "timeout":
                self.status = "completed"
                self._log(logging.INFO, "Processing completed successfully", 
                          step='complete', domain=domain_name)
            
        except Exception as e:
            self.status = "failed"
            error_msg = f"Fatal error: {str(e)}"
            stack_trace = traceback.format_exc()
            
            self._log(logging.ERROR, error_msg, step='processing', domain=domain_name)
            self._log(logging.DEBUG, stack_trace, step='processing', domain=domain_name)
            
            self.errors.append({
                'step': 'processing',
                'error': error_msg,
                'stack_trace': stack_trace,
                'timestamp': datetime.now().isoformat()
            })
        
        finally:
            self.end_time = time.time()
            self._log(logging.INFO, 
                      f"Total time: {self.end_time - self.start_time:.1f}s",
                      step='complete', domain=domain_name)
        
        return self._build_result()
    
    def _process_with_timeout(
        self,
        processor: UniversalDomainProcessor,
        domain_name: str,
        sample_size: Optional[int],
        timeout_seconds: Optional[float],
        progress_callback: Optional[Callable]
    ) -> Dict:
        """Process with timeout monitoring."""
        
        # Define progress update function
        def update_progress(step, percent, message):
            """Update progress and call callback."""
            self.current_step = step
            self.current_progress = percent
            
            self._log(logging.INFO, message, step=step, domain=domain_name)
            
            if progress_callback:
                progress_callback(self.get_progress())
            
            # Check for cancellation
            if self.cancel_event.is_set():
                raise InterruptedError("Processing cancelled by user")
            
            # Check for timeout
            if timeout_seconds:
                elapsed = time.time() - self.start_time
                if elapsed > timeout_seconds:
                    self.status = "timeout"
                    raise TimeoutError(f"Processing exceeded timeout of {timeout_seconds}s")
                
                # Warning at 80%
                if elapsed > timeout_seconds * 0.8 and not hasattr(self, '_timeout_warning_sent'):
                    self._timeout_warning_sent = True
                    self._log(logging.WARNING, 
                              f"Approaching timeout: {elapsed:.0f}s / {timeout_seconds:.0f}s",
                              step=self.current_step, domain=domain_name)
                    self.warnings.append({
                        'type': 'timeout_warning',
                        'message': f"80% of timeout reached",
                        'timestamp': datetime.now().isoformat()
                    })
        
        # Process domain with progress tracking
        update_progress('loading', 0.1, f'Loading domain: {domain_name}')
        
        result = processor.process_domain(
            domain_name=domain_name,
            sample_size=sample_size,
            save_results=True
        )
        
        update_progress('complete', 1.0, 'Processing complete')
        
        return result
    
    def _estimate_timeout(self, sample_size: int) -> int:
        """Estimate timeout in minutes based on sample size."""
        # Rough estimates: ~100 samples/minute
        base_minutes = max(10, sample_size / 100)
        # Add buffer
        return int(base_minutes * 1.5)
    
    def get_progress(self) -> Dict:
        """
        Get current progress.
        
        Returns
        -------
        progress : dict
            Current progress information
        """
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        progress = {
            'status': self.status,
            'step': self.current_step,
            'progress': self.current_progress,
            'elapsed_seconds': elapsed,
            'elapsed_formatted': self._format_time(elapsed),
            'errors_count': len(self.errors),
            'warnings_count': len(self.warnings)
        }
        
        # Add ETA if running
        if self.status == 'running' and self.current_progress > 0.1:
            estimated_total = elapsed / self.current_progress
            remaining = estimated_total - elapsed
            progress['eta_seconds'] = remaining
            progress['eta_formatted'] = self._format_time(remaining)
        
        return progress
    
    def get_logs(self, since_index=0) -> List[Dict]:
        """Get logs since index."""
        return self.log_buffer.get_logs(since_index)
    
    def cancel_processing(self):
        """Cancel processing gracefully."""
        self._log(logging.WARNING, "Cancellation requested", step='cancel', domain='system')
        self.cancel_event.set()
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds as human-readable time."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}min"
        else:
            return f"{seconds/3600:.1f}hr"
    
    def _build_result(self) -> Dict:
        """Build final result dictionary."""
        return {
            'status': self.status,
            'start_time': datetime.fromtimestamp(self.start_time).isoformat() if self.start_time else None,
            'end_time': datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else None,
            'elapsed_seconds': self.end_time - self.start_time if self.end_time and self.start_time else None,
            'errors': self.errors,
            'warnings': self.warnings,
            'results': self.results,
            'progress': self.get_progress()
        }
    
    def save_state(self, filepath: Path):
        """Save orchestrator state to file."""
        state = self._build_result()
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def estimate_time_remaining(self, domain_name: str, sample_size: int) -> Dict:
        """
        Estimate time required for processing.
        
        Parameters
        ----------
        domain_name : str
            Domain name
        sample_size : int
            Number of samples
        
        Returns
        -------
        estimate : dict
            Time estimates
        """
        # Base estimate: ~100 samples/minute with transformers
        # Varies by domain complexity (π)
        config = get_domain(domain_name)
        if config:
            pi = config.estimated_pi
            # Higher π = more complex = slower
            complexity_factor = 1.0 + (pi * 0.5)
        else:
            complexity_factor = 1.0
        
        base_minutes = (sample_size / 100) * complexity_factor
        
        return {
            'estimated_minutes': round(base_minutes, 1),
            'estimated_formatted': self._format_time(base_minutes * 60),
            'samples_per_minute': round(100 / complexity_factor, 1),
            'complexity_factor': round(complexity_factor, 2)
        }


if __name__ == '__main__':
    # Test orchestrator
    orchestrator = PipelineOrchestrator(fail_fast=False)
    
    # Test with small domain
    print("\nTesting orchestrator with tennis (100 samples)...")
    
    def progress_printer(progress):
        print(f"[{progress['progress']*100:.0f}%] {progress['step']}: {progress['elapsed_formatted']}")
    
    result = orchestrator.process_domain_with_monitoring(
        domain_name='tennis',
        sample_size=100,
        timeout_minutes=10,
        progress_callback=progress_printer
    )
    
    print(f"\nStatus: {result['status']}")
    print(f"Elapsed: {result['elapsed_seconds']:.1f}s")
    print(f"Errors: {len(result['errors'])}")
    print(f"Warnings: {len(result['warnings'])}")
    
    # Show logs
    logs = orchestrator.get_logs()
    print(f"\nLogs ({len(logs)} entries):")
    for log in logs[-10:]:
        print(f"  [{log['level']}] {log['message']}")

