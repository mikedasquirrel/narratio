"""
Progress Tracking System

Real-time experiment monitoring with metrics, time estimates, and resource usage.
"""

from typing import Dict, Any, Optional, List
import time
import psutil
import sys
from datetime import datetime, timedelta


class ProgressTracker:
    """
    Tracks and displays progress during experiments.
    
    Provides real-time updates on:
    - Overall progress
    - Current step/operation
    - Time elapsed and estimated remaining
    - Intermediate metrics
    - Resource usage (memory, CPU)
    
    Parameters
    ----------
    total_steps : int
        Total number of steps in the operation
    description : str
        Overall operation description
    show_metrics : bool
        Whether to display intermediate metrics
    show_resources : bool
        Whether to show resource usage
    update_interval : float
        Minimum seconds between display updates
    """
    
    def __init__(
        self,
        total_steps: int,
        description: str = "Processing",
        show_metrics: bool = True,
        show_resources: bool = True,
        update_interval: float = 0.5
    ):
        self.total_steps = total_steps
        self.description = description
        self.show_metrics = show_metrics
        self.show_resources = show_resources
        self.update_interval = update_interval
        
        self.current_step = 0
        self.start_time = time.time()
        self.last_update = 0
        self.step_times: List[float] = []
        self.metrics_history: List[Dict[str, float]] = []
        self.current_metrics: Dict[str, float] = {}
        
        self.process = psutil.Process()
        
    def update(self, step: int, status: str = "", metrics: Optional[Dict[str, float]] = None):
        """
        Update progress to a specific step.
        
        Parameters
        ----------
        step : int
            Current step number (0-indexed)
        status : str
            Current status message
        metrics : dict, optional
            Current metric values to display
        """
        self.current_step = step
        current_time = time.time()
        
        # Track step timing
        if self.step_times:
            step_duration = current_time - self.last_update
            self.step_times.append(step_duration)
        
        self.last_update = current_time
        
        # Update metrics
        if metrics:
            self.current_metrics = metrics
            self.metrics_history.append({
                'step': step,
                'time': current_time,
                **metrics
            })
        
        # Display update
        if current_time - getattr(self, '_last_display', 0) >= self.update_interval:
            self._display_progress(status)
            self._last_display = current_time
    
    def increment(self, status: str = "", metrics: Optional[Dict[str, float]] = None):
        """
        Increment progress by one step.
        
        Parameters
        ----------
        status : str
            Current status message
        metrics : dict, optional
            Current metric values
        """
        self.update(self.current_step + 1, status, metrics)
    
    def _display_progress(self, status: str):
        """Display current progress."""
        elapsed = time.time() - self.start_time
        progress = self.current_step / self.total_steps if self.total_steps > 0 else 0
        
        # Progress bar
        bar_length = 40
        filled = int(bar_length * progress)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        # Time estimates
        if self.current_step > 0:
            avg_step_time = elapsed / self.current_step
            remaining_steps = self.total_steps - self.current_step
            eta_seconds = avg_step_time * remaining_steps
            eta_str = str(timedelta(seconds=int(eta_seconds)))
        else:
            eta_str = "calculating..."
        
        # Format elapsed time
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        
        # Build display string
        display = f"\r{self.description}: |{bar}| {self.current_step}/{self.total_steps} "
        display += f"[{progress*100:.1f}%] "
        display += f"Elapsed: {elapsed_str} ETA: {eta_str}"
        
        if status:
            display += f" | {status}"
        
        # Add metrics
        if self.show_metrics and self.current_metrics:
            metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in self.current_metrics.items())
            display += f" | {metrics_str}"
        
        # Add resource usage
        if self.show_resources:
            memory_mb = self.process.memory_info().rss / 1024 / 1024
            cpu_percent = self.process.cpu_percent()
            display += f" | Mem: {memory_mb:.0f}MB CPU: {cpu_percent:.0f}%"
        
        # Print (with carriage return for overwriting)
        sys.stdout.write('\r' + ' ' * 200)  # Clear line
        sys.stdout.write(display)
        sys.stdout.flush()
    
    def finish(self, final_message: str = "Complete"):
        """
        Mark progress as complete.
        
        Parameters
        ----------
        final_message : str
            Final completion message
        """
        self.current_step = self.total_steps
        self._display_progress(final_message)
        print()  # New line after completion
        
        # Print summary
        elapsed = time.time() - self.start_time
        print(f"\n✓ {self.description} completed in {timedelta(seconds=int(elapsed))}")
        
        if self.metrics_history:
            print("\nFinal Metrics:")
            final_metrics = self.metrics_history[-1]
            for key, value in final_metrics.items():
                if key not in ['step', 'time']:
                    print(f"  {key}: {value:.4f}")
    
    def get_metrics_history(self) -> List[Dict[str, Any]]:
        """Get complete metrics history."""
        return self.metrics_history
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        elapsed = time.time() - self.start_time
        
        summary = {
            'total_steps': self.total_steps,
            'completed_steps': self.current_step,
            'elapsed_seconds': elapsed,
            'avg_step_time': elapsed / self.current_step if self.current_step > 0 else 0,
            'progress_percent': (self.current_step / self.total_steps * 100) if self.total_steps > 0 else 0
        }
        
        # Add final metrics
        if self.metrics_history:
            final_metrics = self.metrics_history[-1]
            summary['final_metrics'] = {k: v for k, v in final_metrics.items() if k not in ['step', 'time']}
        
        return summary


class ExperimentProgressTracker:
    """
    Specialized tracker for narrative experiments.
    
    Tracks progress across multiple narratives and cross-validation folds.
    
    Parameters
    ----------
    n_narratives : int
        Number of narratives being compared
    n_folds : int
        Number of CV folds
    metrics : list of str
        Metric names to track
    """
    
    def __init__(
        self,
        n_narratives: int,
        n_folds: int,
        metrics: List[str]
    ):
        self.n_narratives = n_narratives
        self.n_folds = n_folds
        self.metrics = metrics
        
        self.total_operations = n_narratives * n_folds
        self.current_narrative = 0
        self.current_fold = 0
        
        self.narrative_names: List[str] = []
        self.narrative_scores: Dict[str, List[float]] = {metric: [] for metric in metrics}
        self.best_scores: Dict[str, float] = {metric: -float('inf') for metric in metrics}
        
        self.tracker = ProgressTracker(
            total_steps=self.total_operations,
            description="Running Experiment",
            show_metrics=True,
            show_resources=True
        )
        
        self.start_time = time.time()
    
    def start_narrative(self, narrative_name: str):
        """Start processing a new narrative."""
        self.current_narrative += 1
        self.current_fold = 0
        self.narrative_names.append(narrative_name)
        
        print(f"\n\n{'='*80}")
        print(f"Narrative {self.current_narrative}/{self.n_narratives}: {narrative_name}")
        print(f"{'='*80}")
    
    def start_fold(self, fold: int):
        """Start a new CV fold."""
        self.current_fold = fold
        status = f"Narrative {self.current_narrative}/{self.n_narratives}, Fold {fold+1}/{self.n_folds}"
        
        step = (self.current_narrative - 1) * self.n_folds + fold
        self.tracker.update(step, status)
    
    def update_fold_metrics(self, fold: int, scores: Dict[str, float]):
        """Update metrics after fold completion."""
        # Track scores
        for metric, score in scores.items():
            if metric in self.narrative_scores:
                self.narrative_scores[metric].append(score)
                
                # Update best
                if score > self.best_scores[metric]:
                    self.best_scores[metric] = score
        
        # Display update with current scores
        status = f"Fold {fold+1}/{self.n_folds} complete"
        display_metrics = {k: v for k, v in scores.items() if k in self.metrics}
        
        step = (self.current_narrative - 1) * self.n_folds + fold + 1
        self.tracker.update(step, status, display_metrics)
    
    def finish_narrative(self, cv_results: Dict[str, Any]):
        """Finish current narrative and display summary."""
        print(f"\n\nNarrative '{self.narrative_names[-1]}' Complete:")
        
        for metric in self.metrics:
            if metric in cv_results:
                mean_score = cv_results[metric].get('test_mean', 0)
                std_score = cv_results[metric].get('test_std', 0)
                print(f"  {metric}: {mean_score:.4f} ± {std_score:.4f}")
                
                # Check if this is best
                if mean_score >= self.best_scores[metric]:
                    print(f"    ★ New best {metric}!")
    
    def finish(self):
        """Complete the experiment."""
        self.tracker.finish("All narratives evaluated")
        
        elapsed = time.time() - self.start_time
        
        print(f"\n{'='*80}")
        print("EXPERIMENT COMPLETE")
        print(f"{'='*80}")
        print(f"Total time: {timedelta(seconds=int(elapsed))}")
        print(f"Narratives tested: {self.n_narratives}")
        print(f"CV folds per narrative: {self.n_folds}")
        print(f"\nBest Scores:")
        for metric, score in self.best_scores.items():
            print(f"  {metric}: {score:.4f}")


def create_progress_bar(iterable, description="Processing", show_metrics=True):
    """
    Wrap an iterable with a progress bar.
    
    Parameters
    ----------
    iterable : iterable
        Items to iterate over
    description : str
        Description of operation
    show_metrics : bool
        Whether to show metrics
    
    Yields
    ------
    item
        Items from iterable with progress tracking
    
    Example
    -------
    >>> for item in create_progress_bar(items, "Processing items"):
    ...     process(item)
    """
    items = list(iterable)
    tracker = ProgressTracker(
        total_steps=len(items),
        description=description,
        show_metrics=show_metrics
    )
    
    for i, item in enumerate(items):
        tracker.update(i, f"Item {i+1}/{len(items)}")
        yield item
    
    tracker.finish()


if __name__ == '__main__':
    # Demo
    print("Progress Tracker Demo\n")
    
    # Simple progress bar
    tracker = ProgressTracker(100, "Demo Task")
    for i in range(100):
        time.sleep(0.02)
        tracker.update(i+1, f"Step {i+1}", {'accuracy': 0.7 + i*0.002})
    tracker.finish()
    
    # Experiment tracker
    print("\n\nExperiment Tracker Demo\n")
    exp_tracker = ExperimentProgressTracker(
        n_narratives=3,
        n_folds=5,
        metrics=['accuracy', 'f1']
    )
    
    for n in range(3):
        exp_tracker.start_narrative(f"Narrative_{n+1}")
        for f in range(5):
            exp_tracker.start_fold(f)
            time.sleep(0.1)
            exp_tracker.update_fold_metrics(f, {
                'accuracy': 0.7 + np.random.rand() * 0.2,
                'f1': 0.65 + np.random.rand() * 0.25
            })
        
        exp_tracker.finish_narrative({
            'accuracy': {'test_mean': 0.8, 'test_std': 0.05},
            'f1': {'test_mean': 0.75, 'test_std': 0.06}
        })
    
    exp_tracker.finish()

