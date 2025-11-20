"""
Job Management System

Tracks all domain processing jobs with:
- Unique job ID generation
- Job status tracking
- Metadata storage
- Results persistence
- Error logging
- Job history and cleanup

Author: Narrative Optimization Framework
Date: November 2025
"""

import uuid
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from enum import Enum


class JobStatus(str, Enum):
    """Job status enumeration."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class Job:
    """Represents a single processing job."""
    
    def __init__(
        self,
        job_id: str,
        domain: str,
        sample_size: int,
        status: JobStatus = JobStatus.QUEUED,
        **kwargs
    ):
        self.job_id = job_id
        self.domain = domain
        self.sample_size = sample_size
        self.status = status
        
        self.start_time = kwargs.get('start_time')
        self.end_time = kwargs.get('end_time')
        self.user = kwargs.get('user', 'anonymous')
        self.timeout_minutes = kwargs.get('timeout_minutes')
        self.fail_fast = kwargs.get('fail_fast', False)
        
        self.results_path = kwargs.get('results_path')
        self.error_message = kwargs.get('error_message')
        self.error_count = kwargs.get('error_count', 0)
        self.warning_count = kwargs.get('warning_count', 0)
        
        self.progress = kwargs.get('progress', 0.0)
        self.current_step = kwargs.get('current_step', '')
    
    def to_dict(self) -> Dict:
        """Convert job to dictionary."""
        return {
            'job_id': self.job_id,
            'domain': self.domain,
            'sample_size': self.sample_size,
            'status': self.status.value,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'user': self.user,
            'timeout_minutes': self.timeout_minutes,
            'fail_fast': self.fail_fast,
            'results_path': str(self.results_path) if self.results_path else None,
            'error_message': self.error_message,
            'error_count': self.error_count,
            'warning_count': self.warning_count,
            'progress': self.progress,
            'current_step': self.current_step
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Job':
        """Create job from dictionary."""
        # Convert datetime strings
        if data.get('start_time'):
            data['start_time'] = datetime.fromisoformat(data['start_time'])
        if data.get('end_time'):
            data['end_time'] = datetime.fromisoformat(data['end_time'])
        
        # Convert status
        if isinstance(data.get('status'), str):
            data['status'] = JobStatus(data['status'])
        
        return cls(**data)
    
    def get_duration(self) -> Optional[float]:
        """Get job duration in seconds."""
        if not self.start_time:
            return None
        
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()


class JobManager:
    """
    Manages all processing jobs.
    
    Features:
    - Job creation and tracking
    - Status updates
    - History management
    - Cleanup of old jobs
    - SQLite persistence
    """
    
    def __init__(self, db_path: str = 'narrative_optimization/jobs.db'):
        """
        Initialize job manager.
        
        Parameters
        ----------
        db_path : str
            Path to SQLite database
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                domain TEXT NOT NULL,
                sample_size INTEGER NOT NULL,
                status TEXT NOT NULL,
                start_time TEXT,
                end_time TEXT,
                user TEXT,
                timeout_minutes INTEGER,
                fail_fast BOOLEAN,
                results_path TEXT,
                error_message TEXT,
                error_count INTEGER DEFAULT 0,
                warning_count INTEGER DEFAULT 0,
                progress REAL DEFAULT 0.0,
                current_step TEXT,
                created_at TEXT NOT NULL
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_status ON jobs(status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_domain ON jobs(domain)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_created ON jobs(created_at)')
        
        conn.commit()
        conn.close()
    
    def create_job(
        self,
        domain: str,
        sample_size: int,
        user: str = 'anonymous',
        timeout_minutes: Optional[int] = None,
        fail_fast: bool = False
    ) -> Job:
        """
        Create new job.
        
        Parameters
        ----------
        domain : str
            Domain to process
        sample_size : int
            Number of samples
        user : str
            User who created job
        timeout_minutes : int, optional
            Timeout in minutes
        fail_fast : bool
            Fail on first error
        
        Returns
        -------
        job : Job
            Created job
        """
        job_id = str(uuid.uuid4())[:8]  # Short UUID
        
        job = Job(
            job_id=job_id,
            domain=domain,
            sample_size=sample_size,
            status=JobStatus.QUEUED,
            user=user,
            timeout_minutes=timeout_minutes,
            fail_fast=fail_fast
        )
        
        self._save_job(job)
        
        return job
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM jobs WHERE job_id = ?', (job_id,))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        return self._row_to_job(row)
    
    def update_job(self, job: Job):
        """Update job in database."""
        self._save_job(job)
    
    def start_job(self, job_id: str):
        """Mark job as started."""
        job = self.get_job(job_id)
        if job:
            job.status = JobStatus.RUNNING
            job.start_time = datetime.now()
            self.update_job(job)
    
    def complete_job(
        self,
        job_id: str,
        results_path: Optional[str] = None,
        error_count: int = 0,
        warning_count: int = 0
    ):
        """Mark job as completed."""
        job = self.get_job(job_id)
        if job:
            job.status = JobStatus.COMPLETED
            job.end_time = datetime.now()
            job.progress = 1.0
            job.results_path = results_path
            job.error_count = error_count
            job.warning_count = warning_count
            self.update_job(job)
    
    def fail_job(self, job_id: str, error_message: str):
        """Mark job as failed."""
        job = self.get_job(job_id)
        if job:
            job.status = JobStatus.FAILED
            job.end_time = datetime.now()
            job.error_message = error_message
            self.update_job(job)
    
    def cancel_job(self, job_id: str):
        """Mark job as cancelled."""
        job = self.get_job(job_id)
        if job:
            job.status = JobStatus.CANCELLED
            job.end_time = datetime.now()
            self.update_job(job)
    
    def timeout_job(self, job_id: str):
        """Mark job as timed out."""
        job = self.get_job(job_id)
        if job:
            job.status = JobStatus.TIMEOUT
            job.end_time = datetime.now()
            self.update_job(job)
    
    def update_progress(self, job_id: str, progress: float, current_step: str):
        """Update job progress."""
        job = self.get_job(job_id)
        if job:
            job.progress = progress
            job.current_step = current_step
            self.update_job(job)
    
    def get_all_jobs(self, limit: int = 100) -> List[Job]:
        """Get all jobs (most recent first)."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?',
            (limit,)
        )
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_job(row) for row in rows]
    
    def get_jobs_by_status(self, status: JobStatus, limit: int = 100) -> List[Job]:
        """Get jobs by status."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT * FROM jobs WHERE status = ? ORDER BY created_at DESC LIMIT ?',
            (status.value, limit)
        )
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_job(row) for row in rows]
    
    def get_jobs_by_domain(self, domain: str, limit: int = 50) -> List[Job]:
        """Get jobs for specific domain."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT * FROM jobs WHERE domain = ? ORDER BY created_at DESC LIMIT ?',
            (domain, limit)
        )
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_job(row) for row in rows]
    
    def get_running_jobs(self) -> List[Job]:
        """Get all running jobs."""
        return self.get_jobs_by_status(JobStatus.RUNNING)
    
    def cleanup_old_jobs(self, days: int = 30):
        """
        Delete jobs older than specified days.
        
        Parameters
        ----------
        days : int
            Delete jobs older than this many days
        
        Returns
        -------
        deleted_count : int
            Number of jobs deleted
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute(
            'DELETE FROM jobs WHERE created_at < ?',
            (cutoff_date.isoformat(),)
        )
        
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        return deleted_count
    
    def get_statistics(self) -> Dict:
        """Get job statistics."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Total jobs
        cursor.execute('SELECT COUNT(*) FROM jobs')
        total = cursor.fetchone()[0]
        
        # By status
        stats = {'total': total, 'by_status': {}}
        for status in JobStatus:
            cursor.execute(
                'SELECT COUNT(*) FROM jobs WHERE status = ?',
                (status.value,)
            )
            count = cursor.fetchone()[0]
            stats['by_status'][status.value] = count
        
        # Success rate
        completed = stats['by_status'].get('completed', 0)
        failed = stats['by_status'].get('failed', 0)
        if completed + failed > 0:
            stats['success_rate'] = completed / (completed + failed)
        else:
            stats['success_rate'] = 0.0
        
        conn.close()
        
        return stats
    
    def _save_job(self, job: Job):
        """Save job to database."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO jobs VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        ''', (
            job.job_id,
            job.domain,
            job.sample_size,
            job.status.value,
            job.start_time.isoformat() if job.start_time else None,
            job.end_time.isoformat() if job.end_time else None,
            job.user,
            job.timeout_minutes,
            job.fail_fast,
            str(job.results_path) if job.results_path else None,
            job.error_message,
            job.error_count,
            job.warning_count,
            job.progress,
            job.current_step,
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def _row_to_job(self, row: sqlite3.Row) -> Job:
        """Convert database row to Job."""
        data = dict(row)
        
        # Convert timestamps
        if data.get('start_time'):
            data['start_time'] = datetime.fromisoformat(data['start_time'])
        if data.get('end_time'):
            data['end_time'] = datetime.fromisoformat(data['end_time'])
        
        # Convert status
        data['status'] = JobStatus(data['status'])
        
        return Job(**{k: v for k, v in data.items() if k != 'created_at'})


if __name__ == '__main__':
    # Test job manager
    print("Testing Job Manager\n")
    
    # Create manager
    manager = JobManager(db_path='test_jobs.db')
    
    # Create jobs
    print("1. Creating jobs:")
    job1 = manager.create_job('tennis', 1000, user='test_user')
    print(f"  Created: {job1.job_id} - {job1.domain}")
    
    job2 = manager.create_job('movies', 2000, user='test_user')
    print(f"  Created: {job2.job_id} - {job2.domain}")
    
    # Start job
    print("\n2. Starting job:")
    manager.start_job(job1.job_id)
    print(f"  Started: {job1.job_id}")
    
    # Update progress
    print("\n3. Updating progress:")
    manager.update_progress(job1.job_id, 0.5, 'feature_extraction')
    print(f"  Progress: 50%")
    
    # Complete job
    print("\n4. Completing job:")
    manager.complete_job(job1.job_id, results_path='results/tennis.json')
    print(f"  Completed: {job1.job_id}")
    
    # Get statistics
    print("\n5. Statistics:")
    stats = manager.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Get all jobs
    print("\n6. All jobs:")
    all_jobs = manager.get_all_jobs()
    for job in all_jobs:
        duration = job.get_duration()
        duration_str = f"{duration:.1f}s" if duration else "N/A"
        print(f"  {job.job_id}: {job.domain} - {job.status.value} ({duration_str})")
    
    # Cleanup
    import os
    os.remove('test_jobs.db')
    print("\nTest complete!")

