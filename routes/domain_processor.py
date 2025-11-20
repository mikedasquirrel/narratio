"""
Domain Processing Flask Routes

Web interface for running domain processing pipelines with:
- Real-time progress tracking via Server-Sent Events
- Job management and history
- Error handling and logging
- Graceful cancellation

Author: Narrative Optimization Framework
Date: November 2025
"""

# FIX TENSORFLOW MUTEX DEADLOCK ON MACOS
# Must be set BEFORE any Flask/TensorFlow imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from flask import Blueprint, render_template, request, jsonify, Response, stream_with_context
import sys
from pathlib import Path
import json
import time
from threading import Thread
from typing import Dict, Optional
import logging
from datetime import datetime

# Add narrative_optimization to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'narrative_optimization'))

from src.pipelines.pipeline_orchestrator import PipelineOrchestrator
from src.pipelines.job_manager import JobManager, JobStatus
from src.pipelines.timeout_monitor import TimeoutMonitor
from domain_registry import DOMAINS, get_domain

domain_processor_bp = Blueprint('domain_processor', __name__)

# Initialize job manager
job_manager = JobManager()

# Active orchestrators (one per job)
active_orchestrators: Dict[str, PipelineOrchestrator] = {}

# Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def log_stage(message: str) -> None:
    """Console-friendly log helper with timestamps."""
    timestamp = datetime.now().isoformat(timespec='seconds')
    full_message = f"[domain_processor][{timestamp}] {message}"
    print(full_message)
    logger.info(message)


@domain_processor_bp.route('/domain-processor')
def dashboard():
    """Domain processing dashboard page."""
    log_stage("Rendering domain processor dashboard")
    return render_template('domain_processor_dashboard.html')


@domain_processor_bp.route('/api/domain-list')
def get_domain_list():
    """
    Get list of all registered domains with metadata.
    
    Returns
    -------
    domains : list
        List of domain information
    """
    domains = []
    
    for name, config in DOMAINS.items():
        # Check if data exists
        data_exists = config.data_path.exists()
        
        # Estimate processing time
        sample_size = 1000  # Default estimate
        timeout_estimate = TimeoutMonitor.calculate_timeout(
            sample_size=sample_size,
            domain_pi=config.estimated_pi
        )
        
        domains.append({
            'name': name,
            'pi': config.estimated_pi,
            'type': config.description or 'Unknown',
            'data_exists': data_exists,
            'data_path': str(config.data_path),
            'estimated_timeout_minutes': timeout_estimate,
            'outcome_type': config.outcome_type
        })
    
    # Sort by pi (narrativity)
    domains.sort(key=lambda d: d['pi'], reverse=True)
    
    payload = {
        'domains': domains,
        'total': len(domains),
        'available': sum(1 for d in domains if d['data_exists'])
    }
    log_stage(f"Domain list requested ({payload['available']}/{payload['total']} usable)")
    return jsonify(payload)


@domain_processor_bp.route('/api/process-domain', methods=['POST'])
def start_processing():
    """
    Start domain processing job.
    
    Request JSON:
    {
        "domain": "tennis",
        "sample_size": 1000,
        "timeout_minutes": 15,
        "fail_fast": false,
        "user": "username"
    }
    
    Returns
    -------
    response : dict
        Job information including job_id
    """
    data = request.get_json()
    
    domain = data.get('domain')
    sample_size = data.get('sample_size', 1000)
    timeout_minutes = data.get('timeout_minutes')
    fail_fast = data.get('fail_fast', False)
    user = data.get('user', 'anonymous')
    
    if not domain:
        return jsonify({'error': 'Domain name required'}), 400
    
    # Validate domain exists
    config = get_domain(domain)
    if not config:
        return jsonify({'error': f'Domain "{domain}" not found'}), 404
    
    # Calculate timeout if not provided
    if timeout_minutes is None:
        timeout_minutes = TimeoutMonitor.calculate_timeout(
            sample_size=sample_size,
            domain_pi=config.estimated_pi
        )
    
    # Create job
    job = job_manager.create_job(
        domain=domain,
        sample_size=sample_size,
        user=user,
        timeout_minutes=timeout_minutes,
        fail_fast=fail_fast
    )
    log_stage(f"Job created {job.job_id} for domain={domain}, sample={sample_size}, timeout={timeout_minutes}m, fail_fast={fail_fast}")
    
    # Start processing in background thread
    thread = Thread(
        target=_run_processing,
        args=(job.job_id, domain, sample_size, timeout_minutes, fail_fast),
        daemon=True
    )
    thread.start()
    
    return jsonify({
        'job_id': job.job_id,
        'domain': domain,
        'sample_size': sample_size,
        'timeout_minutes': timeout_minutes,
        'status': 'queued',
        'message': 'Processing started'
    })


def _run_processing(
    job_id: str,
    domain: str,
    sample_size: int,
    timeout_minutes: int,
    fail_fast: bool
):
    """Run processing in background thread."""
    log_stage(f"Job {job_id}: background thread starting for domain={domain}")
    try:
        # Mark job as started
        job_manager.start_job(job_id)
        log_stage(f"Job {job_id}: status set to RUNNING")
        
        # Create orchestrator
        orchestrator = PipelineOrchestrator(fail_fast=fail_fast)
        active_orchestrators[job_id] = orchestrator
        
        # Progress callback
        def progress_callback(progress_dict):
            # Update job in database
            job_manager.update_progress(
                job_id,
                progress_dict['progress'],
                progress_dict['step']
            )
        
        # Process domain
        result = orchestrator.process_domain_with_monitoring(
            domain_name=domain,
            sample_size=sample_size,
            timeout_minutes=timeout_minutes,
            progress_callback=progress_callback
        )
        
        # Update job based on result
        if result['status'] == 'completed':
            job_manager.complete_job(
                job_id,
                results_path=str(orchestrator.results_dir / domain),
                error_count=len(result['errors']),
                warning_count=len(result['warnings'])
            )
        elif result['status'] == 'timeout':
            job_manager.timeout_job(job_id)
        elif result['status'] == 'cancelled':
            job_manager.cancel_job(job_id)
        else:
            error_msg = result['errors'][0]['error'] if result['errors'] else 'Unknown error'
            job_manager.fail_job(job_id, error_msg)
    
    except Exception as e:
        logger.error(f"Error processing job {job_id}: {e}", exc_info=True)
        job_manager.fail_job(job_id, str(e))
        log_stage(f"Job {job_id}: FAILED with error {e}")
    
    finally:
        # Cleanup
        if job_id in active_orchestrators:
            del active_orchestrators[job_id]
        log_stage(f"Job {job_id}: thread finished (status={job_manager.get_job(job_id).status.value if job_manager.get_job(job_id) else 'unknown'})")


@domain_processor_bp.route('/api/process-status/<job_id>')
def get_process_status(job_id):
    """
    Get real-time processing status via Server-Sent Events.
    
    Returns
    -------
    stream : Response
        SSE stream with progress updates
    """
    def generate():
        """Generate SSE stream."""
        last_log_index = 0
        
        while True:
            # Get job from database
            job = job_manager.get_job(job_id)
            
            if not job:
                yield f"data: {json.dumps({'error': 'Job not found'})}\n\n"
                break
            
            # Get orchestrator if active
            orchestrator = active_orchestrators.get(job_id)
            
            # Build status update
            status = {
                'job_id': job_id,
                'status': job.status.value,
                'progress': job.progress,
                'step': job.current_step,
                'domain': job.domain
            }
            
            # Add timing info if job started
            if job.start_time:
                duration = job.get_duration()
                status['elapsed_seconds'] = duration
                status['elapsed_formatted'] = _format_time(duration) if duration else 'N/A'
                
                # Add ETA if running and we have progress
                if job.status == JobStatus.RUNNING and job.progress > 0.1:
                    estimated_total = duration / job.progress
                    remaining = estimated_total - duration
                    status['eta_seconds'] = remaining
                    status['eta_formatted'] = _format_time(remaining)
            
            # Add recent logs if orchestrator active
            if orchestrator:
                new_logs = orchestrator.get_logs(since_index=last_log_index)
                if new_logs:
                    status['logs'] = new_logs
                    last_log_index += len(new_logs)
            
            # Send update
            yield f"data: {json.dumps(status)}\n\n"
            
            # Stop streaming if job finished
            if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, 
                             JobStatus.CANCELLED, JobStatus.TIMEOUT]:
                break
            
            # Wait before next update
            time.sleep(1)
    
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )


@domain_processor_bp.route('/api/process-logs/<job_id>')
def get_process_logs(job_id):
    """
    Get processing logs.
    
    Parameters
    ----------
    since_index : int, optional
        Get logs since this index (query parameter)
    
    Returns
    -------
    logs : dict
        Log entries
    """
    since_index = request.args.get('since_index', 0, type=int)
    
    orchestrator = active_orchestrators.get(job_id)
    
    if not orchestrator:
        return jsonify({'error': 'Job not found or not active'}), 404
    
    logs = orchestrator.get_logs(since_index=since_index)
    
    return jsonify({
        'logs': logs,
        'total_count': len(orchestrator.log_buffer.logs),
        'returned_count': len(logs)
    })


@domain_processor_bp.route('/api/cancel-job/<job_id>', methods=['POST'])
def cancel_job(job_id):
    """
    Cancel running job.
    
    Returns
    -------
    response : dict
        Cancellation status
    """
    job = job_manager.get_job(job_id)
    
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    
    if job.status != JobStatus.RUNNING:
        return jsonify({'error': f'Job is {job.status.value}, cannot cancel'}), 400
    
    # Cancel via orchestrator
    orchestrator = active_orchestrators.get(job_id)
    if orchestrator:
        orchestrator.cancel_processing()
    
    # Update job status
    job_manager.cancel_job(job_id)
    
    return jsonify({
        'job_id': job_id,
        'status': 'cancelled',
        'message': 'Job cancelled successfully'
    })


@domain_processor_bp.route('/api/job-history')
def get_job_history():
    """
    Get job history.
    
    Parameters
    ----------
    limit : int, optional
        Number of jobs to return (query parameter)
    domain : str, optional
        Filter by domain (query parameter)
    
    Returns
    -------
    jobs : dict
        Job history
    """
    limit = request.args.get('limit', 50, type=int)
    domain = request.args.get('domain')
    
    if domain:
        jobs = job_manager.get_jobs_by_domain(domain, limit=limit)
    else:
        jobs = job_manager.get_all_jobs(limit=limit)
    
    jobs_data = []
    for job in jobs:
        job_dict = job.to_dict()
        
        # Add formatted duration
        duration = job.get_duration()
        if duration:
            job_dict['duration_formatted'] = _format_time(duration)
        
        jobs_data.append(job_dict)
    
    return jsonify({
        'jobs': jobs_data,
        'count': len(jobs_data)
    })


@domain_processor_bp.route('/api/job-stats')
def get_job_stats():
    """
    Get job statistics.
    
    Returns
    -------
    stats : dict
        Job statistics
    """
    stats = job_manager.get_statistics()
    
    # Add running jobs
    running_jobs = job_manager.get_running_jobs()
    stats['running_jobs'] = [job.to_dict() for job in running_jobs]
    
    return jsonify(stats)


@domain_processor_bp.route('/api/results/<domain>')
def get_results(domain):
    """
    Get processing results for a domain.
    
    Parameters
    ----------
    domain : str
        Domain name
    
    Returns
    -------
    results : dict
        Processing results
    """
    # Find most recent completed job for domain
    jobs = job_manager.get_jobs_by_domain(domain, limit=10)
    
    completed_job = None
    for job in jobs:
        if job.status == JobStatus.COMPLETED and job.results_path:
            completed_job = job
            break
    
    if not completed_job:
        return jsonify({'error': 'No completed results found for this domain'}), 404
    
    # Load results file
    results_path = Path(completed_job.results_path)
    
    # Try to find results JSON file
    result_files = list(results_path.glob('*_analysis.json'))
    
    if not result_files:
        return jsonify({'error': 'Results file not found'}), 404
    
    with open(result_files[0]) as f:
        results = json.load(f)
    
    return jsonify({
        'job_id': completed_job.job_id,
        'domain': domain,
        'completed_at': completed_job.end_time.isoformat() if completed_job.end_time else None,
        'results': results
    })


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

