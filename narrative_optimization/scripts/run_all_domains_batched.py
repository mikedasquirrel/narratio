"""
Run All Domains in Batches - Master Orchestrator

Executes all 18 domains in 6 manageable batches to prevent timeouts.
Features:
- Batch processing (2-3 domains per batch)
- Checkpoint system (resume from last successful)
- Force cache clearing between batches
- Progress tracking
- Detailed reporting

Usage:
    # Run all batches
    python run_all_domains_batched.py

    # Run specific batch
    python run_all_domains_batched.py --batch batch_1_control

    # Resume from specific batch
    python run_all_domains_batched.py --resume-from batch_3_mid_pi

Author: Narrative Integration System
Date: November 2025
"""

import sys
import json
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
import time
import argparse
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project root
project_root = Path(__file__).parent.parent.parent


def load_status():
    """Load execution status"""
    status_file = project_root / 'narrative_optimization' / 'BATCH_EXECUTION_STATUS.json'
    with open(status_file, 'r') as f:
        return json.load(f)


def save_status(status):
    """Save execution status"""
    status_file = project_root / 'narrative_optimization' / 'BATCH_EXECUTION_STATUS.json'
    status['last_updated'] = datetime.now().isoformat()
    with open(status_file, 'w') as f:
        json.dump(status, f, indent=2)


def clear_cache():
    """Clear feature cache to force recomputation"""
    cache_dir = project_root / 'narrative_optimization' / 'data' / 'features' / 'cache'
    if cache_dir.exists():
        logger.info("Clearing feature cache...")
        shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True)
        logger.info("✓ Cache cleared")
    else:
        logger.info("No cache to clear")


def process_domain(domain_name, timeout_minutes=30):
    """
    Process a single domain by calling process_single_domain.py
    
    Parameters
    ----------
    domain_name : str
        Domain name to process
    timeout_minutes : int
        Timeout in minutes
    
    Returns
    -------
    success : bool
        Whether processing succeeded
    results : dict
        Processing results
    """
    logger.info(f"\n{'=' * 80}")
    logger.info(f"STARTING DOMAIN: {domain_name}")
    logger.info(f"{'=' * 80}")
    
    # Call process_single_domain.py
    script_path = project_root / 'narrative_optimization' / 'scripts' / 'process_single_domain.py'
    
    cmd = [
        sys.executable,
        str(script_path),
        '--domain', domain_name,
        '--force-recompute',
        '--skip-on-error',
        '--timeout', str(timeout_minutes)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_minutes * 60 + 60  # Add 1 min buffer
        )
        
        # Load results
        results_file = project_root / 'narrative_optimization' / 'data' / 'features' / f'{domain_name}_processing_results.json'
        if results_file.exists():
            with open(results_file, 'r') as f:
                results = json.load(f)
        else:
            results = {
                'domain': domain_name,
                'status': 'failed',
                'error': 'No results file generated'
            }
        
        success = results.get('status') == 'success'
        
        if success:
            logger.info(f"✅ {domain_name} completed successfully")
        else:
            logger.error(f"❌ {domain_name} failed: {results.get('error', 'Unknown error')}")
        
        return success, results
        
    except subprocess.TimeoutExpired:
        logger.error(f"❌ {domain_name} timed out after {timeout_minutes} minutes")
        return False, {
            'domain': domain_name,
            'status': 'failed',
            'error': f'Timeout after {timeout_minutes} minutes'
        }
    except Exception as e:
        logger.error(f"❌ {domain_name} failed with exception: {str(e)}")
        return False, {
            'domain': domain_name,
            'status': 'failed',
            'error': str(e)
        }


def process_batch(batch_id, status, force_recompute=True):
    """
    Process a single batch of domains
    
    Parameters
    ----------
    batch_id : str
        Batch identifier
    status : dict
        Current execution status
    force_recompute : bool
        Whether to clear cache before batch
    
    Returns
    -------
    batch_results : dict
        Batch processing results
    """
    batch_config = status['batches'][batch_id]
    
    logger.info("\n" + "=" * 80)
    logger.info(f"BATCH: {batch_config['name'].upper()}")
    logger.info(f"Domains: {', '.join(batch_config['domains'])}")
    logger.info(f"Rationale: {batch_config['rationale']}")
    logger.info("=" * 80)
    
    # Clear cache if force recompute
    if force_recompute:
        clear_cache()
    
    # Update batch status
    batch_config['status'] = 'running'
    batch_config['started_at'] = datetime.now().isoformat()
    save_status(status)
    
    batch_start_time = time.time()
    domain_results = []
    successful_domains = 0
    failed_domains = 0
    
    # Process each domain in batch
    for domain_name in batch_config['domains']:
        # Update domain status
        status['domains'][domain_name]['status'] = 'running'
        status['domains'][domain_name]['started_at'] = datetime.now().isoformat()
        status['current_domain'] = domain_name
        save_status(status)
        
        # Process domain
        success, results = process_domain(domain_name, timeout_minutes=30)
        
        # Update domain status
        if success:
            successful_domains += 1
            status['domains'][domain_name]['status'] = 'completed'
            status['domains'][domain_name]['transformers_completed'] = results.get('transformers_completed', 0)
            status['domains'][domain_name]['total_features'] = results.get('total_features', 0)
            status['domains'][domain_name]['duration_seconds'] = results.get('duration_seconds', 0)
        else:
            failed_domains += 1
            status['domains'][domain_name]['status'] = 'failed'
            status['domains'][domain_name]['error'] = results.get('error', 'Unknown error')
        
        status['domains'][domain_name]['completed_at'] = datetime.now().isoformat()
        save_status(status)
        
        domain_results.append(results)
    
    # Update batch status
    batch_duration = time.time() - batch_start_time
    batch_config['status'] = 'completed' if failed_domains == 0 else 'partial'
    batch_config['completed_at'] = datetime.now().isoformat()
    batch_config['duration_seconds'] = batch_duration
    status['current_domain'] = None
    save_status(status)
    
    # Print batch summary
    logger.info("\n" + "=" * 80)
    logger.info(f"BATCH COMPLETE: {batch_config['name']}")
    logger.info(f"  Successful: {successful_domains}/{len(batch_config['domains'])}")
    logger.info(f"  Failed: {failed_domains}/{len(batch_config['domains'])}")
    logger.info(f"  Duration: {batch_duration/60:.1f} minutes")
    logger.info("=" * 80 + "\n")
    
    return {
        'batch_id': batch_id,
        'batch_name': batch_config['name'],
        'successful': successful_domains,
        'failed': failed_domains,
        'total': len(batch_config['domains']),
        'duration_seconds': batch_duration,
        'domain_results': domain_results
    }


def run_all_batches(resume_from=None, specific_batch=None, force_recompute=True):
    """
    Run all batches or resume from a specific batch
    
    Parameters
    ----------
    resume_from : str, optional
        Batch ID to resume from
    specific_batch : str, optional
        Run only this specific batch
    force_recompute : bool
        Whether to clear cache before each batch
    
    Returns
    -------
    execution_results : dict
        Complete execution results
    """
    logger.info("\n" + "=" * 80)
    logger.info("BATCHED DOMAIN EXECUTION - MASTER ORCHESTRATOR")
    logger.info("=" * 80)
    logger.info(f"Mode: {'Specific batch' if specific_batch else 'Resume' if resume_from else 'Full run'}")
    logger.info(f"Force recompute: {force_recompute}")
    logger.info("=" * 80 + "\n")
    
    # Load status
    status = load_status()
    execution_start_time = time.time()
    
    # Determine which batches to run
    batch_order = [
        'batch_1_control',
        'batch_2_low_pi',
        'batch_3_mid_pi',
        'batch_4_sports',
        'batch_5_subjective',
        'batch_6_identity'
    ]
    
    if specific_batch:
        if specific_batch not in batch_order:
            logger.error(f"Invalid batch ID: {specific_batch}")
            return None
        batches_to_run = [specific_batch]
    elif resume_from:
        if resume_from not in batch_order:
            logger.error(f"Invalid batch ID: {resume_from}")
            return None
        start_idx = batch_order.index(resume_from)
        batches_to_run = batch_order[start_idx:]
    else:
        batches_to_run = batch_order
    
    logger.info(f"Batches to run: {len(batches_to_run)}")
    for batch_id in batches_to_run:
        logger.info(f"  - {batch_id}: {status['batches'][batch_id]['name']}")
    logger.info("")
    
    # Process each batch
    batch_results = []
    total_successful = 0
    total_failed = 0
    
    for batch_id in batches_to_run:
        # Update current batch
        status['current_batch'] = batch_id
        save_status(status)
        
        # Process batch
        result = process_batch(batch_id, status, force_recompute=force_recompute)
        batch_results.append(result)
        
        total_successful += result['successful']
        total_failed += result['failed']
    
    # Final summary
    total_duration = time.time() - execution_start_time
    
    # Update summary
    status['current_batch'] = None
    status['summary']['completed_domains'] = total_successful
    status['summary']['failed_domains'] = total_failed
    status['summary']['total_duration_seconds'] = total_duration
    save_status(status)
    
    logger.info("\n" + "=" * 80)
    logger.info("🎉 ALL BATCHES COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total domains processed: {total_successful + total_failed}")
    logger.info(f"  ✅ Successful: {total_successful}")
    logger.info(f"  ❌ Failed: {total_failed}")
    logger.info(f"Total duration: {total_duration/60:.1f} minutes ({total_duration/3600:.1f} hours)")
    logger.info("=" * 80 + "\n")
    
    # Generate execution report
    execution_report = {
        'execution_mode': 'specific' if specific_batch else 'resume' if resume_from else 'full',
        'started_at': datetime.fromtimestamp(execution_start_time).isoformat(),
        'completed_at': datetime.now().isoformat(),
        'total_duration_seconds': total_duration,
        'batches_run': len(batches_to_run),
        'total_domains': total_successful + total_failed,
        'successful_domains': total_successful,
        'failed_domains': total_failed,
        'batch_results': batch_results
    }
    
    # Save execution report
    report_file = project_root / 'narrative_optimization' / 'data' / 'features' / 'execution_report.json'
    with open(report_file, 'w') as f:
        json.dump(execution_report, f, indent=2)
    
    logger.info(f"Execution report saved: {report_file}\n")
    
    return execution_report


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Run all domains in batches')
    parser.add_argument('--batch', type=str, help='Run specific batch only')
    parser.add_argument('--resume-from', type=str, help='Resume from specific batch')
    parser.add_argument('--no-force-recompute', action='store_true', help='Do not clear cache')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.batch and args.resume_from:
        logger.error("Cannot specify both --batch and --resume-from")
        sys.exit(1)
    
    # Run batches
    force_recompute = not args.no_force_recompute
    
    results = run_all_batches(
        resume_from=args.resume_from,
        specific_batch=args.batch,
        force_recompute=force_recompute
    )
    
    if results is None:
        sys.exit(1)
    
    # Exit with appropriate code
    exit_code = 0 if results['failed_domains'] == 0 else 1
    sys.exit(exit_code)


if __name__ == '__main__':
    main()

