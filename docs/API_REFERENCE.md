# Domain Processing API Reference

Complete API documentation for programmatic access to domain processing.

## Base URL

```
http://localhost:5000/process
```

---

## Endpoints

### 1. List Available Domains

Get all registered domains with metadata.

**Endpoint**: `GET /api/domain-list`

**Response**:
```json
{
  "domains": [
    {
      "name": "tennis",
      "pi": 0.75,
      "type": "Individual Sport",
      "data_exists": true,
      "data_path": "data/domains/tennis_complete_dataset.json",
      "estimated_timeout_minutes": 18,
      "outcome_type": "binary"
    }
  ],
  "total": 15,
  "available": 12
}
```

**Example**:
```bash
curl http://localhost:5000/process/api/domain-list
```

---

### 2. Start Processing Job

Start a new domain processing job.

**Endpoint**: `POST /api/process-domain`

**Request Body**:
```json
{
  "domain": "tennis",
  "sample_size": 1000,
  "timeout_minutes": 20,
  "fail_fast": false,
  "user": "researcher1"
}
```

**Parameters**:
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `domain` | string | Yes | - | Domain name |
| `sample_size` | integer | No | 1000 | Number of samples |
| `timeout_minutes` | integer | No | Auto | Timeout in minutes |
| `fail_fast` | boolean | No | false | Stop on first error |
| `user` | string | No | "anonymous" | User identifier |

**Response**:
```json
{
  "job_id": "a1b2c3d4",
  "domain": "tennis",
  "sample_size": 1000,
  "timeout_minutes": 18,
  "status": "queued",
  "message": "Processing started"
}
```

**Example**:
```bash
curl -X POST http://localhost:5000/process/api/process-domain \
  -H "Content-Type: application/json" \
  -d '{
    "domain": "tennis",
    "sample_size": 1000
  }'
```

**Python Example**:
```python
import requests

response = requests.post(
    'http://localhost:5000/process/api/process-domain',
    json={
        'domain': 'tennis',
        'sample_size': 1000,
        'timeout_minutes': 20
    }
)

job_data = response.json()
job_id = job_data['job_id']
print(f"Job started: {job_id}")
```

---

### 3. Get Processing Status

Get real-time status via Server-Sent Events (SSE).

**Endpoint**: `GET /api/process-status/<job_id>`

**Response** (Streaming):
```
data: {"job_id":"a1b2c3d4","status":"running","progress":0.45,"step":"feature_extraction","domain":"tennis","elapsed_seconds":120,"elapsed_formatted":"2.0min","eta_seconds":146,"eta_formatted":"2.4min"}

data: {"job_id":"a1b2c3d4","status":"running","progress":0.50,"step":"pattern_discovery",...}
```

**Status Values**:
- `queued` - Job created, waiting to start
- `running` - Currently processing
- `completed` - Finished successfully
- `failed` - Error occurred
- `cancelled` - User cancelled
- `timeout` - Exceeded time limit

**Example** (JavaScript):
```javascript
const eventSource = new EventSource(
  `/process/api/process-status/${jobId}`
);

eventSource.onmessage = (event) => {
  const status = JSON.parse(event.data);
  console.log(`Progress: ${status.progress * 100}%`);
  console.log(`Step: ${status.step}`);
  
  if (['completed', 'failed', 'cancelled', 'timeout'].includes(status.status)) {
    eventSource.close();
  }
};
```

**Python Example**:
```python
import sseclient
import requests

response = requests.get(
    f'http://localhost:5000/process/api/process-status/{job_id}',
    stream=True
)

client = sseclient.SSEClient(response)
for event in client.events():
    status = json.loads(event.data)
    print(f"Progress: {status['progress']*100:.0f}%")
    
    if status['status'] in ['completed', 'failed']:
        break
```

---

### 4. Get Processing Logs

Get processing logs for a job.

**Endpoint**: `GET /api/process-logs/<job_id>`

**Query Parameters**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `since_index` | integer | 0 | Get logs since this index |

**Response**:
```json
{
  "logs": [
    {
      "timestamp": "2025-11-17T14:30:45",
      "level": "INFO",
      "message": "Loading domain: tennis",
      "step": "loading",
      "domain": "tennis"
    }
  ],
  "total_count": 150,
  "returned_count": 50
}
```

**Example**:
```bash
curl http://localhost:5000/process/api/process-logs/a1b2c3d4?since_index=0
```

---

### 5. Cancel Job

Cancel a running job.

**Endpoint**: `POST /api/cancel-job/<job_id>`

**Response**:
```json
{
  "job_id": "a1b2c3d4",
  "status": "cancelled",
  "message": "Job cancelled successfully"
}
```

**Example**:
```bash
curl -X POST http://localhost:5000/process/api/cancel-job/a1b2c3d4
```

**Python Example**:
```python
response = requests.post(
    f'http://localhost:5000/process/api/cancel-job/{job_id}'
)
print(response.json())
```

---

### 6. Get Job History

Get history of all processing jobs.

**Endpoint**: `GET /api/job-history`

**Query Parameters**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | integer | 50 | Max jobs to return |
| `domain` | string | - | Filter by domain |

**Response**:
```json
{
  "jobs": [
    {
      "job_id": "a1b2c3d4",
      "domain": "tennis",
      "sample_size": 1000,
      "status": "completed",
      "start_time": "2025-11-17T14:30:00",
      "end_time": "2025-11-17T14:35:30",
      "duration_formatted": "5.5min",
      "user": "researcher1",
      "error_count": 0,
      "warning_count": 2,
      "results_path": "narrative_optimization/results/domains/tennis"
    }
  ],
  "count": 1
}
```

**Example**:
```bash
# All jobs
curl http://localhost:5000/process/api/job-history

# Filter by domain
curl http://localhost:5000/process/api/job-history?domain=tennis&limit=10
```

---

### 7. Get Job Statistics

Get overall job statistics.

**Endpoint**: `GET /api/job-stats`

**Response**:
```json
{
  "total": 45,
  "by_status": {
    "completed": 38,
    "failed": 3,
    "running": 1,
    "cancelled": 2,
    "timeout": 1
  },
  "success_rate": 0.927,
  "running_jobs": [
    {
      "job_id": "x1y2z3",
      "domain": "movies",
      "progress": 0.65,
      "elapsed_seconds": 180
    }
  ]
}
```

**Example**:
```bash
curl http://localhost:5000/process/api/job-stats
```

---

### 8. Get Results

Get processing results for a completed domain.

**Endpoint**: `GET /api/results/<domain>`

**Response**:
```json
{
  "job_id": "a1b2c3d4",
  "domain": "tennis",
  "completed_at": "2025-11-17T14:35:30",
  "results": {
    "domain": "tennis",
    "sample_size": 1000,
    "n_patterns": 23,
    "significant_correlations": [
      {
        "pattern_id": "pattern_12",
        "correlation": 0.45,
        "p_value": 0.001,
        "effect_size": 0.42,
        "description": "Underdog resilience pattern"
      }
    ],
    "discovery_summary": { ... },
    "validation_results": { ... }
  }
}
```

**Example**:
```bash
curl http://localhost:5000/process/api/results/tennis
```

---

## Error Responses

All endpoints return standard error format:

```json
{
  "error": "Error message describing what went wrong"
}
```

**HTTP Status Codes**:
- `200` - Success
- `400` - Bad request (invalid parameters)
- `404` - Not found (domain/job doesn't exist)
- `500` - Server error (internal error)

---

## Python SDK Example

Complete example of using the API programmatically:

```python
import requests
import json
import time
from typing import Dict, Optional

class DomainProcessorClient:
    """Client for Domain Processing API."""
    
    def __init__(self, base_url='http://localhost:5000/process'):
        self.base_url = base_url
    
    def list_domains(self) -> Dict:
        """Get all available domains."""
        response = requests.get(f'{self.base_url}/api/domain-list')
        return response.json()
    
    def start_processing(
        self,
        domain: str,
        sample_size: int = 1000,
        timeout_minutes: Optional[int] = None,
        fail_fast: bool = False
    ) -> str:
        """
        Start processing job.
        
        Returns job_id
        """
        response = requests.post(
            f'{self.base_url}/api/process-domain',
            json={
                'domain': domain,
                'sample_size': sample_size,
                'timeout_minutes': timeout_minutes,
                'fail_fast': fail_fast
            }
        )
        data = response.json()
        return data['job_id']
    
    def get_status(self, job_id: str) -> Dict:
        """Get current job status (one-time, not streaming)."""
        # For streaming, use SSE client
        response = requests.get(
            f'{self.base_url}/api/job-history?limit=100'
        )
        jobs = response.json()['jobs']
        
        for job in jobs:
            if job['job_id'] == job_id:
                return job
        
        return None
    
    def wait_for_completion(
        self,
        job_id: str,
        poll_interval: int = 5,
        callback=None
    ) -> Dict:
        """
        Wait for job to complete.
        
        Parameters
        ----------
        job_id : str
            Job ID to wait for
        poll_interval : int
            Seconds between status checks
        callback : callable, optional
            Called with status dict on each update
        
        Returns
        -------
        final_status : dict
            Final job status
        """
        while True:
            status = self.get_status(job_id)
            
            if callback:
                callback(status)
            
            if status['status'] in ['completed', 'failed', 'cancelled', 'timeout']:
                return status
            
            time.sleep(poll_interval)
    
    def cancel_job(self, job_id: str) -> Dict:
        """Cancel running job."""
        response = requests.post(
            f'{self.base_url}/api/cancel-job/{job_id}'
        )
        return response.json()
    
    def get_results(self, domain: str) -> Dict:
        """Get results for completed domain."""
        response = requests.get(
            f'{self.base_url}/api/results/{domain}'
        )
        return response.json()


# Usage example
client = DomainProcessorClient()

# List domains
domains = client.list_domains()
print(f"Available domains: {len(domains['available'])}")

# Start processing
job_id = client.start_processing('tennis', sample_size=1000)
print(f"Started job: {job_id}")

# Wait for completion with progress updates
def progress_callback(status):
    if status:
        print(f"Progress: {status.get('progress', 0)*100:.0f}%")

final_status = client.wait_for_completion(job_id, callback=progress_callback)
print(f"Final status: {final_status['status']}")

# Get results
if final_status['status'] == 'completed':
    results = client.get_results('tennis')
    print(f"Patterns discovered: {results['results']['n_patterns']}")
```

---

## Rate Limiting

Currently no rate limiting enforced. For production:
- Limit concurrent jobs per user
- Queue system for job scheduling
- Priority based on user tier

---

## Authentication

Current version: No authentication required (development mode)

For production:
- Add API key authentication
- User-based job isolation
- Permission system for domains

---

## WebSocket Alternative

For lower latency, consider WebSocket:

```javascript
const ws = new WebSocket('ws://localhost:5000/process/ws');

ws.send(JSON.stringify({
  action: 'start',
  domain: 'tennis',
  sample_size: 1000
}));

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  console.log(message);
};
```

---

## Best Practices

1. **Poll responsibly**: Use SSE for real-time updates instead of polling
2. **Handle errors**: Always check response status codes
3. **Timeout appropriately**: Set reasonable HTTP timeouts (>5min)
4. **Cache domain list**: No need to fetch repeatedly
5. **Validate before submitting**: Check domain exists before starting job

---

## Troubleshooting

### Connection Refused

**Problem**: Can't reach API

**Solution**: Ensure Flask app is running on correct port

### SSE Not Streaming

**Problem**: Events not received

**Solution**: 
- Check browser/library supports SSE
- Verify no proxy buffering
- Use appropriate SSE client library

### Job Stuck in Queue

**Problem**: Job never starts

**Solution**:
- Check system resources (memory, CPU)
- Verify no other jobs blocking
- Check Flask logs for errors

---

**Last Updated**: November 2025  
**Version**: 2.0

