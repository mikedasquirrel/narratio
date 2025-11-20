# Setup Guide

**Complete setup guide for the narrative optimization system**

---

## Quick Setup (5 minutes)

```bash
# 1. Clone/navigate to project
cd narrative_optimization

# 2. Install dependencies
pip install -r requirements.txt

# 3. Initialize system
python INITIALIZE_SYSTEM.py

# 4. Run demo
python DEMO_COMPLETE_SYSTEM.py

# 5. Add your first domain
python MASTER_INTEGRATION.py my_domain data/domains/my_data.json
```

---

## Production Setup

### Option 1: Local Deployment

```bash
# Run production setup script
bash deploy/production_setup.sh

# Start API server
python api/api_server.py

# (Optional) Start continuous learning
python workflows/continuous_learning_workflow.py --continuous
```

### Option 2: Docker Deployment

```bash
# Build and run with Docker Compose
bash deploy/deploy_docker.sh

# Services will start:
#   - API server on port 5000
#   - Continuous learner (background)
```

---

## System Components

### Core Services
- **API Server**: `python api/api_server.py`
- **Continuous Learning**: `python workflows/continuous_learning_workflow.py`
- **Monitoring Dashboard**: `python monitoring/dashboard.py`

### Tools
- **Add Domain**: `python MASTER_INTEGRATION.py DOMAIN file`
- **List Domains**: `python LIST_DOMAINS.py --list`
- **Health Check**: `python scripts/health_check.py`
- **Benchmark**: `python scripts/benchmark_performance.py`

### Batch Operations
- **Integrate Existing**: `python tools/integrate_existing_domains.py`
- **Validate All**: `python tools/validate_all_domains.py`
- **Discover Patterns**: `python tools/discover_all_archetypes.py`
- **Batch Analyze**: `python tools/batch_analyze_domains.py`

---

## Configuration

Edit `src/pipeline_config.py` or create `config.json`:

```python
from src.pipeline_config import PipelineConfig, set_config

config = PipelineConfig(
    learning_enabled=True,
    incremental_learning=True,
    auto_prune_patterns=True,
    enable_caching=True
)

set_config(config)
```

---

## Common Workflows

### Adding a New Domain

```bash
# 1. Prepare data/domains/chess.json
# 2. Run integration
python MASTER_INTEGRATION.py chess data/domains/chess.json --pi 0.78 --type expertise

# 3. Review results
cat narrative_optimization/domains/chess/ANALYSIS_REPORT.md

# 4. (Optional) Refine
#    - Add custom transformer in src/transformers/archetypes/
#    - Add config entry in src/config/domain_archetypes.py
#    - Re-run

# 5. Register
python tools/integrate_existing_domains.py
```

### Continuous Improvement

```bash
# Start continuous learning (runs every hour)
python workflows/continuous_learning_workflow.py --continuous --interval 3600

# Or run single cycle
python workflows/continuous_learning_workflow.py
```

### Monitoring

```bash
# Live dashboard
python monitoring/dashboard.py

# Or single snapshot
python monitoring/dashboard.py --once

# Generate report
python scripts/automated_report.py
```

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test suite
pytest tests/test_integration_system.py -v
pytest tests/test_end_to_end.py -v

# Run regression tests
python scripts/regression_test.py

# Benchmark performance
python scripts/benchmark_performance.py
```

---

## Maintenance

### Regular Tasks

**Daily**:
```bash
# Check system health
python scripts/health_check.py

# Monitor performance
python scripts/monitor_performance.py
```

**Weekly**:
```bash
# Validate all domains
python tools/validate_all_domains.py

# Analyze pattern quality
python scripts/analyze_pattern_quality.py

# Generate report
python scripts/automated_report.py
```

**Monthly**:
```bash
# Batch analyze all domains
python tools/batch_analyze_domains.py

# Discover new patterns
python tools/discover_all_archetypes.py

# Generate documentation
python tools/generate_domain_docs.py
```

### Cleanup

```bash
# Clean cache
make clean

# Or manually
rm -rf ~/.narrative_optimization/cache/*
```

---

## Troubleshooting

### Issue: ImportError

**Solution**: Ensure all dependencies installed
```bash
pip install -r requirements.txt
```

### Issue: No data found

**Solution**: Check data directory
```bash
ls data/domains/
# Put data files here
```

### Issue: Low performance

**Solution**: Enable caching and run benchmark
```bash
python scripts/benchmark_performance.py
# Check for bottlenecks
```

### Issue: Tests failing

**Solution**: Run health check
```bash
python scripts/health_check.py
# Fix any issues reported
```

---

## Advanced Usage

### Programmatic Access

```python
from src.learning import LearningPipeline
from src.analysis.domain_specific_analyzer import DomainSpecificAnalyzer

# Initialize
pipeline = LearningPipeline()

# Ingest data
pipeline.ingest_domain('my_domain', texts, outcomes)

# Learn
metrics = pipeline.learn_cycle()

# Analyze
analyzer = DomainSpecificAnalyzer('my_domain')
results = analyzer.analyze_complete(texts, outcomes)
```

### API Access

```bash
# Start server
python api/api_server.py

# Make requests
curl http://localhost:5000/api/health
curl http://localhost:5000/api/domains
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"domain":"golf","text":"narrative...","outcome":1}'
```

---

## Next Steps

1. **Add your domains**: Follow `DOMAIN_ADDITION_TEMPLATE.md`
2. **Run learning**: Let system discover patterns
3. **Monitor**: Use dashboard to track progress
4. **Optimize**: Use benchmarking to improve performance
5. **Deploy**: Use Docker for production

---

**System is production-ready and continuously improving from data.**

