# Narrative Optimization Framework - Makefile

.PHONY: help init test demo clean integrate validate discover docs all health monitor benchmark report api

help:
	@echo "Narrative Optimization Framework"
	@echo "================================"
	@echo ""
	@echo "Setup:"
	@echo "  make init        - Initialize system"
	@echo "  make install     - Install dependencies"
	@echo "  make health      - Run health check"
	@echo ""
	@echo "Operations:"
	@echo "  make integrate   - Integrate existing domains"
	@echo "  make validate    - Validate all domains"
	@echo "  make discover    - Discover archetypes for all"
	@echo "  make analyze     - Batch analyze all domains"
	@echo ""
	@echo "Learning:"
	@echo "  make learn       - Run learning cycle"
	@echo "  make continuous  - Start continuous learning"
	@echo ""
	@echo "Development:"
	@echo "  make test        - Run test suite"
	@echo "  make regression  - Run regression tests"
	@echo "  make demo        - Run complete demo"
	@echo "  make docs        - Generate documentation"
	@echo ""
	@echo "Monitoring:"
	@echo "  make monitor     - Start monitoring dashboard"
	@echo "  make benchmark   - Run performance benchmark"
	@echo "  make report      - Generate system report"
	@echo ""
	@echo "Deployment:"
	@echo "  make api         - Start API server"
	@echo "  make docker      - Deploy with Docker"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean       - Clean cache and temp files"
	@echo "  make list        - List all domains"
	@echo "  make export      - Export learned patterns"
	@echo "  make all         - Run complete workflow"

init:
	@echo "Initializing system..."
	@python INITIALIZE_SYSTEM.py

install:
	@echo "Installing dependencies..."
	@pip install -r requirements.txt

integrate:
	@echo "Integrating existing domains..."
	@python tools/integrate_existing_domains.py

validate:
	@echo "Validating all domains..."
	@python tools/validate_all_domains.py

discover:
	@echo "Discovering archetypes..."
	@python tools/discover_all_archetypes.py

analyze:
	@echo "Batch analyzing domains..."
	@python tools/batch_analyze_domains.py

test:
	@echo "Running tests..."
	@pytest tests/ -v

demo:
	@echo "Running complete system demo..."
	@python DEMO_COMPLETE_SYSTEM.py

docs:
	@echo "Generating documentation..."
	@python tools/generate_domain_docs.py

clean:
	@echo "Cleaning cache and temp files..."
	@rm -rf ~/.narrative_optimization/cache/*
	@rm -f narrative_optimization/*.pyc
	@rm -rf narrative_optimization/__pycache__
	@rm -rf src/__pycache__
	@echo "✓ Cleaned"

list:
	@echo "Listing all domains..."
	@python LIST_DOMAINS.py --list

health:
	@echo "Running health check..."
	@python scripts/health_check.py

monitor:
	@echo "Starting monitoring dashboard..."
	@python monitoring/dashboard.py

benchmark:
	@echo "Running performance benchmark..."
	@python scripts/benchmark_performance.py

report:
	@echo "Generating system report..."
	@python scripts/automated_report.py

learn:
	@echo "Running learning cycle..."
	@python workflows/continuous_learning_workflow.py

continuous:
	@echo "Starting continuous learning (Ctrl+C to stop)..."
	@python workflows/continuous_learning_workflow.py --continuous --interval 3600

regression:
	@echo "Running regression tests..."
	@python scripts/regression_test.py

api:
	@echo "Starting API server..."
	@python api/api_server.py

docker:
	@echo "Deploying with Docker..."
	@bash deploy/deploy_docker.sh

export:
	@echo "Exporting learned patterns..."
	@python scripts/export_learned_patterns.py

all:
	@echo "Running complete workflow..."
	@make init
	@make health
	@make integrate
	@make discover
	@make validate
	@make analyze
	@make learn
	@make docs
	@make report
	@echo ""
	@echo "✓ Complete workflow finished"


