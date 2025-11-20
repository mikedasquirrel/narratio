# System Overview

**Production-ready holistic learning system for narrative analysis across domains**

---

## Core Principle

> **"What are the stories of this domain? Do they match familiar stories from structurally similar domains? Do they unfold at predicted frequency (accounting for observation bias)? What patterns emerge in story realization?"**

---

## Architecture

### Three Learning Layers

**Universal** → Cross-domain patterns (underdog, comeback, pressure, rivalry, dominance)  
**Domain** → Domain-specific patterns (golf: course mastery, tennis: surface expertise)  
**Context** → Situational patterns (championship vs regular, high-stakes vs routine)

### Learning Loop

```
Data → Discover → Validate → Measure → Update → Apply
  ↑______________________________________________|
```

Continuous improvement through feedback.

---

## Key Files

**Entry Points**:
- `MASTER_INTEGRATION.py` - Add/analyze domains
- `LIST_DOMAINS.py` - View all domains
- `DEMO_COMPLETE_SYSTEM.py` - Complete demonstration

**Core System**:
- `src/learning/` - Holistic learning (14 modules)
- `src/analysis/` - Domain analyzers
- `src/transformers/` - 56+ feature extractors
- `src/config/` - Domain configurations

**Tools**:
- `tools/` - Batch processing, discovery, validation
- `scripts/` - Health, monitoring, reporting
- `workflows/` - Automated workflows

**Deployment**:
- `Dockerfile` - Container definition
- `docker-compose.yml` - Multi-service setup
- `api/` - REST API server

---

## Quick Commands

```bash
# Setup
make init               # Initialize system
make install            # Install dependencies

# Add domain
python MASTER_INTEGRATION.py DOMAIN data/domains/DOMAIN.json

# Learn
make learn              # Single cycle
make continuous         # Continuous (every hour)

# Monitor
make health             # Health check
make monitor            # Live dashboard
make report             # Generate report

# Deploy
make api                # Start API server
make docker             # Docker deployment

# Complete workflow
make all                # Run everything
```

---

## System Capabilities

✓ **Domain-agnostic learning** - Universal patterns work everywhere  
✓ **Domain-specific learning** - Unique patterns per domain  
✓ **Structural similarity** - Auto-finds related domains  
✓ **Pattern transfer** - Learns from similar domains  
✓ **Statistical validation** - All patterns tested (p < 0.05)  
✓ **Hierarchical structures** - Multi-level pattern hierarchies  
✓ **Active learning** - Focuses on uncertain patterns  
✓ **Meta-learning** - Cross-domain knowledge transfer  
✓ **Ensemble methods** - Multiple hypotheses  
✓ **Online learning** - Real-time updates  
✓ **Causal discovery** - Identifies causal patterns  
✓ **Full interpretability** - Human-readable explanations  
✓ **Version control** - A/B testing, rollback  
✓ **Caching** - Fast repeated queries  
✓ **API access** - RESTful endpoints  
✓ **Docker ready** - Containerized deployment  
✓ **CI/CD pipeline** - Automated testing  
✓ **Monitoring** - Real-time dashboard  

---

## Current Status

- **50+ files created/modified**
- **Comprehensive learning system operational**
- **12 domains integrated with archetypes**
- **56+ transformers available**
- **Production-ready deployment**
- **Full test coverage**
- **Automated workflows**
- **Clean, organized codebase**

---

##For Future Developers

### Adding a Domain
1. Prepare `data/domains/YOUR_DOMAIN.json`
2. Run `python MASTER_INTEGRATION.py YOUR_DOMAIN data/domains/YOUR_DOMAIN.json`
3. Review `narrative_optimization/domains/YOUR_DOMAIN/ANALYSIS_REPORT.md`
4. (Optional) Add custom transformer and config
5. Re-run for refined analysis

### Understanding the System
- **The Genome (ж)**: `[nominative, archetypal, historial, uniquity]`
- **Story Quality (ю)**: Distance from domain's ideal archetype (Ξ)
- **Narrative Agency (Д)**: Bridge between story and outcomes
- **Narrativity (п)**: Domain's openness to narrative influence

### Key Equations
```
Regular domains: Д = п × |r| × κ
Prestige domains: Д = ة + θ - λ
Story quality: ю = f(ж, Ξ)
```

---

## What Makes This System Unique

1. **Learns continuously** - Patterns improve from data
2. **Transfers knowledge** - Applies learnings across domains
3. **Statistically rigorous** - All patterns validated
4. **Fully interpretable** - Every prediction explainable
5. **Production-ready** - Docker, API, monitoring, CI/CD
6. **Seamlessly integrated** - All components work together

---

**The system discovers what stories exist, how they relate to familiar stories, and whether they unfold as predicted given domain constraints.**

