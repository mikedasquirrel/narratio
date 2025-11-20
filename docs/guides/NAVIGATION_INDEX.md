# 🗺️ Navigation Index — Quick Reference

**Project**: Narrative Optimization Framework  
**Version**: 2.0.0

---

## 🚀 Quick Actions

| I Want To... | Go Here |
|-------------|---------|
| **Get started immediately** | Read `START_HERE.md` |
| **Try the AI comparison** | Open `http://127.0.0.1:5738/analyze/compare` |
| **Run the server** | Execute `python app.py` |
| **Understand the project** | Read `README.md` |
| **Integrate external data** | Read `DOMAIN_INTEGRATION_GUIDE.md` |

---

## 📂 Documentation Map

### Essential Reading
1. **START_HERE.md** - Immediate quick start
2. **README.md** - Complete project overview
3. **HOW_TO_USE_COMPARISON.md** - Using the comparison system
4. **DOMAIN_INTEGRATION_GUIDE.md** - Integrating external data

### Technical Documentation
- **COMPARISON_SYSTEM_GUIDE.md** - System architecture
- **PROJECT_ORGANIZATION.md** - File structure guide
- **IMPLEMENTATION_COMPLETE.md** - What's been built

### Research Framework
- **narrative_optimization/README.md** - Framework overview
- **narrative_optimization/docs/architecture.md** - Design patterns
- **narrative_optimization/docs/hypotheses.md** - Research hypotheses (H1-H10)
- **narrative_optimization/docs/findings.md** - Research results
- **narrative_optimization/docs/roadmap.md** - Future plans

### Integration
- **domain_schemas/text_domain_schema.json** - Text data format
- **domain_schemas/feature_domain_schema.json** - Feature data format
- **domain_schemas/mixed_domain_schema.json** - Mixed data format
- **domain_schemas/ekko_domain_schema.json** - Example integration

---

## 🌐 Web Application URLs

| Page | URL | Purpose |
|------|-----|---------|
| **Home** | `/` | Dashboard overview |
| **Comparison** | `/analyze/compare` | AI-powered text comparison (MAIN FEATURE) |
| **Experiments** | `/experiments` | View and run experiments |
| **Network Explorer** | `/viz` | Interactive network visualization |
| **API Docs** | `/api` | API documentation |

---

## 🧪 Research Framework Files

### Transformers
| Transformer | File | Features |
|------------|------|----------|
| Statistical Baseline | `src/transformers/statistical.py` | TF-IDF (baseline) |
| Semantic Narrative | `src/transformers/semantic.py` | Embeddings, clustering |
| Domain Text | `src/transformers/domain_text.py` | Style, structure, topics |
| Nominative | `src/transformers/nominative.py` | Naming, categorization (24 features) |
| Self-Perception | `src/transformers/self_perception.py` | Identity, agency (21 features) |
| Narrative Potential | `src/transformers/narrative_potential.py` | Growth, openness (25 features) |
| Linguistic | `src/transformers/linguistic_advanced.py` | Voice, temporality (26 features) |
| Relational | `src/transformers/relational.py` | Complementarity (9 features) |
| Ensemble | `src/transformers/ensemble.py` | Network effects (11 features) |

### Core Framework
| Component | File | Purpose |
|-----------|------|---------|
| Base Transformer | `src/transformers/base.py` | Abstract base class |
| Pipeline Assembly | `src/pipelines/narrative_pipeline.py` | Build pipelines |
| Experiment Runner | `src/experiments/experiment.py` | Compare narratives |
| Evaluator | `src/evaluation/evaluator.py` | Multi-objective evaluation |
| Contextual Analyzer | `src/analysis/contextual_analyzer.py` | Missing variable detection |

---

## 💻 Command Reference

### Running the Application
```bash
# Start web server
python app.py

# Access at
http://127.0.0.1:5738
```

### Running Experiments
```bash
cd narrative_optimization

# Run specific experiment
python run_experiment.py --experiment 01_baseline_comparison

# Run all experiments
python run_all_experiments.py

# List available experiments
python run_experiment.py --list
```

### Testing
```bash
cd narrative_optimization

# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test
pytest tests/test_transformers.py
```

### Jupyter Notebooks
```bash
cd narrative_optimization

# Start Jupyter
jupyter notebook

# Open quick start
# notebooks/00_quick_start.ipynb
```

---

## 🎯 Use Cases

### Comparing Two Texts
**URL**: `/analyze/compare`
- Enter any two texts
- Ask a comparative question
- Get AI-powered analysis with 100+ features

### Running Experiments
**Command**: `python narrative_optimization/run_experiment.py`
- Test narrative hypotheses
- Compare feature engineering approaches
- Generate research reports

### Integrating External Data
**Guide**: `DOMAIN_INTEGRATION_GUIDE.md`
- Format your data
- Select transformers
- Run analysis
- Export results

### Interactive Exploration
**URL**: `/viz`
- Visualize narrative networks
- Explore feature relationships
- Interactive charts

---

## 🔑 Key Concepts

### Narrative Transformer
Converts text → interpretable features that tell a story

### Narrative Pipeline
Composes transformers into coherent hypothesis

### Narrative Experiment
Compares competing narrative approaches systematically

### AI Enhancement
GPT-4 interprets transformer outputs and answers questions

### Domain Integration
Universal format for bringing external data into framework

---

## 🎨 Design System

### Colors
- **Accent Cyan**: #06b6d4
- **Accent Purple**: #9333ea
- **Accent Pink**: #ec4899
- **Accent Gold**: #fbbf24
- **Accent Emerald**: #10b981

### Typography
- **Headings**: Playfair Display, weights 200-300
- **Body**: Manrope, weights 300-500
- **Code**: Monaco, monospace

### Effects
- **Glassmorphism**: backdrop-filter: blur(20px)
- **Animated waves**: 25-35s infinite loops
- **Micro-animations**: 0.2-0.3s transitions
- **Hover transforms**: translateY(-2px to -4px)

---

## 📊 What's Analyzed

### Per Text (100+ Features)
- **24 Nominative** features (naming, semantic fields)
- **21 Self-Perception** features (identity, agency)
- **25 Narrative Potential** features (growth, openness)
- **26 Linguistic** features (voice, temporality)
- **9 Relational** features (complementarity)
- **11 Ensemble** features (network, diversity)

### Visualizations
- **Radar Chart** - 6 dimensions at once
- **Heatmap** - 10 semantic fields
- **Pie Charts** - Voice and temporal distribution
- **Bar Chart** - Feature importance rankings

### AI Insights
- **Comparison type** detection
- **Direct answer** to user's question
- **Key insights** (4-5 findings)
- **Important features** (5-8 with explanations)
- **Narrative themes** (essence of each text)
- **Implications** and recommendations

---

## 🚦 Status

### Fully Implemented ✅
- AI-powered comparison system
- 6 narrative transformers
- Rich visualizations
- Question answering
- Dark glassmorphism UI
- Contextual analysis
- Domain integration specs
- Comprehensive documentation

### Production Ready ✅
- Error handling
- Logging and debugging
- Responsive design
- Fast performance
- Professional polish

### Well-Documented ✅
- User guides
- Technical docs
- Integration manuals
- Code comments
- Research papers

---

## 🎯 Quick Test

```bash
# 1. Start server
python app.py

# 2. Open browser
# http://127.0.0.1:5738/analyze/compare

# 3. Enter:
Text A: "dodgers"
Text B: "phillies"
Question: "who is likely to win tonight?"

# 4. Click "run comprehensive analysis"

# 5. Watch AI analyze:
# - 6 transformers run
# - 100+ features extracted
# - GPT-4 interprets patterns
# - Answers your question
# - Shows visualizations
```

---

**Organization Date**: November 10, 2025  
**Framework Version**: 2.0.0  
**Status**: ✅ CLEAN, ORGANIZED, AND OPERATIONAL

