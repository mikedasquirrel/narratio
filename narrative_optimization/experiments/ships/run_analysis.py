"""
Ships Narrative Analysis
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from domains.ships.data_loader import ShipDataLoader
from domains.ships.gravitas_analyzer import GravitasAnalyzer
from src.transformers.ships.gravitas_transformer import GravitasTransformer
import numpy as np
from scipy import stats

loader = ShipDataLoader()
ships = loader.load_ships()

print(f"\n{'='*70}")
print(f"SHIPS NARRATIVE ANALYSIS - {len(ships)} vessels")
print(f"{'='*70}\n")

# Analyzer
analyzer = GravitasAnalyzer()
category_analysis = analyzer.analyze_category_effects(ships)

print("Category Effects:")
for cat_data in category_analysis['rankings'][:5]:
    print(f"  {cat_data['category']}: mean={cat_data['mean']:.1f}, n={cat_data['n']}")

# Transformer
names = [s['name'] for s in ships if 'name' in s]
significance = [s['historical_significance_score'] for s in ships if 'historical_significance_score' in s]

transformer = GravitasTransformer()
X = transformer.fit_transform(names[:len(significance)])

from sklearn.linear_model import Ridge
model = Ridge(alpha=1.0)
model.fit(X, significance)

from sklearn.metrics import r2_score
r2 = r2_score(significance, model.predict(X))
print(f"\nGravitas transformer → Significance: R² = {r2:.3f}")

print(f"\n{'='*70}\n")

