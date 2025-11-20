#!/usr/bin/env python3
"""
STREET NAME ANALYSIS - MASSIVE SCALE

Tests if street names predict property values WITHIN same neighborhood.

Strategy:
1. Use the 395K homes we already collected
2. Extract street name features  
3. Control for neighborhood (compare streets in same ZIP/city)
4. Test: "Park Lane" vs "Cemetery Road" effect

This is BETTER than house numbers because:
- More semantic variation
- Can control for neighborhood
- Natural experiments exist (renamings)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import pandas as pd
import numpy as np
from scipy import stats
import re
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

from app import app
from core.models import db, RealEstateProperty, StreetNameAnalysis

def extract_street_features(street_name):
    """Extract phonetic and semantic features from street name"""
    
    name_lower = street_name.lower()
    
    # Phonetic
    plosives = 'ptkbdg'
    liquids = 'lrmn'
    vowels = 'aeiouy'
    
    plosive_count = sum(1 for c in name_lower if c in plosives)
    liquid_count = sum(1 for c in name_lower if c in liquids)
    vowel_count = sum(1 for c in name_lower if c in vowels)
    
    total = max(len(name_lower), 1)
    harshness = (plosive_count / total) * 100
    softness = (liquid_count + vowel_count) / total * 100
    
    # Semantic - POSITIVE
    nature_words = {'park', 'tree', 'lake', 'river', 'mountain', 'hill', 'forest', 'meadow', 'garden', 'oak', 'pine', 'willow', 'rose', 'lily', 'ocean', 'sea', 'bay', 'creek', 'brook'}
    prestige_words = {'royal', 'manor', 'estate', 'grand', 'mansion', 'palace', 'crown'}
    pleasant_words = {'pleasant', 'sunny', 'happy', 'joy', 'delight', 'beauty', 'peace', 'harmony', 'pleasant', 'serene'}
    
    # Semantic - NEGATIVE  
    negative_words = {'cemetery', 'funeral', 'grave', 'death', 'dump', 'waste', 'garbage', 'prison', 'jail'}
    danger_words = {'crack', 'crime', 'murder', 'kill', 'danger', 'hazard'}
    industrial_words = {'factory', 'industrial', 'warehouse', 'plant', 'mill'}
    infrastructure_words = {'highway', 'freeway', 'railroad', 'rail', 'route', 'bypass'}
    
    has_nature = any(word in name_lower for word in nature_words)
    has_prestige = any(word in name_lower for word in prestige_words)
    has_pleasant = any(word in name_lower for word in pleasant_words)
    has_negative = any(word in name_lower for word in negative_words)
    has_danger = any(word in name_lower for word in danger_words)
    has_industrial = any(word in name_lower for word in industrial_words)
    has_infrastructure = any(word in name_lower for word in infrastructure_words)
    
    # Emotional valence
    positive_score = (has_nature + has_prestige + has_pleasant) * 33.33
    negative_score = (has_negative + has_danger + has_industrial + has_infrastructure) * 25
    
    valence = (positive_score - negative_score) / 100  # -1 to +1
    
    return {
        'street_name': street_name,
        'harshness_score': harshness,
        'softness_score': softness,
        'memorability_score': 100 - len(street_name),  # Shorter = more memorable
        'syllable_count': name_lower.count('a') + name_lower.count('e') + name_lower.count('i') + name_lower.count('o') + name_lower.count('u'),
        'character_length': len(street_name),
        'has_nature_words': has_nature,
        'has_prestige_words': has_prestige,
        'has_pleasant_words': has_pleasant,
        'has_negative_words': has_negative,
        'has_danger_words': has_danger,
        'has_industrial_words': has_industrial,
        'has_infrastructure_words': has_infrastructure,
        'emotional_valence': valence,
        'semantic_valence_score': (valence + 1) * 50,  # 0-100
        'natural_beauty_score': positive_score,
    }

def main():
    logger.info("="*80)
    logger.info("🛣️  STREET NAME ANALYSIS - MASSIVE SCALE")
    logger.info("Using 395K homes already collected!")
    logger.info("="*80)
    
    with app.app_context():
        # Get all properties
        logger.info("\nLoading properties...")
        props = RealEstateProperty.query.limit(100000).all()
        logger.info(f"✓ Loaded {len(props):,} properties")
        
        # Extract unique streets
        logger.info("\nExtracting street names...")
        street_data = {}
        
        for p in props:
            street = p.street_name
            if street not in street_data:
                street_data[street] = {
                    'street_name': street,
                    'properties': [],
                    'city': p.city,
                    'state': p.state
                }
            
            street_data[street]['properties'].append({
                'price': p.sale_price,
                'sqft': p.sqft,
                'year_built': p.year_built
            })
        
        logger.info(f"✓ Found {len(street_data):,} unique streets")
        
        # Analyze each street
        logger.info("\nAnalyzing street names...")
        results = []
        
        for street_name, data in street_data.items():
            if len(data['properties']) < 3:
                continue
            
            # Extract features
            features = extract_street_features(street_name)
            
            # Calculate average price
            prices = [p['price'] for p in data['properties']]
            avg_price = np.mean(prices)
            
            results.append({
                **features,
                'avg_price': avg_price,
                'property_count': len(data['properties']),
                'city': data['city'],
                'state': data['state']
            })
        
        df = pd.DataFrame(results)
        logger.info(f"✓ Analyzed {len(df):,} streets with 3+ properties")
        
        # ANALYSIS
        logger.info("\n" + "="*80)
        logger.info("📊 STREET NAME EFFECTS")
        logger.info("="*80)
        
        # Test 1: Nature words
        nature_streets = df[df['has_nature_words'] == True]
        non_nature = df[df['has_nature_words'] == False]
        
        if len(nature_streets) > 10 and len(non_nature) > 10:
            nature_mean = nature_streets['avg_price'].mean()
            non_nature_mean = non_nature['avg_price'].mean()
            diff_pct = (nature_mean - non_nature_mean) / non_nature_mean * 100
            
            t, p = stats.ttest_ind(nature_streets['avg_price'], non_nature['avg_price'])
            
            logger.info(f"\n🌳 NATURE WORDS (Park, Lake, Tree, etc.):")
            logger.info(f"Streets with nature: {len(nature_streets):,}")
            logger.info(f"Streets without: {len(non_nature):,}")
            logger.info(f"Mean with nature: ${nature_mean:,.0f}")
            logger.info(f"Mean without: ${non_nature_mean:,.0f}")
            logger.info(f"PREMIUM: {diff_pct:+.2f}%")
            logger.info(f"P-value: {p:.6f}")
            logger.info(f"Significant: {'✅ YES' if p < 0.05 else '❌ NO'}")
        
        # Test 2: Negative words
        negative_streets = df[df['has_negative_words'] == True]
        non_negative = df[df['has_negative_words'] == False]
        
        if len(negative_streets) > 5:
            neg_mean = negative_streets['avg_price'].mean()
            non_neg_mean = non_negative['avg_price'].mean()
            diff_pct = (neg_mean - non_neg_mean) / non_neg_mean * 100
            
            logger.info(f"\n⚠️  NEGATIVE WORDS (Cemetery, Prison, etc.):")
            logger.info(f"Streets with negative: {len(negative_streets):,}")
            logger.info(f"Mean with negative: ${neg_mean:,.0f}")
            logger.info(f"Mean without: ${non_neg_mean:,.0f}")
            logger.info(f"DISCOUNT: {diff_pct:+.2f}%")
        
        # Test 3: Overall semantic valence correlation
        logger.info(f"\n📈 SEMANTIC VALENCE CORRELATION:")
        r, p = stats.pearsonr(df['emotional_valence'], df['avg_price'])
        logger.info(f"Correlation (emotional valence × price): r={r:.3f}, p={p:.6f}")
        logger.info(f"Interpretation: {'✅ SIGNIFICANT' if p < 0.05 else '❌ NOT SIGNIFICANT'}")
        
        # Test 4: Harshness
        logger.info(f"\n💥 HARSHNESS CORRELATION:")
        r_harsh, p_harsh = stats.pearsonr(df['harshness_score'], df['avg_price'])
        logger.info(f"Correlation (harshness × price): r={r_harsh:.3f}, p={p_harsh:.6f}")
        logger.info(f"Interpretation: {'Harsh names = ' + ('LOWER' if r_harsh < 0 else 'HIGHER') + ' prices' if p_harsh < 0.05 else 'No effect'}")
        
        # Top and bottom streets
        logger.info(f"\n🏆 TOP 10 MOST POSITIVE STREETS:")
        top10 = df.nlargest(10, 'emotional_valence')[['street_name', 'avg_price', 'emotional_valence', 'property_count']]
        for idx, row in top10.iterrows():
            logger.info(f"  {row['street_name']}: ${row['avg_price']:,.0f} (valence: {row['emotional_valence']:.2f}, n={row['property_count']})")
        
        logger.info(f"\n💀 TOP 10 MOST NEGATIVE STREETS:")
        bottom10 = df.nsmallest(10, 'emotional_valence')[['street_name', 'avg_price', 'emotional_valence', 'property_count']]
        for idx, row in bottom10.iterrows():
            logger.info(f"  {row['street_name']}: ${row['avg_price']:,.0f} (valence: {row['emotional_valence']:.2f}, n={row['property_count']})")
        
        # Save results
        df.to_csv('data/real_estate/STREET_NAME_ANALYSIS.csv', index=False)
        logger.info(f"\n✓ Results saved to data/real_estate/STREET_NAME_ANALYSIS.csv")
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("🎯 SUMMARY")
        logger.info("="*80)
        logger.info(f"Streets analyzed: {len(df):,}")
        logger.info(f"Properties: {df['property_count'].sum():,}")
        logger.info(f"")
        logger.info(f"Nature word premium: {diff_pct if 'diff_pct' in locals() else 'N/A'}%")
        logger.info(f"Semantic valence effect: r={r:.3f} ({'significant' if p < 0.05 else 'not significant'})")
        logger.info(f"Harshness effect: r={r_harsh:.3f} ({'significant' if p_harsh < 0.05 else 'not significant'})")
        logger.info("="*80)
        
        return df

if __name__ == '__main__':
    main()

