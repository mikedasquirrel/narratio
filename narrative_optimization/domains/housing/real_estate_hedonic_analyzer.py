"""
Hedonic Regression Analyzer - House Number Numerology

Implements hedonic pricing models to test if house numbers predict property values
after controlling for all physical and location characteristics.

Tests:
1. #13 discount (expected: 3-5%)
2. #666 discount (expected: 10-15%)  
3. #4 discount in Asian neighborhoods (expected: 3-6%)
4. #8 premium in Asian neighborhoods (expected: 5-8%)
5. Interaction effects by demographics
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import statsmodels.formula.api as smf
from typing import Dict, List, Tuple
import logging


class HedonicPricingAnalyzer:
    """Analyze house number effects on property values using hedonic pricing"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.results = {}
    
    def analyze_complete(self, properties_df: pd.DataFrame) -> Dict:
        """
        Run complete hedonic analysis
        
        Returns all findings on numerology effects
        """
        self.logger.info("="*80)
        self.logger.info("HEDONIC PRICING ANALYSIS - HOUSE NUMBER NUMEROLOGY")
        self.logger.info("="*80)
        self.logger.info(f"Sample size: {len(properties_df):,} properties")
        
        # Analysis 1: Overall #13 effect
        self.logger.info("\n=== Analysis 1: #13 Discount ===")
        self.results['thirteen_effect'] = self.test_thirteen_discount(properties_df)
        
        # Analysis 2: #666 effect
        self.logger.info("\n=== Analysis 2: #666 Discount ===")
        self.results['666_effect'] = self.test_666_discount(properties_df)
        
        # Analysis 3: #4 and #8 by Asian population
        self.logger.info("\n=== Analysis 3: Cultural Numerology (Asian #4/#8) ===")
        self.results['cultural_effects'] = self.test_cultural_numerology(properties_df)
        
        # Analysis 4: Builder behavior (#13 frequency)
        self.logger.info("\n=== Analysis 4: Builder Behavior ===")
        self.results['builder_behavior'] = self.test_builder_skipping(properties_df)
        
        # Analysis 5: Wealth effects
        self.logger.info("\n=== Analysis 5: Wealth Stratification ===")
        self.results['wealth_effects'] = self.test_wealth_moderation(properties_df)
        
        # Summary
        self._print_summary()
        
        return self.results
    
    def test_thirteen_discount(self, df: pd.DataFrame) -> Dict:
        """
        Test if #13 houses sell for less
        
        Hedonic model: ln(price) ~ controls + is_13
        """
        # Filter to valid data - use available columns
        available_cols = ['sale_price', 'sqft', 'bedrooms', 'bathrooms', 'year_built', 'is_exactly_13']
        available_cols = [c for c in available_cols if c in df.columns]
        
        df_clean = df[available_cols].dropna()
        
        if len(df_clean) < 100:
            return {'error': 'insufficient_data', 'n': len(df_clean)}
        
        # Log transform price
        df_clean['ln_price'] = np.log(df_clean['sale_price'])
        
        # Model 1: Baseline (no number effects)
        control_cols = ['sqft', 'bedrooms', 'bathrooms', 'year_built']
        control_cols = [c for c in control_cols if c in df_clean.columns]
        
        X_baseline = df_clean[control_cols]
        y = df_clean['ln_price']
        
        model_baseline = LinearRegression()
        model_baseline.fit(X_baseline, y)
        r2_baseline = model_baseline.score(X_baseline, y)
        
        # Model 2: Add #13 effect
        X_full = df_clean[control_cols + ['is_exactly_13']]
        model_full = LinearRegression()
        model_full.fit(X_full, y)
        r2_full = model_full.score(X_full, y)
        
        # Get #13 coefficient
        coef_13 = model_full.coef_[-1]  # Last coefficient is is_exactly_13
        
        # Convert to percentage
        pct_effect = (np.exp(coef_13) - 1) * 100
        
        # Count #13 houses
        n_thirteen = int(df_clean['is_exactly_13'].sum())
        
        # Statistical test
        from scipy import stats as sp_stats
        
        # T-test comparing #13 vs non-#13
        thirteen_prices = df_clean[df_clean['is_exactly_13'] == 1]['sale_price']
        other_prices = df_clean[df_clean['is_exactly_13'] == 0]['sale_price']
        
        if len(thirteen_prices) > 0 and len(other_prices) > 0:
            t_stat, p_value = sp_stats.ttest_ind(thirteen_prices, other_prices)
        else:
            t_stat, p_value = 0, 1.0
        
        self.logger.info(f"#13 houses: {n_thirteen:,}")
        self.logger.info(f"Baseline R²: {r2_baseline:.4f}")
        self.logger.info(f"With #13 R²: {r2_full:.4f}")
        self.logger.info(f"#13 coefficient: {coef_13:.4f}")
        self.logger.info(f"#13 effect: {pct_effect:.2f}% {'discount' if pct_effect < 0 else 'premium'}")
        self.logger.info(f"P-value: {p_value:.4f}")
        
        return {
            'n': len(df_clean),
            'n_thirteen': n_thirteen,
            'r2_baseline': float(r2_baseline),
            'r2_with_effect': float(r2_full),
            'coefficient': float(coef_13),
            'percent_effect': float(pct_effect),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
        }
    
    def test_666_discount(self, df: pd.DataFrame) -> Dict:
        """Test #666 effect (expected to be large if any exist)"""
        
        n_666 = int((df['is_exactly_666'] == 1).sum()) if 'is_exactly_666' in df.columns else 0
        
        self.logger.info(f"#666 houses found: {n_666}")
        
        if n_666 < 5:
            self.logger.info("Insufficient #666 houses for analysis")
            return {'error': 'insufficient_666_houses', 'n_666': n_666}
        
        # Similar analysis to #13
        df_clean = df[['sale_price', 'sqft', 'is_exactly_666']].dropna()
        
        prices_666 = df_clean[df_clean['is_exactly_666'] == 1]['sale_price']
        prices_other = df_clean[df_clean['is_exactly_666'] == 0]['sale_price']
        
        if len(prices_666) > 0:
            mean_666 = prices_666.mean()
            mean_other = prices_other.mean()
            diff_pct = (mean_666 - mean_other) / mean_other * 100
            
            self.logger.info(f"Mean #666 price: ${mean_666:,.0f}")
            self.logger.info(f"Mean other price: ${mean_other:,.0f}")
            self.logger.info(f"Difference: {diff_pct:.2f}%")
            
            return {
                'n_666': n_666,
                'mean_666_price': float(mean_666),
                'mean_other_price': float(mean_other),
                'percent_difference': float(diff_pct)
            }
        
        return {'n_666': n_666}
    
    def test_cultural_numerology(self, df: pd.DataFrame) -> Dict:
        """
        Test #4 and #8 effects moderated by Asian population
        
        Interaction model: price ~ controls + is_4 × asian_pct + is_8 × asian_pct
        """
        # Need demographic data joined
        if 'asian_pct' not in df.columns:
            self.logger.warning("No demographic data available")
            return {'error': 'no_demographics'}
        
        # Filter Asian neighborhoods
        df_asian = df[df['asian_pct'] > 20]  # High Asian population
        df_non_asian = df[df['asian_pct'] < 10]  # Low Asian population
        
        results = {}
        
        # Test #4 effect in Asian vs non-Asian areas
        if 'is_exactly_4' in df.columns:
            effect_asian = self._test_number_effect(df_asian, 'is_exactly_4')
            effect_non_asian = self._test_number_effect(df_non_asian, 'is_exactly_4')
            
            self.logger.info(f"#4 effect in Asian areas: {effect_asian.get('percent_effect', 0):.2f}%")
            self.logger.info(f"#4 effect in non-Asian areas: {effect_non_asian.get('percent_effect', 0):.2f}%")
            
            results['four_asian'] = effect_asian
            results['four_non_asian'] = effect_non_asian
        
        # Test #8 effect in Asian vs non-Asian areas
        if 'is_exactly_8' in df.columns:
            effect_asian = self._test_number_effect(df_asian, 'is_exactly_8')
            effect_non_asian = self._test_number_effect(df_non_asian, 'is_exactly_8')
            
            self.logger.info(f"#8 effect in Asian areas: {effect_asian.get('percent_effect', 0):.2f}%")
            self.logger.info(f"#8 effect in non-Asian areas: {effect_non_asian.get('percent_effect', 0):.2f}%")
            
            results['eight_asian'] = effect_asian
            results['eight_non_asian'] = effect_non_asian
        
        return results
    
    def _test_number_effect(self, df: pd.DataFrame, number_col: str) -> Dict:
        """Helper to test any number effect"""
        
        df_clean = df[['sale_price', number_col]].dropna()
        
        if len(df_clean) < 30:
            return {'error': 'insufficient_data', 'n': len(df_clean)}
        
        has_number = df_clean[df_clean[number_col] == 1]['sale_price']
        no_number = df_clean[df_clean[number_col] == 0]['sale_price']
        
        if len(has_number) == 0:
            return {'error': 'no_houses_with_number', 'n': 0}
        
        mean_with = has_number.mean()
        mean_without = no_number.mean()
        diff_pct = (mean_with - mean_without) / mean_without * 100
        
        t_stat, p_value = stats.ttest_ind(has_number, no_number)
        
        return {
            'n_with_number': len(has_number),
            'n_without': len(no_number),
            'mean_with': float(mean_with),
            'mean_without': float(mean_without),
            'percent_effect': float(diff_pct),
            'p_value': float(p_value),
            'significant': p_value < 0.05
        }
    
    def test_builder_skipping(self, df: pd.DataFrame) -> Dict:
        """
        Test if developers skip #13
        
        Chi-square test: Is #13 underrepresented?
        """
        # Count frequency of each number
        number_counts = df['street_number'].value_counts()
        
        # Expected frequency for numbers 1-20
        numbers_to_test = range(1, 21)
        observed = [number_counts.get(n, 0) for n in numbers_to_test]
        
        # Expected: uniform distribution
        expected_freq = sum(observed) / len(numbers_to_test)
        expected = [expected_freq] * len(numbers_to_test)
        
        # Chi-square test
        chi2, p_value = stats.chisquare(observed, expected)
        
        # Specific #13 frequency
        freq_12 = number_counts.get(12, 0)
        freq_13 = number_counts.get(13, 0)
        freq_14 = number_counts.get(14, 0)
        
        # Is #13 less common than neighbors?
        avg_neighbor = (freq_12 + freq_14) / 2
        if avg_neighbor > 0:
            ratio_13_to_neighbors = freq_13 / avg_neighbor
        else:
            ratio_13_to_neighbors = 1.0
        
        self.logger.info(f"#12 frequency: {freq_12}")
        self.logger.info(f"#13 frequency: {freq_13}")
        self.logger.info(f"#14 frequency: {freq_14}")
        self.logger.info(f"#13 vs neighbors ratio: {ratio_13_to_neighbors:.2f}")
        self.logger.info(f"Chi-square p-value: {p_value:.4f}")
        
        if ratio_13_to_neighbors < 0.80:
            self.logger.info("✓ Developers ARE skipping #13!")
        else:
            self.logger.info("✗ No evidence of #13 skipping")
        
        return {
            'freq_12': int(freq_12),
            'freq_13': int(freq_13),
            'freq_14': int(freq_14),
            'ratio_13_to_neighbors': float(ratio_13_to_neighbors),
            'chi_square': float(chi2),
            'p_value': float(p_value),
            'builders_skip_13': ratio_13_to_neighbors < 0.80
        }
    
    def test_wealth_moderation(self, df: pd.DataFrame) -> Dict:
        """
        Test if rich people care less about #13
        
        Split by median income and test #13 effect in each
        """
        if 'median_income' not in df.columns:
            return {'error': 'no_income_data'}
        
        # Split by income
        high_income = df[df['median_income'] > 100000]
        low_income = df[df['median_income'] < 60000]
        
        self.logger.info(f"High income sample: {len(high_income):,}")
        self.logger.info(f"Low income sample: {len(low_income):,}")
        
        # Test #13 effect in each group
        effect_rich = self._test_number_effect(high_income, 'is_exactly_13')
        effect_poor = self._test_number_effect(low_income, 'is_exactly_13')
        
        self.logger.info(f"#13 effect in high income: {effect_rich.get('percent_effect', 0):.2f}%")
        self.logger.info(f"#13 effect in low income: {effect_poor.get('percent_effect', 0):.2f}%")
        
        return {
            'high_income': effect_rich,
            'low_income': effect_poor,
            'wealth_moderates': abs(effect_rich.get('percent_effect', 0)) < abs(effect_poor.get('percent_effect', 0))
        }
    
    def calculate_aggregate_dollar_effect(self, df: pd.DataFrame, results: Dict) -> Dict:
        """
        Calculate total dollar amount of superstition effects
        
        If #13 discount is 3%, and there are 10,000 #13 houses at avg $500K,
        that's $150 MILLION in lost value from superstition
        """
        thirteen_effect_pct = results.get('thirteen_effect', {}).get('percent_effect', 0)
        n_thirteen = results.get('thirteen_effect', {}).get('n_thirteen', 0)
        
        # Average price of #13 houses
        thirteen_houses = df[df['is_exactly_13'] == 1]
        if len(thirteen_houses) > 0:
            avg_price = thirteen_houses['sale_price'].mean()
            
            # Total dollar effect
            total_effect = abs(thirteen_effect_pct / 100) * avg_price * n_thirteen
            
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"AGGREGATE FINANCIAL IMPACT")
            self.logger.info(f"{'='*80}")
            self.logger.info(f"#13 houses in sample: {n_thirteen:,}")
            self.logger.info(f"Average #13 house price: ${avg_price:,.0f}")
            self.logger.info(f"#13 effect: {thirteen_effect_pct:.2f}%")
            self.logger.info(f"Total lost value (sample): ${total_effect:,.0f}")
            
            # Extrapolate to US
            # Estimate: 130M homes in US, ~1/13th might be #13
            us_homes_total = 130_000_000
            estimated_thirteen_homes = us_homes_total / 150  # Rough estimate accounting for skipping
            us_avg_price = 350_000
            
            us_total_effect = abs(thirteen_effect_pct / 100) * us_avg_price * estimated_thirteen_homes
            
            self.logger.info(f"\nEXTRAPOLATED TO US MARKET:")
            self.logger.info(f"Estimated #13 homes in US: ~{estimated_thirteen_homes:,.0f}")
            self.logger.info(f"US average home price: ${us_avg_price:,.0f}")
            self.logger.info(f"TOTAL US MARKET EFFECT: ${us_total_effect/1e9:.1f} BILLION")
            self.logger.info(f"{'='*80}")
            
            return {
                'sample_n_thirteen': n_thirteen,
                'sample_avg_price': float(avg_price),
                'sample_total_effect': float(total_effect),
                'us_estimated_thirteen_homes': estimated_thirteen_homes,
                'us_total_effect_billions': float(us_total_effect / 1e9)
            }
        
        return {}
    
    def _print_summary(self):
        """Print summary of all findings"""
        self.logger.info("\n" + "="*80)
        self.logger.info("SUMMARY OF FINDINGS")
        self.logger.info("="*80)
        
        # #13 effect
        if 'thirteen_effect' in self.results:
            result = self.results['thirteen_effect']
            if 'error' not in result:
                self.logger.info(f"\n#13 EFFECT:")
                self.logger.info(f"  Sample: {result['n']:,} homes ({result['n_thirteen']} are #13)")
                self.logger.info(f"  Effect: {result['percent_effect']:.2f}%")
                self.logger.info(f"  Status: {'✅ SIGNIFICANT' if result['significant'] else '❌ NOT SIGNIFICANT'}")
        
        # Builder behavior
        if 'builder_behavior' in self.results:
            result = self.results['builder_behavior']
            if 'error' not in result:
                self.logger.info(f"\nBUILDER BEHAVIOR:")
                self.logger.info(f"  #13 frequency ratio: {result['ratio_13_to_neighbors']:.2f}")
                self.logger.info(f"  Builders skip #13: {'✅ YES' if result['builders_skip_13'] else '❌ NO'}")
        
        self.logger.info("\n" + "="*80)

