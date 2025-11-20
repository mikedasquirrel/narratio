"""
Crypto Market Transformer

Extracts NARRATIVITY FROM CRYPTOCURRENCY MARKET METRICS.
Stats as story: volatility = risk narrative, volume = attention narrative,
development activity = innovation narrative.

Author: Narrative Integration System
Date: November 14, 2025
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union
from sklearn.base import BaseEstimator, TransformerMixin


class CryptoMarketTransformer(BaseEstimator, TransformerMixin):
    """
    Extract narrative features from cryptocurrency market metrics.
    
    Features: 25 total
    - Price narrative (8)
    - Market structure (6)
    - Development narrative (5)
    - Community narrative (6)
    """
    
    def __init__(self, normalize: bool = True, log_transform: bool = True):
        """
        Initialize crypto analyzer.
        
        Parameters
        ----------
        normalize : bool
            Normalize stats relative to dataset
        log_transform : bool
            Log-transform skewed metrics (market cap, volume)
        """
        self.normalize = normalize
        self.log_transform = log_transform
        self.stat_means_ = {}
        self.stat_stds_ = {}
    
    def fit(self, X, y=None):
        """Learn normalization parameters"""
        if isinstance(X, list):
            df = pd.DataFrame(X)
        elif isinstance(X, pd.DataFrame):
            df = X
        else:
            return self
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            values = df[col].dropna()
            if len(values) > 0:
                self.stat_means_[col] = values.mean()
                self.stat_stds_[col] = values.std() if values.std() > 0 else 1.0
        
        return self
    
    def transform(self, X):
        """Transform crypto metrics to narrative features"""
        if isinstance(X, pd.DataFrame):
            X = X.to_dict('records')
        elif not isinstance(X, list):
            X = [X]
        
        features = []
        for stats in X:
            feat = self._extract_crypto_features(stats)
            features.append(feat)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_crypto_features(self, stats: Dict) -> List[float]:
        """Extract all crypto market narrative features"""
        features = []
        
        # === PRICE NARRATIVE (8 features) ===
        
        # 1. Size narrative (market cap)
        market_cap = stats.get('market_cap', stats.get('marketcap', 1e9))
        if self.log_transform and market_cap > 0:
            market_cap_log = np.log10(market_cap)
            features.append(min(1.0, (market_cap_log - 6) / 6))  # Scale 10^6 to 10^12
        else:
            features.append(self._normalize_stat('market_cap', market_cap))
        
        # 2. Risk narrative (price volatility)
        volatility = stats.get('volatility', stats.get('price_volatility', 0.05))
        features.append(min(1.0, volatility * 10))  # Scale typical 5% volatility to 0.5
        
        # 3. Performance narrative (ROI)
        roi = stats.get('roi', stats.get('return_on_investment', 0.0))
        features.append(np.clip((roi + 1) / 3, 0, 1))  # Normalize -100% to +200%
        
        # 4. Recovery narrative (distance from ATH)
        ath_distance = stats.get('ath_distance', stats.get('distance_from_ath', 0.5))
        features.append(1.0 - ath_distance)  # Closer to ATH = better
        
        # 5. Short-term momentum (7-day change)
        change_7d = stats.get('change_7d', stats.get('7d_change', 0.0))
        features.append(np.clip((change_7d + 0.5) / 1.5, 0, 1))
        
        # 6. Medium-term trend (30-day change)
        change_30d = stats.get('change_30d', stats.get('30d_change', 0.0))
        features.append(np.clip((change_30d + 0.5) / 1.5, 0, 1))
        
        # 7. Long-term trajectory (90-day change)
        change_90d = stats.get('change_90d', stats.get('90d_change', 0.0))
        features.append(np.clip((change_90d + 0.5) / 1.5, 0, 1))
        
        # 8. Liquidity narrative (volume-to-mcap ratio)
        volume = stats.get('volume_24h', stats.get('volume', 1e6))
        vol_mcap_ratio = volume / (market_cap + 1)
        features.append(min(1.0, vol_mcap_ratio * 100))  # Typical ~0.01-0.1
        
        # === MARKET STRUCTURE (6 features) ===
        
        # 9. Liquidity (trading volume)
        if self.log_transform and volume > 0:
            volume_log = np.log10(volume)
            features.append(min(1.0, (volume_log - 5) / 5))  # Scale 10^5 to 10^10
        else:
            features.append(self._normalize_stat('volume', volume))
        
        # 10. Accessibility (exchange count)
        exchange_count = stats.get('exchange_count', stats.get('exchanges', 10))
        features.append(min(1.0, exchange_count / 50))  # Top exchanges ~50
        
        # 11. Legitimacy (top exchange presence)
        top_exchange = stats.get('top_exchange_presence', stats.get('on_top_exchanges', 0.5))
        features.append(top_exchange)
        
        # 12. Market depth (liquidity depth)
        liquidity_depth = stats.get('liquidity_depth', 0.5)
        features.append(liquidity_depth)
        
        # 13. Efficiency (bid-ask spread)
        spread = stats.get('bid_ask_spread', 0.01)
        features.append(1.0 - min(1.0, spread * 100))  # Lower spread = more efficient
        
        # 14. Decentralization (whale concentration - Gini)
        gini = stats.get('gini_coefficient', stats.get('whale_concentration', 0.5))
        features.append(1.0 - gini)  # Lower Gini = more decentralized
        
        # === DEVELOPMENT NARRATIVE (5 features) ===
        
        # 15. Innovation activity (GitHub commits)
        commits = stats.get('github_commits', stats.get('commits', 100))
        features.append(min(1.0, commits / 500))  # Active project ~500+ commits
        
        # 16. Team size (contributors)
        contributors = stats.get('contributors_count', stats.get('contributors', 10))
        features.append(min(1.0, contributors / 50))
        
        # 17. Code quality
        code_quality = stats.get('code_quality_score', 0.5)
        features.append(code_quality)
        
        # 18. Update frequency (commits per month)
        update_freq = stats.get('update_frequency', 20)
        features.append(min(1.0, update_freq / 100))
        
        # 19. Development trend (commit trend)
        dev_trend = stats.get('development_trend', 0.5)
        features.append(dev_trend)
        
        # === COMMUNITY NARRATIVE (6 features) ===
        
        # 20. Adoption (active addresses)
        active_addresses = stats.get('active_addresses', 10000)
        if self.log_transform and active_addresses > 0:
            addresses_log = np.log10(active_addresses)
            features.append(min(1.0, (addresses_log - 3) / 4))  # Scale 10^3 to 10^7
        else:
            features.append(self._normalize_stat('active_addresses', active_addresses))
        
        # 21. Usage (transaction count)
        transactions = stats.get('transaction_count', 1000)
        if self.log_transform and transactions > 0:
            tx_log = np.log10(transactions)
            features.append(min(1.0, (tx_log - 3) / 4))
        else:
            features.append(self._normalize_stat('transaction_count', transactions))
        
        # 22. Holder distribution (holder count)
        holders = stats.get('holder_count', 10000)
        if self.log_transform and holders > 0:
            holders_log = np.log10(holders)
            features.append(min(1.0, (holders_log - 3) / 4))
        else:
            features.append(self._normalize_stat('holder_count', holders))
        
        # 23. Attention narrative (social media mentions)
        social_mentions = stats.get('social_media_mentions', stats.get('social_volume', 100))
        features.append(min(1.0, social_mentions / 10000))  # Viral = 10k+
        
        # 24. Community engagement (Reddit activity)
        reddit_activity = stats.get('reddit_activity', 50)
        features.append(min(1.0, reddit_activity / 500))
        
        # 25. Interest narrative (search volume)
        search_volume = stats.get('search_volume', stats.get('google_trends', 50))
        features.append(search_volume / 100.0)
        
        return features
    
    def _normalize_stat(self, stat_name: str, value: float) -> float:
        """Normalize stat to z-score"""
        if not self.normalize or stat_name not in self.stat_means_:
            return min(1.0, value / 1e9) if value > 1e6 else value
        
        mean = self.stat_means_[stat_name]
        std = self.stat_stds_[stat_name]
        
        z_score = (value - mean) / std
        return np.clip((z_score + 3) / 6, 0, 1)
    
    def get_feature_names_out(self, input_features=None):
        """Return feature names"""
        return np.array([
            # Price
            'crypto_market_cap', 'crypto_volatility', 'crypto_roi', 'crypto_ath_distance',
            'crypto_7d_change', 'crypto_30d_change', 'crypto_90d_change', 'crypto_vol_mcap_ratio',
            
            # Market Structure
            'crypto_volume', 'crypto_exchange_count', 'crypto_top_exchange_presence',
            'crypto_liquidity_depth', 'crypto_bid_ask_spread', 'crypto_decentralization',
            
            # Development
            'crypto_commits', 'crypto_contributors', 'crypto_code_quality',
            'crypto_update_frequency', 'crypto_dev_trend',
            
            # Community
            'crypto_active_addresses', 'crypto_transactions', 'crypto_holders',
            'crypto_social_mentions', 'crypto_reddit_activity', 'crypto_search_volume'
        ])

