"""
UFC Comprehensive Feature Extraction

Extracts 150-200 features from all 61 available columns in real UFC data.
Maximum feature engineering for optimization.

Categories:
- Physical Performance: 40+ features
- Nominative: 35+ features  
- Temporal/Career: 25+ features
- Context: 20+ features
- Interactions: 30+ features
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import re

class UFCComprehensiveFeatureExtractor:
    """Extract maximum features from UFC fight data"""
    
    def __init__(self):
        self.feature_names = []
        
    def load_data(self, path=None):
        """Load real UFC data"""
        if path is None:
            path = Path('data/domains/UFC-DataLab/data/merged_stats_n_scorecards/merged_stats_n_scorecards.csv')
        
        print(f"Loading data from: {path}")
        df = pd.read_csv(path)
        print(f"✓ Loaded {len(df)} fights with {len(df.columns)} columns")
        
        return df
    
    def extract_all_features(self, df):
        """Extract comprehensive feature set"""
        
        print("\n" + "="*80)
        print("EXTRACTING COMPREHENSIVE FEATURES")
        print("="*80)
        
        features_list = []
        outcomes = []
        
        for idx, row in df.iterrows():
            if idx % 1000 == 0:
                print(f"  Processing fight {idx}/{len(df)}...")
            
            try:
                # Outcome
                red_won = 1 if row['red_fighter_result'] == 'W' else 0
                outcomes.append(red_won)
                
                # Extract all feature categories
                features = {}
                
                # 1. PHYSICAL PERFORMANCE (40+ features)
                phys = self._extract_physical_features(row)
                features.update(phys)
                
                # 2. NOMINATIVE (35+ features)
                nom = self._extract_nominative_features(row)
                features.update(nom)
                
                # 3. TEMPORAL/CAREER (25+ features)
                temp = self._extract_temporal_features(row)
                features.update(temp)
                
                # 4. CONTEXT (20+ features)
                ctx = self._extract_context_features(row)
                features.update(ctx)
                
                # 5. INTERACTIONS (30+ features)
                inter = self._extract_interaction_features(row, phys, nom, ctx)
                features.update(inter)
                
                features_list.append(features)
                
            except Exception as e:
                if idx < 10:  # Only print first few errors
                    print(f"  Error on row {idx}: {e}")
                continue
        
        X_df = pd.DataFrame(features_list)
        y = np.array(outcomes[:len(features_list)])
        
        self.feature_names = list(X_df.columns)
        
        print(f"\n✓ Extracted {len(X_df.columns)} features from {len(X_df)} fights")
        print(f"\nFeature breakdown:")
        print(f"  Physical: {sum(1 for f in self.feature_names if any(x in f for x in ['strike', 'td', 'ctrl', 'kd', 'sub', 'rev']))}")
        print(f"  Nominative: {sum(1 for f in self.feature_names if any(x in f for x in ['name', 'nick', 'vowel', 'syllable']))}")
        print(f"  Temporal: {sum(1 for f in self.feature_names if any(x in f for x in ['year', 'month', 'day']))}")
        print(f"  Context: {sum(1 for f in self.feature_names if any(x in f for x in ['title', 'bonus', 'method', 'round', 'weight', 'location']))}")
        print(f"  Interactions: {sum(1 for f in self.feature_names if '_x_' in f)}")
        
        return X_df, y
    
    def _extract_physical_features(self, row):
        """Extract 40+ physical performance features"""
        
        def safe_parse_fraction(val):
            """Parse '75 of 144' format"""
            if pd.isna(val) or val == '---':
                return 0, 0
            try:
                parts = str(val).split(' of ')
                return int(parts[0]), int(parts[1])
            except:
                return 0, 0
        
        def safe_float(val, default=0):
            if pd.isna(val) or val == '---' or val == '-':
                return default
            try:
                return float(val)
            except:
                return default
        
        def safe_int(val, default=0):
            if pd.isna(val) or val == '---' or val == '-':
                return default
            try:
                return int(val)
            except:
                return default
        
        def parse_time_seconds(val):
            if pd.isna(val) or val == '---':
                return 0
            try:
                parts = str(val).split(':')
                return int(parts[0]) * 60 + int(parts[1])
            except:
                return 0
        
        features = {}
        
        # Striking stats (landed/attempted)
        red_sig_str_landed, red_sig_str_att = safe_parse_fraction(row['red_fighter_sig_str'])
        blue_sig_str_landed, blue_sig_str_att = safe_parse_fraction(row['blue_fighter_sig_str'])
        
        features['red_sig_str_landed'] = red_sig_str_landed
        features['red_sig_str_att'] = red_sig_str_att
        features['red_sig_str_pct'] = safe_float(row['red_fighter_sig_str_pct']) / 100
        features['blue_sig_str_landed'] = blue_sig_str_landed
        features['blue_sig_str_att'] = blue_sig_str_att
        features['blue_sig_str_pct'] = safe_float(row['blue_fighter_sig_str_pct']) / 100
        
        features['sig_str_diff'] = abs(features['red_sig_str_pct'] - features['blue_sig_str_pct'])
        features['sig_str_volume_diff'] = abs(red_sig_str_landed - blue_sig_str_landed)
        
        # Total strikes
        red_total_str_landed, red_total_str_att = safe_parse_fraction(row['red_fighter_total_str'])
        blue_total_str_landed, blue_total_str_att = safe_parse_fraction(row['blue_fighter_total_str'])
        
        features['red_total_str_landed'] = red_total_str_landed
        features['blue_total_str_landed'] = blue_total_str_landed
        features['total_str_diff'] = abs(red_total_str_landed - blue_total_str_landed)
        
        # Striking targets (head/body/leg)
        red_head_landed, _ = safe_parse_fraction(row['red_fighter_sig_str_head'])
        blue_head_landed, _ = safe_parse_fraction(row['blue_fighter_sig_str_head'])
        red_body_landed, _ = safe_parse_fraction(row['red_fighter_sig_str_body'])
        blue_body_landed, _ = safe_parse_fraction(row['blue_fighter_sig_str_body'])
        red_leg_landed, _ = safe_parse_fraction(row['red_fighter_sig_str_leg'])
        blue_leg_landed, _ = safe_parse_fraction(row['blue_fighter_sig_str_leg'])
        
        features['red_head_strikes'] = red_head_landed
        features['blue_head_strikes'] = blue_head_landed
        features['red_body_strikes'] = red_body_landed
        features['blue_body_strikes'] = blue_body_landed
        features['red_leg_strikes'] = red_leg_landed
        features['blue_leg_strikes'] = blue_leg_landed
        
        features['head_strike_diff'] = abs(red_head_landed - blue_head_landed)
        features['body_strike_diff'] = abs(red_body_landed - blue_body_landed)
        features['leg_strike_diff'] = abs(red_leg_landed - blue_leg_landed)
        
        # Striking positions (distance/clinch/ground)
        red_distance, _ = safe_parse_fraction(row['red_fighter_sig_str_distance'])
        blue_distance, _ = safe_parse_fraction(row['blue_fighter_sig_str_distance'])
        red_clinch, _ = safe_parse_fraction(row['red_fighter_sig_str_clinch'])
        blue_clinch, _ = safe_parse_fraction(row['blue_fighter_sig_str_clinch'])
        red_ground, _ = safe_parse_fraction(row['red_fighter_sig_str_ground'])
        blue_ground, _ = safe_parse_fraction(row['blue_fighter_sig_str_ground'])
        
        features['red_distance_strikes'] = red_distance
        features['blue_distance_strikes'] = blue_distance
        features['red_clinch_strikes'] = red_clinch
        features['blue_clinch_strikes'] = blue_clinch
        features['red_ground_strikes'] = red_ground
        features['blue_ground_strikes'] = blue_ground
        
        # Takedowns
        red_td_landed, red_td_att = safe_parse_fraction(row['red_fighter_TD'])
        blue_td_landed, blue_td_att = safe_parse_fraction(row['blue_fighter_TD'])
        
        features['red_td_landed'] = red_td_landed
        features['red_td_att'] = red_td_att
        features['red_td_pct'] = safe_float(row['red_fighter_TD_pct']) / 100
        features['blue_td_landed'] = blue_td_landed
        features['blue_td_att'] = blue_td_att
        features['blue_td_pct'] = safe_float(row['blue_fighter_TD_pct']) / 100
        
        features['td_diff'] = abs(red_td_landed - blue_td_landed)
        
        # Submissions
        features['red_sub_att'] = safe_int(row['red_fighter_sub_att'])
        features['blue_sub_att'] = safe_int(row['blue_fighter_sub_att'])
        features['sub_att_diff'] = abs(features['red_sub_att'] - features['blue_sub_att'])
        
        # Knockdowns
        features['red_kd'] = safe_int(row['red_fighter_KD'])
        features['blue_kd'] = safe_int(row['blue_fighter_KD'])
        features['kd_diff'] = abs(features['red_kd'] - features['blue_kd'])
        features['total_kd'] = features['red_kd'] + features['blue_kd']
        
        # Reversals
        features['red_rev'] = safe_int(row['red_fighter_rev'])
        features['blue_rev'] = safe_int(row['blue_fighter_rev'])
        features['rev_diff'] = abs(features['red_rev'] - features['blue_rev'])
        
        # Control time
        red_ctrl = parse_time_seconds(row['red_fighter_ctrl'])
        blue_ctrl = parse_time_seconds(row['blue_fighter_ctrl'])
        
        features['red_ctrl'] = red_ctrl
        features['blue_ctrl'] = blue_ctrl
        features['ctrl_diff'] = abs(red_ctrl - blue_ctrl)
        features['ctrl_ratio'] = red_ctrl / (blue_ctrl + 1)  # Avoid div by zero
        features['total_ctrl'] = red_ctrl + blue_ctrl
        
        return features
    
    def _extract_nominative_features(self, row):
        """Extract 35+ nominative features"""
        
        features = {}
        
        # Fighter names
        red_name = str(row['red_fighter_name']).upper()
        blue_name = str(row['blue_fighter_name']).upper()
        
        # Basic name features
        features['red_name_len'] = len(red_name)
        features['blue_name_len'] = len(blue_name)
        features['name_len_diff'] = abs(len(red_name) - len(blue_name))
        features['name_len_ratio'] = len(red_name) / (len(blue_name) + 1)
        
        # Name components (first/last)
        red_parts = red_name.split()
        blue_parts = blue_name.split()
        
        features['red_name_parts'] = len(red_parts)
        features['blue_name_parts'] = len(blue_parts)
        features['name_parts_diff'] = abs(len(red_parts) - len(blue_parts))
        
        if len(red_parts) > 0:
            features['red_first_name_len'] = len(red_parts[0])
            features['red_last_name_len'] = len(red_parts[-1]) if len(red_parts) > 1 else 0
        else:
            features['red_first_name_len'] = 0
            features['red_last_name_len'] = 0
        
        if len(blue_parts) > 0:
            features['blue_first_name_len'] = len(blue_parts[0])
            features['blue_last_name_len'] = len(blue_parts[-1]) if len(blue_parts) > 1 else 0
        else:
            features['blue_first_name_len'] = 0
            features['blue_last_name_len'] = 0
        
        # Phonetic features (vowels, consonants, syllables proxy)
        red_vowels = len([c for c in red_name if c in 'AEIOU'])
        blue_vowels = len([c for c in blue_name if c in 'AEIOU'])
        red_consonants = len([c for c in red_name if c.isalpha() and c not in 'AEIOU'])
        blue_consonants = len([c for c in blue_name if c.isalpha() and c not in 'AEIOU'])
        
        features['red_vowels'] = red_vowels
        features['blue_vowels'] = blue_vowels
        features['red_consonants'] = red_consonants
        features['blue_consonants'] = blue_consonants
        features['vowel_diff'] = abs(red_vowels - blue_vowels)
        features['consonant_diff'] = abs(red_consonants - blue_consonants)
        
        features['red_vowel_ratio'] = red_vowels / (len(red_name) + 1)
        features['blue_vowel_ratio'] = blue_vowels / (len(blue_name) + 1)
        
        # Nicknames
        red_nick = str(row['red_fighter_nickname']) if pd.notna(row['red_fighter_nickname']) else ""
        blue_nick = str(row['blue_fighter_nickname']) if pd.notna(row['blue_fighter_nickname']) else ""
        
        features['red_has_nickname'] = 1 if red_nick and red_nick != 'nan' and red_nick != '-' else 0
        features['blue_has_nickname'] = 1 if blue_nick and blue_nick != 'nan' and blue_nick != '-' else 0
        features['both_have_nicknames'] = features['red_has_nickname'] * features['blue_has_nickname']
        features['neither_has_nickname'] = (1 - features['red_has_nickname']) * (1 - features['blue_has_nickname'])
        
        features['red_nickname_len'] = len(red_nick) if features['red_has_nickname'] else 0
        features['blue_nickname_len'] = len(blue_nick) if features['blue_has_nickname'] else 0
        
        # Name memorability (shorter, punchier names = more memorable)
        features['red_name_memorability'] = 1 / (len(red_name) / 10 + 1)
        features['blue_name_memorability'] = 1 / (len(blue_name) / 10 + 1)
        features['memorability_diff'] = abs(features['red_name_memorability'] - features['blue_name_memorability'])
        
        # Name "hardness" (more consonants = harsher sound)
        features['red_name_hardness'] = red_consonants / (len(red_name) + 1)
        features['blue_name_hardness'] = blue_consonants / (len(blue_name) + 1)
        
        # Cultural patterns (detect common patterns)
        # Slavic endings
        features['red_slavic'] = 1 if any(red_name.endswith(end) for end in ['OV', 'EV', 'SKI', 'VIC', 'VICH']) else 0
        features['blue_slavic'] = 1 if any(blue_name.endswith(end) for end in ['OV', 'EV', 'SKI', 'VIC', 'VICH']) else 0
        
        # Brazilian/Portuguese
        features['red_brazilian'] = 1 if any(part in red_name for part in ['SILVA', 'SANTOS', 'OLIVEIRA', 'SOUZA']) else 0
        features['blue_brazilian'] = 1 if any(part in blue_name for part in ['SILVA', 'SANTOS', 'OLIVEIRA', 'SOUZA']) else 0
        
        return features
    
    def _extract_temporal_features(self, row):
        """Extract 25+ temporal/career features"""
        
        features = {}
        
        # Parse date
        try:
            date_str = row['event_date']
            date_obj = datetime.strptime(date_str, '%d/%m/%Y')
            
            features['year'] = date_obj.year
            features['month'] = date_obj.month
            features['day_of_week'] = date_obj.weekday()
            features['quarter'] = (date_obj.month - 1) // 3 + 1
            
            # Era classification
            features['era_early'] = 1 if date_obj.year < 2015 else 0
            features['era_middle'] = 1 if 2015 <= date_obj.year < 2020 else 0
            features['era_recent'] = 1 if date_obj.year >= 2020 else 0
            
            # Weekend fight
            features['is_weekend'] = 1 if date_obj.weekday() >= 5 else 0
            
        except:
            features['year'] = 2020
            features['month'] = 6
            features['day_of_week'] = 0
            features['quarter'] = 2
            features['era_early'] = 0
            features['era_middle'] = 1
            features['era_recent'] = 0
            features['is_weekend'] = 0
        
        # Judge scores (proxy for career success/dominance)
        try:
            red_pts = str(row['red_fighter_total_pts'])
            blue_pts = str(row['blue_fighter_total_pts'])
            
            if red_pts and red_pts != 'nan' and red_pts != '---' and red_pts != '-':
                red_scores = [int(x) for x in red_pts.split() if x.isdigit()]
                if red_scores:
                    features['red_avg_score'] = np.mean(red_scores)
                    features['red_score_consistency'] = np.std(red_scores) if len(red_scores) > 1 else 0
                else:
                    features['red_avg_score'] = 0
                    features['red_score_consistency'] = 0
            else:
                features['red_avg_score'] = 0
                features['red_score_consistency'] = 0
            
            if blue_pts and blue_pts != 'nan' and blue_pts != '---' and blue_pts != '-':
                blue_scores = [int(x) for x in blue_pts.split() if x.isdigit()]
                if blue_scores:
                    features['blue_avg_score'] = np.mean(blue_scores)
                    features['blue_score_consistency'] = np.std(blue_scores) if len(blue_scores) > 1 else 0
                else:
                    features['blue_avg_score'] = 0
                    features['blue_score_consistency'] = 0
            else:
                features['blue_avg_score'] = 0
                features['blue_score_consistency'] = 0
            
            features['score_diff'] = abs(features['red_avg_score'] - features['blue_avg_score'])
            
        except:
            features['red_avg_score'] = 0
            features['red_score_consistency'] = 0
            features['blue_avg_score'] = 0
            features['blue_score_consistency'] = 0
            features['score_diff'] = 0
        
        return features
    
    def _extract_context_features(self, row):
        """Extract 20+ context features"""
        
        features = {}
        
        # Bout type (title fight)
        bout_type = str(row['bout_type']) if pd.notna(row['bout_type']) else ""
        features['is_title_fight'] = 1 if 'Title' in bout_type else 0
        features['is_championship'] = 1 if 'Championship' in bout_type else 0
        
        # Weight class
        weight_classes = {
            'Flyweight': 1, 'Bantamweight': 2, 'Featherweight': 3,
            'Lightweight': 4, 'Welterweight': 5, 'Middleweight': 6,
            'Light Heavyweight': 7, 'Heavyweight': 8
        }
        
        weight_class_num = 0
        for wc, num in weight_classes.items():
            if wc in bout_type:
                weight_class_num = num
                break
        
        features['weight_class_num'] = weight_class_num
        features['is_heavyweight_division'] = 1 if weight_class_num >= 7 else 0
        features['is_lightweight_division'] = 1 if weight_class_num <= 4 else 0
        
        # Women's fight
        features['is_womens_fight'] = 1 if 'Women' in bout_type else 0
        
        # Bonus
        bonus = str(row['bonus']) if pd.notna(row['bonus']) else ""
        features['has_bonus'] = 1 if bonus and bonus != 'nan' and bonus != '-' and bonus != '---' else 0
        features['is_belt'] = 1 if bonus == 'belt' else 0
        features['is_perf_bonus'] = 1 if 'perf' in bonus.lower() else 0
        features['is_fight_bonus'] = 1 if 'fight' in bonus.lower() else 0
        
        # Method
        method = str(row['method']) if pd.notna(row['method']) else ""
        features['method_ko'] = 1 if 'KO' in method or 'TKO' in method else 0
        features['method_sub'] = 1 if 'Sub' in method else 0
        features['method_decision'] = 1 if 'Decision' in method else 0
        features['is_finish'] = features['method_ko'] + features['method_sub']
        
        # Decision type
        features['decision_unanimous'] = 1 if 'Unanimous' in method else 0
        features['decision_split'] = 1 if 'Split' in method else 0
        features['decision_majority'] = 1 if 'Majority' in method else 0
        
        # Round
        try:
            features['round'] = int(row['round'])
            features['early_finish'] = 1 if features['round'] == 1 else 0
            features['late_finish'] = 1 if features['round'] >= 3 else 0
        except:
            features['round'] = 3
            features['early_finish'] = 0
            features['late_finish'] = 0
        
        # Time format (3 vs 5 round fight)
        time_format = str(row['time_format']) if pd.notna(row['time_format']) else ""
        features['is_5_round_fight'] = 1 if '5 Rnd' in time_format or '5-5-5-5-5' in time_format else 0
        
        # Location (major UFC markets)
        location = str(row['event_location']) if pd.notna(row['event_location']) else ""
        features['location_las_vegas'] = 1 if 'Las Vegas' in location else 0
        features['location_abu_dhabi'] = 1 if 'Abu Dhabi' in location else 0
        features['location_usa'] = 1 if 'United States' in location else 0
        features['location_international'] = 1 if not features['location_usa'] else 0
        
        # Referee (could indicate fight style)
        referee = str(row['referee']) if pd.notna(row['referee']) else ""
        features['referee_herb_dean'] = 1 if 'Herb Dean' in referee else 0
        features['referee_marc_goddard'] = 1 if 'Marc Goddard' in referee else 0
        features['referee_jason_herzog'] = 1 if 'Jason Herzog' in referee else 0
        
        return features
    
    def _extract_interaction_features(self, row, phys, nom, ctx):
        """Extract 30+ interaction features"""
        
        features = {}
        
        # Physical × Context
        if 'is_title_fight' in ctx and 'red_sig_str_pct' in phys:
            features['title_x_strike_diff'] = ctx['is_title_fight'] * phys['sig_str_diff']
            features['title_x_kd_total'] = ctx['is_title_fight'] * phys['total_kd']
            features['title_x_ctrl_diff'] = ctx['is_title_fight'] * phys['ctrl_diff']
        
        # Nominative × Physical
        if 'both_have_nicknames' in nom and 'sig_str_diff' in phys:
            features['nickname_x_strike'] = nom['both_have_nicknames'] * phys['sig_str_diff']
            features['nickname_x_finish'] = nom['both_have_nicknames'] * ctx.get('is_finish', 0)
        
        # Name memorability × Performance
        if 'red_name_memorability' in nom and 'red_sig_str_pct' in phys:
            features['red_mem_x_perf'] = nom['red_name_memorability'] * phys['red_sig_str_pct']
            features['blue_mem_x_perf'] = nom['blue_name_memorability'] * phys['blue_sig_str_pct']
        
        # Physical dominance indicators
        if 'red_kd' in phys and 'red_ctrl' in phys:
            features['red_dominance'] = phys['red_kd'] * 10 + phys['red_ctrl'] / 60
            features['blue_dominance'] = phys['blue_kd'] * 10 + phys['blue_ctrl'] / 60
            features['dominance_diff'] = abs(features['red_dominance'] - features['blue_dominance'])
        
        # Style clash (striking vs grappling)
        if 'red_td_landed' in phys and 'red_sig_str_landed' in phys:
            red_grappler = 1 if phys['red_td_landed'] > 2 or phys['red_sub_att'] > 1 else 0
            blue_grappler = 1 if phys['blue_td_landed'] > 2 or phys['blue_sub_att'] > 1 else 0
            red_striker = 1 if phys['red_sig_str_landed'] > 50 and phys['red_td_landed'] <= 1 else 0
            blue_striker = 1 if phys['blue_sig_str_landed'] > 50 and phys['blue_td_landed'] <= 1 else 0
            
            features['red_grappler'] = red_grappler
            features['blue_grappler'] = blue_grappler
            features['red_striker'] = red_striker
            features['blue_striker'] = blue_striker
            features['style_clash'] = (red_striker * blue_grappler) + (red_grappler * blue_striker)
            features['grappler_vs_grappler'] = red_grappler * blue_grappler
            features['striker_vs_striker'] = red_striker * blue_striker
        
        # Volume vs Efficiency
        if 'red_sig_str_att' in phys and 'red_sig_str_pct' in phys:
            features['red_volume_fighter'] = 1 if phys['red_sig_str_att'] > 100 else 0
            features['blue_volume_fighter'] = 1 if phys['blue_sig_str_att'] > 100 else 0
            features['red_efficient_fighter'] = 1 if phys['red_sig_str_pct'] > 0.5 else 0
            features['blue_efficient_fighter'] = 1 if phys['blue_sig_str_pct'] > 0.5 else 0
        
        # Name length × Weight class
        if 'weight_class_num' in ctx and 'red_name_len' in nom:
            features['heavyweight_x_short_name'] = ctx['is_heavyweight_division'] * (1 if nom['red_name_len'] < 12 else 0)
        
        # Cultural clash
        if 'red_slavic' in nom and 'blue_brazilian' in nom:
            features['cultural_clash'] = abs(nom['red_slavic'] - nom['blue_slavic']) + abs(nom['red_brazilian'] - nom['blue_brazilian'])
        
        # Finish context
        if 'is_finish' in ctx and 'early_finish' in ctx:
            features['early_knockout'] = ctx['is_finish'] * ctx['early_finish']
            features['late_finish_struggle'] = ctx['is_finish'] * ctx['late_finish']
        
        # Title fight × Even matchup
        if 'is_title_fight' in ctx and 'sig_str_diff' in phys:
            even_matchup = 1 if phys['sig_str_diff'] < 0.1 and phys['kd_diff'] == 0 else 0
            features['title_x_even'] = ctx['is_title_fight'] * even_matchup
        
        return features


def main():
    """Run comprehensive feature extraction"""
    
    print("="*80)
    print("UFC COMPREHENSIVE FEATURE EXTRACTION")
    print("="*80)
    
    extractor = UFCComprehensiveFeatureExtractor()
    
    # Load data
    df = extractor.load_data()
    
    # Extract all features
    X_df, y = extractor.extract_all_features(df)
    
    # Clean data
    X_df = X_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    print(f"\n✓ Final feature matrix: {X_df.shape}")
    print(f"  Features: {X_df.shape[1]}")
    print(f"  Samples: {X_df.shape[0]}")
    print(f"  Red wins: {y.sum()} ({100*y.mean():.1f}%)")
    
    # Save
    output_dir = Path('narrative_optimization/domains/ufc')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    X_df.to_csv(output_dir / 'ufc_comprehensive_features.csv', index=False)
    np.save(output_dir / 'ufc_comprehensive_outcomes.npy', y)
    
    # Save feature names
    with open(output_dir / 'ufc_feature_names.txt', 'w') as f:
        for fname in extractor.feature_names:
            f.write(f"{fname}\n")
    
    print(f"\n✓ Saved features to: {output_dir}")
    print(f"  - ufc_comprehensive_features.csv")
    print(f"  - ufc_comprehensive_outcomes.npy")
    print(f"  - ufc_feature_names.txt")
    
    return X_df, y


if __name__ == "__main__":
    main()

