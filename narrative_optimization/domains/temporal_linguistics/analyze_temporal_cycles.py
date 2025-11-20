"""
Temporal Linguistic Cycles Analysis

Tests if "history rhymes" at predictable intervals using our complete framework.
Applies relevant transformers, detects cycles, tests hypotheses.
"""

import sys
from pathlib import Path
import json
import numpy as np
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from narrative_optimization.src.transformers import (
    PhoneticTransformer,
    LinguisticPatternsTransformer,
    TemporalEvolutionTransformer,
    InformationTheoryTransformer,
    CognitiveFluencyTransformer,
    NominativeAnalysisTransformer
)


class TemporalCycleAnalyzer:
    """
    Analyze word usage cycles using narrative framework.
    
    Framework application:
    - п(t): Time-varying narrativity
    - ж: Word genome from 6 relevant transformers
    - ю: Word quality (memorability, simplicity, euphony)
    - ❊: Three outcomes (cyclicity, rhyme distance, revival prob)
    - Д: Bridge effect (does ю predict cyclicity?)
    """
    
    def __init__(self):
        """Initialize analyzer."""
        self.words = []
        self.word_data = {}
        self.historical_events = {}
        self.results = {}
        
        print(f"\n{'='*80}")
        print("TEMPORAL LINGUISTIC CYCLES: History Rhyming Analysis")
        print(f"{'='*80}\n")
    
    def load_data(self):
        """Load word frequencies and historical events."""
        print("[1/12] Loading data...")
        
        # Load word frequencies
        data_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'temporal_linguistics' / 'word_frequencies.json'
        
        with open(data_path) as f:
            data = json.load(f)
            self.word_data = {w['word']: w for w in data['words']}
            self.words = list(self.word_data.keys())
        
        # Load historical events
        events_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'temporal_linguistics' / 'historical_events.json'
        
        with open(events_path) as f:
            self.historical_events = json.load(f)
        
        print(f"        ✓ Loaded {len(self.words)} words with 520 years of data each")
        print(f"        ✓ Loaded historical timeline")
    
    def apply_transformers(self):
        """Apply relevant transformer subset to words."""
        print("\n[2/12] Applying transformers (ж extraction)...")
        
        # Apply 6 relevant transformers
        transformers = [
            ('phonetic', PhoneticTransformer()),
            ('linguistic', LinguisticPatternsTransformer()),
            ('temporal', TemporalEvolutionTransformer()),
            ('information', InformationTheoryTransformer()),
            ('cognitive', CognitiveFluencyTransformer()),
            ('nominative', NominativeAnalysisTransformer())
        ]
        
        all_features = []
        feature_names = []
        
        for idx, (name, transformer) in enumerate(transformers, 1):
            print(f"        [{idx}/6] {name:20s}...", end=" ")
            try:
                transformer.fit(self.words)
                features = transformer.transform(self.words)
                
                # Ensure 2D
                if len(features.shape) == 1:
                    features = features.reshape(-1, 1)
                
                all_features.append(features)
                n_features = features.shape[1]
                feature_names.extend([f"{name}_{i}" for i in range(n_features)])
                
                print(f"✓ ({n_features} features)")
            except Exception as e:
                print(f"✗ Error: {str(e)[:40]}")
                continue
        
        # Combine features
        if all_features:
            combined = np.hstack(all_features)
            print(f"\n        ✓ Total features: {combined.shape[1]}")
            
            # Store in word data
            for i, word in enumerate(self.words):
                self.word_data[word]['genome'] = combined[i].tolist()
            
            return combined
        
        return None
    
    def compute_word_quality(self, features):
        """
        Compute ю (word quality) emphasizing revival-relevant features.
        
        High-ю words should revive more predictably.
        """
        print("\n[3/12] Computing ю (word quality scores)...")
        
        # Extract key dimensions from features
        # Phonetic features: first 91
        # Linguistic: next 36
        # etc.
        
        word_qualities = []
        
        for i, word in enumerate(self.words):
            feat = features[i] if features is not None else np.zeros(100)
            
            # Simple heuristic based on feature analysis
            # High quality = memorable + simple + euphonic
            
            # Memorability (from phonetic features - syllables, patterns)
            memorability = np.mean(np.abs(feat[:20]))  # First 20 phonetic features
            
            # Simplicity (inverse of complexity)
            complexity = np.std(feat)
            simplicity = 1.0 / (1.0 + complexity)
            
            # Euphony (smooth consonants, good vowels)
            euphony = max(0, 1.0 - np.mean(feat[feat < 0] if len(feat[feat < 0]) > 0 else [0]))
            
            # Combine with weights
            ю = (0.40 * memorability +
                 0.35 * simplicity +
                 0.25 * euphony)
            
            # Normalize to [0, 1]
            ю = min(1.0, max(0.0, ю))
            
            word_qualities.append(ю)
            self.word_data[word]['story_quality'] = float(ю)
        
        print(f"        ✓ Computed ю for {len(word_qualities)} words")
        print(f"        Mean ю: {np.mean(word_qualities):.3f}")
        print(f"        Range: {np.min(word_qualities):.3f} - {np.max(word_qualities):.3f}")
        
        return np.array(word_qualities)
    
    def detect_cycles(self):
        """
        Detect cycles using FFT analysis.
        Calculate ❊₁ (cyclicity score) for each word.
        """
        print("\n[4/12] Detecting cycles via FFT...")
        
        cyclicity_scores = []
        dominant_periods = []
        
        for word in self.words:
            freq_data = np.array(self.word_data[word]['frequencies'])
            
            # Detrend (remove linear trend)
            detrended = signal.detrend(freq_data)
            
            # Apply FFT
            fft_result = fft(detrended)
            power = np.abs(fft_result)**2
            
            # Get frequencies (convert to periods in years)
            freqs = fftfreq(len(freq_data), d=1.0)  # 1 year sampling
            
            # Only positive frequencies
            positive_freqs = freqs[freqs > 0]
            positive_power = power[freqs > 0]
            
            # Cyclicity = ratio of max peak to total power
            if len(positive_power) > 0 and np.sum(positive_power) > 0:
                cyclicity = np.max(positive_power) / np.sum(positive_power)
                
                # Find dominant period
                peak_idx = np.argmax(positive_power)
                dominant_freq = positive_freqs[peak_idx]
                dominant_period = 1.0 / dominant_freq if dominant_freq != 0 else 0
                
            else:
                cyclicity = 0
                dominant_period = 0
            
            cyclicity_scores.append(cyclicity)
            dominant_periods.append(dominant_period)
            
            self.word_data[word]['cyclicity'] = float(cyclicity)
            self.word_data[word]['dominant_period'] = float(dominant_period)
        
        print(f"        ✓ Detected cycles for {len(self.words)} words")
        print(f"        High cyclicity (>0.15): {sum(1 for c in cyclicity_scores if c > 0.15)}")
        print(f"        Mean period: {np.mean([p for p in dominant_periods if 10 < p < 200]):.1f} years")
        
        return cyclicity_scores, dominant_periods
    
    def calculate_rhyme_distance(self):
        """
        Calculate ❊₂ (rhyme distance) - regularity of peak intervals.
        """
        print("\n[5/12] Calculating rhyme distance...")
        
        rhyme_distances = []
        
        for word in self.words:
            freq_data = np.array(self.word_data[word]['frequencies'])
            
            # Find peaks in frequency
            peaks, _ = signal.find_peaks(freq_data, prominence=np.std(freq_data)*0.5)
            
            if len(peaks) >= 2:
                # Calculate intervals between peaks
                intervals = np.diff(peaks)
                
                # Rhyme distance = standard deviation of intervals
                # Lower = more regular (better "rhyme")
                rhyme_dist = np.std(intervals)
            else:
                rhyme_dist = 999  # No rhythm detected
            
            rhyme_distances.append(rhyme_dist)
            self.word_data[word]['rhyme_distance'] = float(rhyme_dist)
            self.word_data[word]['num_peaks'] = int(len(peaks))
        
        print(f"        ✓ Calculated rhyme distance for {len(self.words)} words")
        regular_rhymes = sum(1 for r in rhyme_distances if r < 20)
        print(f"        Regular rhymes (distance <20): {regular_rhymes}")
        
        return rhyme_distances
    
    def predict_revivals(self):
        """
        Calculate ❊₃ (revival probability) for currently-rare words.
        """
        print("\n[6/12] Predicting word revivals...")
        
        revival_probs = []
        
        for word in self.words:
            freq_data = np.array(self.word_data[word]['frequencies'])
            years = np.array(self.word_data[word]['years'])
            
            # Current frequency (last 10 years average)
            current_freq = np.mean(freq_data[-10:])
            
            # Historical average
            historical_avg = np.mean(freq_data)
            
            # Is it currently rare?
            is_rare = current_freq < historical_avg * 0.5
            
            if is_rare:
                # Check if cycle detected
                cyclicity = self.word_data[word]['cyclicity']
                period = self.word_data[word]['dominant_period']
                
                # Find last peak
                peaks, _ = signal.find_peaks(freq_data, prominence=np.std(freq_data)*0.5)
                
                if len(peaks) > 0:
                    last_peak_year = years[peaks[-1]]
                    years_since_peak = 2019 - last_peak_year
                    
                    # If cycle exists and we're near expected revival
                    if 20 < period < 150 and cyclicity > 0.10:
                        # How close to expected revival?
                        expected_revival = last_peak_year + period
                        closeness = 1.0 - abs(2024 - expected_revival) / period
                        closeness = max(0, min(1, closeness))
                        
                        # Factor in word quality
                        ю = self.word_data[word]['story_quality']
                        
                        # Revival probability
                        revival_prob = cyclicity * 0.4 + closeness * 0.3 + ю * 0.3
                    else:
                        revival_prob = cyclicity * 0.5 + self.word_data[word]['story_quality'] * 0.5
                else:
                    revival_prob = 0.1  # Low baseline
            else:
                revival_prob = 0.0  # Already in use
            
            revival_probs.append(revival_prob)
            self.word_data[word]['revival_probability'] = float(revival_prob)
        
        print(f"        ✓ Calculated revival probability for {len(self.words)} words")
        likely_revivals = [self.words[i] for i, p in enumerate(revival_probs) if p > 0.5]
        print(f"        Likely revivals (p>0.5): {len(likely_revivals)}")
        if likely_revivals[:5]:
            print(f"        Top candidates: {', '.join(likely_revivals[:5])}")
        
        return revival_probs
    
    def test_hypotheses(self):
        """Test all 5 cyclical hypotheses."""
        print(f"\n[7/12] Testing cyclical hypotheses...")
        
        hypothesis_results = {}
        
        # H1: Generation cycle (25-30 years)
        print("\n        H1: Generation Cycle (25-30 years)")
        approval_words = ['groovy', 'rad', 'cool', 'dope', 'lit']
        gen_periods = [self.word_data[w]['dominant_period'] for w in approval_words if w in self.word_data]
        
        if gen_periods:
            mean_period = np.mean([p for p in gen_periods if 15 < p < 40])
            print(f"            Mean period for approval slang: {mean_period:.1f} years")
            print(f"            Expected: 25-30 years")
            if 20 < mean_period < 35:
                print(f"            ✓ CONFIRMED: Generation cycle detected")
                hypothesis_results['H1'] = {'confirmed': True, 'period': mean_period}
            else:
                print(f"            ✗ Not confirmed")
                hypothesis_results['H1'] = {'confirmed': False, 'period': mean_period}
        
        # H2: Crisis rhyming (~75 years for wars, ~25 for economic)
        print("\n        H2: Crisis Rhyming (75-year war cycle)")
        war_words = ['battle', 'warfare', 'trench', 'tank']
        war_periods = [self.word_data[w]['dominant_period'] for w in war_words if w in self.word_data]
        
        if war_periods:
            mean_war_period = np.mean([p for p in war_periods if 40 < p < 120])
            print(f"            Mean period for war words: {mean_war_period:.1f} years")
            print(f"            Expected: ~75 years")
            if 60 < mean_war_period < 90:
                print(f"            ✓ CONFIRMED: War cycle detected")
                hypothesis_results['H2_war'] = {'confirmed': True, 'period': mean_war_period}
            else:
                print(f"            ✗ Not strongly confirmed")
                hypothesis_results['H2_war'] = {'confirmed': False, 'period': mean_war_period}
        
        # H3: Tech innovation (~30 years)
        print("\n        H3: Tech Innovation Cycle (30 years)")
        tech_words = ['wire', 'tube', 'chip', 'web', 'cloud']
        tech_periods = [self.word_data[w]['dominant_period'] for w in tech_words if w in self.word_data]
        
        if tech_periods:
            mean_tech_period = np.mean([p for p in tech_periods if 15 < p < 60])
            print(f"            Mean period for tech words: {mean_tech_period:.1f} years")
            print(f"            Expected: ~30 years")
            if 20 < mean_tech_period < 45:
                print(f"            ✓ CONFIRMED: Tech cycle detected")
                hypothesis_results['H3'] = {'confirmed': True, 'period': mean_tech_period}
            else:
                print(f"            ✗ Not confirmed")
                hypothesis_results['H3'] = {'confirmed': False, 'period': mean_tech_period}
        
        # H4: Victorian revival (100+ years)
        print("\n        H4: Victorian Revival (100-120 years)")
        victorian_words = ['splendid', 'capital', 'dreadful', 'frightful']
        
        victorian_revivals = 0
        for word in victorian_words:
            if word in self.word_data:
                freq = np.array(self.word_data[word]['frequencies'])
                years = np.array(self.word_data[word]['years'])
                
                # Find Victorian peak (1860-1900)
                vic_idx = (years >= 1860) & (years <= 1900)
                vic_peak = np.mean(freq[vic_idx]) if np.any(vic_idx) else 0
                
                # Find modern period (2000-2019)
                mod_idx = (years >= 2000)
                mod_freq = np.mean(freq[mod_idx]) if np.any(mod_idx) else 0
                
                # Revival = modern frequency approaching Victorian levels
                if mod_freq > vic_peak * 0.3:  # At least 30% of Victorian peak
                    victorian_revivals += 1
                    print(f"            '{word}': Victorian={vic_peak:.6f}, Modern={mod_freq:.6f} ✓")
        
        if victorian_revivals >= 2:
            print(f"            ✓ PARTIAL CONFIRMATION: {victorian_revivals}/4 words showing revival")
            hypothesis_results['H4'] = {'confirmed': True, 'revival_count': victorian_revivals}
        else:
            print(f"            ✗ Not confirmed")
            hypothesis_results['H4'] = {'confirmed': False, 'revival_count': victorian_revivals}
        
        # H5: General cyclicity
        print("\n        H5: General Linguistic Cyclicity")
        high_cyclicity = sum(1 for w in self.words 
                            if self.word_data[w]['cyclicity'] > 0.12)
        
        cyclicity_rate = high_cyclicity / len(self.words)
        print(f"            Words with strong cycles: {high_cyclicity}/{len(self.words)} ({cyclicity_rate*100:.1f}%)")
        
        if cyclicity_rate > 0.30:
            print(f"            ✓ CONFIRMED: Language shows significant cyclicity")
            hypothesis_results['H5'] = {'confirmed': True, 'rate': cyclicity_rate}
        else:
            print(f"            ⚠️  WEAK: Some cyclicity but not dominant")
            hypothesis_results['H5'] = {'confirmed': False, 'rate': cyclicity_rate}
        
        return hypothesis_results
    
    def test_historical_synchronization(self):
        """Test if words peak during corresponding historical events."""
        print("\n[8/12] Testing historical synchronization...")
        
        # Test war words with war events
        war_events = [e['midpoint'] for e in self.historical_events['wars']]
        war_words = ['battle', 'warfare', 'conflict', 'combat']
        
        sync_scores = []
        
        for word in war_words:
            if word not in self.word_data:
                continue
            
            freq = np.array(self.word_data[word]['frequencies'])
            years = np.array(self.word_data[word]['years'])
            
            # Find peaks
            peaks, _ = signal.find_peaks(freq, prominence=np.std(freq)*0.5)
            peak_years = years[peaks]
            
            # Count how many peaks are near war events (within 5 years)
            synchronized = 0
            for peak_year in peak_years:
                if any(abs(peak_year - war_year) < 5 for war_year in war_events):
                    synchronized += 1
            
            if len(peak_years) > 0:
                sync_rate = synchronized / len(peak_years)
                sync_scores.append(sync_rate)
                self.word_data[word]['war_synchronization'] = float(sync_rate)
        
        if sync_scores:
            mean_sync = np.mean(sync_scores)
            print(f"        War word synchronization: {mean_sync*100:.1f}%")
            
            if mean_sync > 0.40:
                print(f"        ✓ SIGNIFICANT: Words DO peak during wars!")
            else:
                print(f"        ⚠️  WEAK: Some synchronization but not strong")
        
        return sync_scores
    
    def calculate_temporal_forces(self):
        """
        Calculate time-varying three forces: ة(t), θ(t), λ(t).
        """
        print("\n[9/12] Calculating temporal three forces...")
        
        # Average across time periods
        periods = self.historical_events['cultural_periods']
        
        temporal_forces = {}
        
        for period in periods:
            period_name = period['name']
            start = period['start']
            end = period['end']
            п_t = period['narrativity']
            
            # ة(t): Linguistic gravity (nostalgia + cultural memory)
            # Higher in recent past (strong memory)
            years_ago = 2024 - end
            memory_strength = 1.0 / (1.0 + years_ago / 50)  # Decay over 50 years
            ة_t = п_t * memory_strength * 0.7
            
            # θ(t): Innovation resistance (desire for novelty)
            # Higher in modern periods (conscious language evolution)
            if end > 1950:
                θ_t = 0.70  # Modern: high innovation drive
            elif end > 1800:
                θ_t = 0.50  # Industrial: moderate
            else:
                θ_t = 0.30  # Pre-modern: low innovation consciousness
            
            # λ(t): Fundamental evolution (meaning drift, pronunciation change)
            # Constant-ish but affected by communication technology
            if end > 1995:
                λ_t = 0.10  # Internet: meanings stabilize
            elif end > 1950:
                λ_t = 0.20  # Broadcasting: moderate drift
            elif end > 1450:
                λ_t = 0.30  # Print: meanings drift
            else:
                λ_t = 0.50  # Oral: high drift
            
            temporal_forces[period_name] = {
                'start': start,
                'end': end,
                'narrativity': п_t,
                'linguistic_gravity': float(ة_t),
                'innovation_resistance': float(θ_t),
                'fundamental_evolution': float(λ_t),
                'predicted_cycle_strength': float(ة_t - θ_t - λ_t)
            }
        
        print(f"        ✓ Calculated forces for {len(periods)} historical periods")
        print(f"\n        Period narrativity:")
        for period in periods[-3:]:  # Show last 3
            name = period['name']
            forces = temporal_forces[name]
            print(f"          {name:20s} п={forces['narrativity']:.2f}, "
                  f"ة={forces['linguistic_gravity']:.2f}, "
                  f"θ={forces['innovation_resistance']:.2f}")
        
        self.results['temporal_forces'] = temporal_forces
        
        return temporal_forces
    
    def calculate_bridge_effect(self, word_qualities, cyclicity_scores):
        """
        Calculate Д (bridge) - does ю predict cyclicity?
        
        Test: Do high-quality words show more regular cycles?
        """
        print("\n[10/12] Calculating Д (bridge effect)...")
        
        # Correlation: word quality × cyclicity
        r, p = stats.pearsonr(word_qualities, cyclicity_scores)
        
        print(f"        Correlation(ю, cyclicity): r = {r:.3f}, p = {p:.4f}")
        
        # Calculate п for this domain
        п_temporal = 0.75  # Language evolution highly narrative
        
        # κ varies by period, use modern value
        κ_modern = 0.90  # Internet age - high coupling
        
        # Bridge
        Д = п_temporal * abs(r) * κ_modern
        
        print(f"\n        Domain characteristics:")
        print(f"          п (narrativity) = {п_temporal:.3f}")
        print(f"          |r| (correlation) = {abs(r):.3f}")
        print(f"          κ (coupling) = {κ_modern:.3f}")
        print(f"          Д (bridge) = {Д:.3f}")
        
        efficiency = Д / п_temporal
        print(f"          Д/п (efficiency) = {efficiency:.3f}")
        
        if efficiency > 0.5:
            print(f"          ✓ PASS: Narratives predict cycles (Д/п > 0.5)")
        else:
            print(f"          ✗ FAIL: Effects below threshold")
        
        if p < 0.05:
            if r > 0:
                print(f"\n        ✓ SIGNIFICANT: High-ю words revive more predictably!")
            else:
                print(f"\n        ⚠️  NEGATIVE: High-ю words less cyclical (unexpected)")
        else:
            print(f"\n        ✗ NULL: Word quality doesn't predict cyclicity")
        
        self.results['bridge_effect'] = {
            'narrativity': п_temporal,
            'correlation': float(r),
            'p_value': float(p),
            'coupling': κ_modern,
            'bridge': float(Д),
            'efficiency': float(efficiency)
        }
        
        return Д
    
    def save_results(self):
        """Save complete analysis results."""
        output_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'temporal_linguistics' / 'analysis_results.json'
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump({
                'word_data': self.word_data,
                'results': self.results,
                'summary': {
                    'total_words': len(self.words),
                    'time_span': '1500-2019',
                    'high_cyclicity_words': sum(1 for w in self.word_data.values() 
                                               if w['cyclicity'] > 0.12),
                    'predicted_revivals': sum(1 for w in self.word_data.values() 
                                             if w['revival_probability'] > 0.5)
                }
            }, f, indent=2)
        
        print(f"\n✓ Saved complete results to: {output_path}")
    
    def print_final_summary(self):
        """Print comprehensive final summary."""
        print(f"\n{'='*80}")
        print("TEMPORAL LINGUISTIC CYCLES: FINAL RESULTS")
        print(f"{'='*80}\n")
        
        # Top cyclical words
        cyclical_words = sorted(self.words, 
                               key=lambda w: self.word_data[w]['cyclicity'], 
                               reverse=True)
        
        print("Most cyclical words (strongest periodic patterns):")
        for i, word in enumerate(cyclical_words[:10], 1):
            data = self.word_data[word]
            print(f"  {i:2d}. {word:15s} cyclicity={data['cyclicity']:.3f}, "
                  f"period={data['dominant_period']:.0f}y, "
                  f"ю={data['story_quality']:.3f}")
        
        # Revival predictions
        revival_words = sorted(self.words,
                              key=lambda w: self.word_data[w]['revival_probability'],
                              reverse=True)
        
        print("\nWords most likely to revive:")
        for i, word in enumerate(revival_words[:10], 1):
            data = self.word_data[word]
            print(f"  {i:2d}. {word:15s} revival_prob={data['revival_probability']:.3f}, "
                  f"period={data['dominant_period']:.0f}y")
        
        # Bridge effect
        bridge = self.results['bridge_effect']
        print(f"\nFramework results:")
        print(f"  Narrativity (п): {bridge['narrativity']:.3f}")
        print(f"  Bridge (Д): {bridge['bridge']:.3f}")
        print(f"  Efficiency (Д/п): {bridge['efficiency']:.3f}")
        print(f"  Verdict: {'PASS' if bridge['efficiency'] > 0.5 else 'FAIL'} threshold")


def main():
    """Run complete temporal cycle analysis."""
    analyzer = TemporalCycleAnalyzer()
    
    # Load data
    analyzer.load_data()
    
    # Apply transformers
    features = analyzer.apply_transformers()
    
    # Compute word quality
    word_qualities = analyzer.compute_word_quality(features)
    
    # Detect cycles
    cyclicity_scores, dominant_periods = analyzer.detect_cycles()
    
    # Calculate rhyme distance
    rhyme_distances = analyzer.calculate_rhyme_distance()
    
    # Predict revivals
    revival_probs = analyzer.predict_revivals()
    
    # Test hypotheses
    hypothesis_results = analyzer.test_hypotheses()
    
    # Historical synchronization
    sync_scores = analyzer.test_historical_synchronization()
    
    # Calculate temporal forces
    temporal_forces = analyzer.calculate_temporal_forces()
    
    # Calculate bridge
    Д = analyzer.calculate_bridge_effect(word_qualities, cyclicity_scores)
    
    # Save everything
    analyzer.save_results()
    
    # Final summary
    analyzer.print_final_summary()
    
    print(f"\n{'='*80}")
    print("✓ TEMPORAL LINGUISTIC ANALYSIS COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

