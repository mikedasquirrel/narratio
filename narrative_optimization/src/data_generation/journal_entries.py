"""
Journal Entry Generator for Wellness Domain

Generate synthetic journal entries with controlled self-perception,
growth mindset, and narrative potential dimensions.
"""

from typing import List, Dict, Any
import numpy as np
import random
from datetime import datetime, timedelta


class JournalEntryGenerator:
    """
    Generate journal entries with controlled narrative patterns.
    
    Tests self-perception and narrative potential transformers.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        random.seed(random_state)
        np.random.seed(random_state)
        
        # Growth mindset phrases
        self.growth_phrases = [
            "I'm learning", "I'm developing", "I'm growing", "I'm improving",
            "I'm becoming", "I realized", "I discovered", "I'm progressing"
        ]
        
        self.fixed_phrases = [
            "I am", "I've always been", "I remain", "I'm stuck", "I can't change"
        ]
        
        # Agency phrases
        self.high_agency = [
            "I decided", "I made", "I created", "I achieved", "I accomplished",
            "I took action", "I initiated", "I pursued"
        ]
        
        self.low_agency = [
            "it happened", "I was affected by", "circumstances led", "I ended up",
            "I couldn't", "things were", "I had to"
        ]
        
        # Future orientation
        self.future_words = [
            "will", "going to", "plan to", "hope to", "aspire to", "want to",
            "excited for", "looking forward", "future", "tomorrow", "next"
        ]
        
        # Emotional words
        self.positive_emotions = [
            "happy", "grateful", "proud", "hopeful", "confident", "peaceful",
            "excited", "content", "fulfilled", "encouraged"
        ]
        
        self.negative_emotions = [
            "sad", "worried", "anxious", "frustrated", "discouraged", "overwhelmed",
            "stuck", "lost", "confused", "uncertain"
        ]
    
    def generate_entry(
        self,
        wellbeing_level: float,  # 0-1
        growth_trajectory: str,  # 'improving', 'stable', 'declining'
        day_number: int = 1
    ) -> str:
        """
        Generate a single journal entry.
        
        Parameters
        ----------
        wellbeing_level : float
            Current wellbeing (0=low, 1=high)
        growth_trajectory : str
            Narrative trajectory type
        day_number : int
            Day in sequence (affects narrative)
        
        Returns
        -------
        entry : str
            Generated journal entry
        """
        parts = []
        
        # 1. Opening with date context
        parts.append(f"Day {day_number}.")
        
        # 2. Current state reflection (agency varies with wellbeing)
        if growth_trajectory == 'improving':
            # High agency, growth mindset
            action = random.choice(self.high_agency + self.growth_phrases)
            parts.append(f"Today {action} something meaningful.")
        elif growth_trajectory == 'declining':
            # Low agency, fixed mindset
            action = random.choice(self.low_agency + self.fixed_phrases)
            parts.append(f"Today {action} challenging.")
        else:
            # Balanced
            parts.append("Today was a typical day.")
        
        # 3. Emotional state
        if wellbeing_level > 0.6:
            emotion = random.choice(self.positive_emotions)
            parts.append(f"I feel {emotion}.")
        elif wellbeing_level < 0.4:
            emotion = random.choice(self.negative_emotions)
            parts.append(f"I feel {emotion}.")
        else:
            parts.append("I feel okay, neither great nor terrible.")
        
        # 4. Self-perception patterns
        if growth_trajectory == 'improving':
            self_reflection = random.choice([
                "I'm noticing positive changes in myself.",
                "I can see how I'm developing.",
                "I'm proud of my progress.",
                "I realize I'm capable of growth."
            ])
        elif growth_trajectory == 'declining':
            self_reflection = random.choice([
                "I don't think I can change this.",
                "I've always been this way.",
                "I'm not sure I'm capable.",
                "I feel stuck in patterns."
            ])
        else:
            self_reflection = "I'm maintaining my routine."
        
        parts.append(self_reflection)
        
        # 5. Future orientation (narrative potential)
        if growth_trajectory == 'improving':
            future = random.choice([
                f"Tomorrow I {random.choice(self.future_words[:3])} try something new.",
                "I'm excited about future possibilities.",
                "I'm looking forward to continued growth.",
                "I plan to build on today's insights."
            ])
            parts.append(future)
        elif growth_trajectory == 'stable':
            parts.append("I'll continue as I have been.")
        # declining has no future focus
        
        return " ".join(parts)
    
    def generate_longitudinal_series(
        self,
        n_days: int,
        trajectory_type: str,  # 'growth', 'decline', 'recovery', 'stable'
        starting_wellbeing: float = 0.5
    ) -> Dict[str, Any]:
        """
        Generate time series of journal entries.
        
        Parameters
        ----------
        n_days : int
            Number of days to generate
        trajectory_type : str
            Overall trajectory pattern
        starting_wellbeing : float
            Initial wellbeing level
        
        Returns
        -------
        series : dict
            {
                'entries': [texts],
                'wellbeing_levels': [scores],
                'trajectory_type': str,
                'dates': [dates]
            }
        """
        entries = []
        wellbeing_levels = []
        
        current_wellbeing = starting_wellbeing
        
        for day in range(1, n_days + 1):
            # Update wellbeing based on trajectory
            if trajectory_type == 'growth':
                # Gradual improvement with noise
                current_wellbeing += (0.8 - current_wellbeing) * 0.1 + np.random.normal(0, 0.05)
                growth_traj = 'improving'
            elif trajectory_type == 'decline':
                # Gradual decline
                current_wellbeing += (0.2 - current_wellbeing) * 0.1 + np.random.normal(0, 0.05)
                growth_traj = 'declining'
            elif trajectory_type == 'recovery':
                # U-shaped: decline then improve
                midpoint = n_days // 2
                if day < midpoint:
                    current_wellbeing += (0.3 - current_wellbeing) * 0.15
                    growth_traj = 'declining'
                else:
                    current_wellbeing += (0.8 - current_wellbeing) * 0.15
                    growth_traj = 'improving'
            else:  # stable
                current_wellbeing += np.random.normal(0, 0.03)
                growth_traj = 'stable'
            
            # Clip to valid range
            current_wellbeing = np.clip(current_wellbeing, 0, 1)
            
            # Generate entry
            entry = self.generate_entry(current_wellbeing, growth_traj, day)
            
            entries.append(entry)
            wellbeing_levels.append(current_wellbeing)
        
        # Generate dates
        start_date = datetime.now() - timedelta(days=n_days)
        dates = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(n_days)]
        
        return {
            'entries': entries,
            'wellbeing_levels': wellbeing_levels,
            'trajectory_type': trajectory_type,
            'dates': dates,
            'final_wellbeing': current_wellbeing
        }
    
    def generate_dataset(
        self,
        n_users: int = 100,
        days_per_user: int = 30
    ) -> Dict[str, Any]:
        """
        Generate complete wellness dataset with multiple users.
        
        Returns dataset with longitudinal series for each user.
        """
        print(f"Generating wellness dataset: {n_users} users × {days_per_user} days...")
        
        all_entries = []
        all_wellbeing = []
        all_user_ids = []
        all_day_numbers = []
        trajectory_labels = []
        
        for user_id in range(n_users):
            # Random trajectory
            traj = random.choice(['growth', 'decline', 'recovery', 'stable'])
            start_well = np.random.beta(2, 2)
            
            series = self.generate_longitudinal_series(days_per_user, traj, start_well)
            
            all_entries.extend(series['entries'])
            all_wellbeing.extend(series['wellbeing_levels'])
            all_user_ids.extend([user_id] * days_per_user)
            all_day_numbers.extend(range(1, days_per_user + 1))
            trajectory_labels.extend([traj] * days_per_user)
        
        print(f"✓ Generated {len(all_entries)} total entries")
        
        # Binary outcome: final wellbeing > initial
        outcomes = []
        for user_id in range(n_users):
            user_entries_idx = [i for i, uid in enumerate(all_user_ids) if uid == user_id]
            initial = all_wellbeing[user_entries_idx[0]]
            final = all_wellbeing[user_entries_idx[-1]]
            improved = 1 if final > initial else 0
            outcomes.append(improved)
        
        print(f"  Users who improved: {sum(outcomes)} ({sum(outcomes)/len(outcomes)*100:.1f}%)")
        
        return {
            'entries': all_entries,
            'wellbeing_levels': all_wellbeing,
            'user_ids': all_user_ids,
            'day_numbers': all_day_numbers,
            'trajectory_labels': trajectory_labels,
            'user_outcomes': outcomes
        }


if __name__ == '__main__':
    print("Journal Entry Generator Demo\n")
    
    generator = JournalEntryGenerator()
    
    # Example series
    print("Growth Trajectory:")
    series = generator.generate_longitudinal_series(5, 'growth', 0.4)
    for i, entry in enumerate(series['entries']):
        print(f"  Day {i+1} ({series['wellbeing_levels'][i]:.2f}): {entry}")
    
    print("\n" + "="*80 + "\n")
    
    # Generate full dataset
    dataset = generator.generate_dataset(n_users=50, days_per_user=14)
    print("\n✅ Wellness dataset complete!")

