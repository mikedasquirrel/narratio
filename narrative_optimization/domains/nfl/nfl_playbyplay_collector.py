"""
NFL Play-by-Play Data Collection

Collect 500K+ plays (2014-2024) with:
- Play descriptions ("Mahomes to Kelce for 12 yards")
- Situational terms ("3rd and 7", "red zone")
- Player names (passer, receiver, rusher, tackler)
- Down, distance, field position, time, score

This is the TRUE nominative goldmine - not just roster, but actual play-level language.
"""

import nfl_data_py as nfl
import pandas as pd
import json
from pathlib import Path
from typing import List
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("NFL PLAY-BY-PLAY DATA COLLECTION")
print("="*80)
print("\nCollecting THE REAL NOMINATIVE GOLDMINE:")
print("  - Play descriptions (text narratives)")
print("  - QB-to-WR relational phrases")
print("  - Situational terminology (3rd-and-long, red zone)")
print("  - Down/distance/field position")
print("  - Time/score context")

# Collect play-by-play data
years = list(range(2014, 2025))

print(f"\n[1/3] Collecting play-by-play data for {len(years)} seasons...")
print("  (This may take 2-5 minutes - downloading ~500K plays)")

pbp_data = nfl.import_pbp_data(years, downcast=True, cache=False)

print(f"\n✓ Collected {len(pbp_data)} plays")
print(f"  Columns: {len(pbp_data.columns)}")
print(f"  Size: {pbp_data.memory_usage().sum() / 1024 / 1024:.1f} MB")

# Show sample
print("\n[2/3] Sample play description:")
sample_play = pbp_data[pbp_data['desc'].notna()].iloc[1000]
print(f"  {sample_play['desc']}")
print(f"  Down: {sample_play.get('down')}, Distance: {sample_play.get('ydstogo')}")
print(f"  Passer: {sample_play.get('passer_player_name')}")
print(f"  Receiver: {sample_play.get('receiver_player_name')}")

# Filter to meaningful plays (have descriptions)
print("\n[3/3] Filtering to plays with descriptions...")
pbp_filtered = pbp_data[pbp_data['desc'].notna()].copy()

print(f"✓ Filtered to {len(pbp_filtered)} plays with descriptions")
print(f"  {100*len(pbp_filtered)/len(pbp_data):.1f}% of plays have descriptions")

# Key columns for nominative analysis
key_cols = [
    'game_id', 'desc', 'down', 'ydstogo', 'yardline_100', 'qtr', 'game_seconds_remaining',
    'score_differential', 'posteam', 'defteam', 
    'passer_player_name', 'receiver_player_name', 'rusher_player_name',
    'complete_pass', 'yards_gained', 'touchdown', 'interception', 'fumble',
    'third_down_converted', 'fourth_down_converted',
    'play_type', 'pass_length', 'pass_location', 'run_location', 'run_gap'
]

# Keep only available columns
available_cols = [c for c in key_cols if c in pbp_filtered.columns]
pbp_slim = pbp_filtered[available_cols].copy()

print(f"✓ Extracted {len(available_cols)} key columns")

# Save to JSON (sample for speed)
output_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'nfl_playbyplay_sample.json'

# Take sample (10K plays for fast processing)
sample_size = 10000
pbp_sample = pbp_slim.sample(n=min(sample_size, len(pbp_slim)), random_state=42)

# Convert to records
records = pbp_sample.to_dict('records')

# Clean NaN values
for record in records:
    for key, value in record.items():
        if pd.isna(value):
            record[key] = None

print(f"\n✓ Saving {len(records)} sample plays...")

with open(output_path, 'w') as f:
    json.dump(records, f, indent=2)

print(f"✓ Saved to: {output_path}")

# Statistics
print("\n" + "="*80)
print("PLAY-BY-PLAY STATISTICS")
print("="*80)

print(f"\nTotal plays collected: {len(pbp_filtered)}")
print(f"Sample for analysis: {len(records)}")

# Situational breakdown
third_downs = pbp_filtered[pbp_filtered['down'] == 3]
red_zone = pbp_filtered[pbp_filtered['yardline_100'] <= 20]
two_minute = pbp_filtered[pbp_filtered['game_seconds_remaining'] <= 120]

print(f"\nSituational plays:")
print(f"  3rd down: {len(third_downs)} ({100*len(third_downs)/len(pbp_filtered):.1f}%)")
print(f"  Red zone: {len(red_zone)} ({100*len(red_zone)/len(pbp_filtered):.1f}%)")
print(f"  2-minute: {len(two_minute)} ({100*len(two_minute)/len(pbp_filtered):.1f}%)")

# Player involvement
passers = pbp_filtered['passer_player_name'].value_counts()
receivers = pbp_filtered['receiver_player_name'].value_counts()

print(f"\nPlayer involvement:")
print(f"  Unique QBs: {passers.count()}")
print(f"  Unique receivers: {receivers.count()}")
print(f"  Top QB: {passers.index[0]} ({passers.iloc[0]} plays)")
print(f"  Top receiver: {receivers.index[0]} ({receivers.iloc[0]} targets)")

print("\n" + "="*80)
print("PLAY-BY-PLAY COLLECTION COMPLETE")
print("="*80)
print(f"\nNext: Extract nominative features from play descriptions")
print(f"  - QB-to-WR connections (relational)")
print(f"  - Situational terms (3rd-and-long, red zone)")
print(f"  - Player usage patterns")
print(f"  - Down/distance/field position")

