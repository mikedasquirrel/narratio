# Missing Narrative Data for MLB - Critical Gaps

**Key Insight from NFL**: Real individual names (coaches, QBs) created **1,403x improvement** (0.01% → 14% R²)

**Current MLB Data**: Team-level only (team names, stadiums, records)  
**Missing**: Individual-level narratives (pitchers, managers, players)

---

## Critical Missing Data (High Impact)

### 1. **Starting Pitcher Names** ⭐⭐⭐ (HIGHEST PRIORITY)

**Why Critical**: 
- Starting pitcher is the **most important individual** in baseball (like QB in NFL)
- Pitcher name carries prestige, reputation, narrative weight
- Pitcher matchups create storylines (Kershaw vs Scherzer, deGrom vs Cole)

**Narrative Value**:
- Pitcher prestige (Kershaw=0.98, Scherzer=0.95, deGrom=0.97)
- Pitcher vs team history (pitcher's record vs opponent)
- Pitcher streaks (consecutive quality starts, no-hitter attempts)
- Pitcher narratives (veteran ace, rising star, comeback story)

**MLB Stats API Available**:
- `gameData.matchup.pitchers.home.id` - Home pitcher ID
- `gameData.matchup.pitchers.away.id` - Away pitcher ID
- Can resolve IDs to names via `/people/{id}` endpoint

**Expected Impact**: Similar to NFL QB names (could unlock 10-20x improvement)

---

### 2. **Manager Names** ⭐⭐⭐ (HIGH PRIORITY)

**Why Critical**:
- Manager determines strategy, culture, lineup decisions
- Manager name carries prestige (like Belichick, Reid in NFL)
- Manager matchups matter (Maddon vs Bochy, Showalter vs Francona)

**Narrative Value**:
- Manager prestige (calculated from win rate, tenure, championships)
- Manager vs team history (manager's record vs opponent)
- Manager narratives (veteran strategist, young innovator, comeback story)

**MLB Stats API Available**:
- `gameData.managers.home.person.id` - Home manager ID
- `gameData.managers.away.person.id` - Away manager ID
- Can resolve IDs to names via `/people/{id}` endpoint

**Expected Impact**: Similar to NFL coach names (could unlock 5-10x improvement)

---

### 3. **Key Hitter Names** ⭐⭐ (MEDIUM-HIGH PRIORITY)

**Why Critical**:
- Star hitters create narratives (Trout, Judge, Ohtani, Acuña)
- Key hitters in lineup (cleanup hitter, leadoff hitter)
- Hitter vs pitcher history (career stats vs specific pitcher)

**Narrative Value**:
- Hitter prestige (Trout=0.98, Judge=0.95, Ohtani=0.97)
- Hitter streaks (hitting streaks, home run streaks)
- Hitter narratives (MVP candidate, rising star, veteran leader)

**MLB Stats API Available**:
- `gameData.lineups.home.batters` - Home lineup (IDs)
- `gameData.lineups.away.batters` - Away lineup (IDs)
- Can resolve IDs to names and get positions

**Expected Impact**: Moderate (less than pitcher, but still valuable)

---

## Important Missing Data (Medium Impact)

### 4. **Historical Context** ⭐⭐

**Missing**:
- Head-to-head records (team vs team, pitcher vs team, hitter vs pitcher)
- Playoff history (past playoff matchups, championship history)
- Recent matchup results (last 5-10 games between teams)
- Season series record (current season head-to-head)

**Narrative Value**:
- Rivalry intensity (Yankees-Red Sox playoff history)
- Revenge narratives (team seeking revenge for earlier loss)
- Dominance narratives (team has dominated recent matchups)

**MLB Stats API Available**:
- Historical game data via `/schedule` endpoint
- Can calculate head-to-head records from past games

---

### 5. **Streak Data** ⭐⭐

**Missing**:
- Team winning/losing streaks
- Pitcher quality start streaks
- Hitter hitting streaks
- Team home/away streaks

**Narrative Value**:
- Momentum narratives (team on hot streak)
- Pressure narratives (team trying to break losing streak)
- Clutch narratives (team performs well in streak situations)

**MLB Stats API Available**:
- Can calculate from game results
- Streak data may be in game notes or box scores

---

### 6. **Injury Narratives** ⭐

**Missing**:
- Key player injuries (star players out)
- Return from injury (player returning from IL)
- Injury impact (how team performs without key player)

**Narrative Value**:
- Adversity narratives (team overcoming injuries)
- Return narratives (player making comeback)
- Depth narratives (team's ability to overcome injuries)

**MLB Stats API Available**:
- Injury data via `/transactions` endpoint
- Transaction type: "Injury" or "Activated from IL"

---

### 7. **Rookie/Veteran Narratives** ⭐

**Missing**:
- Rookie debuts (first MLB game)
- Career milestones (3,000 hits, 500 home runs, 300 wins)
- Farewell tours (veteran's final season)

**Narrative Value**:
- Debut narratives (excitement around rookie debut)
- Milestone narratives (chasing history)
- Legacy narratives (veteran's final games)

**MLB Stats API Available**:
- Player debut dates via `/people/{id}` endpoint
- Career stats via `/stats` endpoint
- Can identify milestones from career totals

---

### 8. **Weather/Time Context** ⭐

**Missing**:
- Day vs night games
- Weather conditions (temperature, wind, precipitation)
- Time of day (afternoon, evening, late night)

**Narrative Value**:
- Day game narratives (traditional baseball)
- Weather narratives (wind affecting home runs, cold affecting pitchers)
- Time narratives (West Coast late games, East Coast early games)

**MLB Stats API Available**:
- Game time via `gameData.datetime`
- Weather data may be in game notes or external APIs

---

### 9. **Attendance Data** ⭐

**Missing**:
- Attendance numbers
- Sellout status
- Crowd energy (derived from attendance)

**Narrative Value**:
- Fan support narratives (packed stadium = home advantage)
- Atmosphere narratives (electric crowd, quiet stadium)
- Market narratives (large market vs small market)

**MLB Stats API Available**:
- Attendance via `gameData.attendance` field

---

### 10. **Umpire Names** ⭐

**Missing**:
- Home plate umpire name
- Umpire crew chief
- Umpire strike zone tendencies

**Narrative Value**:
- Umpire narratives (pitcher-friendly vs hitter-friendly umpire)
- Consistency narratives (consistent strike zone)
- Reputation narratives (respected vs controversial umpire)

**MLB Stats API Available**:
- Umpire data via `gameData.officials` endpoint

---

## Lower Priority (But Still Valuable)

### 11. **Trade Narratives**
- Recent trades (player facing former team)
- Trade deadline context (pre/post deadline)
- Player returns (player's first game back at former stadium)

### 12. **Contract Narratives**
- Free agency year (player in contract year)
- Extension narratives (player just signed extension)
- Salary narratives (high-paid player performance)

### 13. **Media Narratives**
- National TV games (ESPN, FOX, TBS)
- Local vs national broadcast
- Announcer teams (famous announcers)

### 14. **Comeback Narratives**
- Walk-off wins (dramatic finishes)
- Comeback wins (team overcoming deficit)
- Blown leads (team losing after leading)

---

## Implementation Priority

### Phase 1: Critical (Do First)
1. ✅ **Starting Pitcher Names** - Highest impact (like NFL QBs)
2. ✅ **Manager Names** - High impact (like NFL coaches)
3. ✅ **Key Hitter Names** - Medium-high impact (star players)

### Phase 2: Important (Do Second)
4. Historical context (head-to-head records)
5. Streak data (winning/losing streaks)
6. Injury narratives (key player injuries)

### Phase 3: Enhancement (Do Third)
7. Rookie/veteran narratives
8. Weather/time context
9. Attendance data
10. Umpire names

---

## Expected Impact

### Current State
- Correlation |r|: 0.0097 (after optimization)
- Efficiency Δ/π: 0.0034
- R²: -0.8% (negative, worse than baseline)

### With Pitcher + Manager Names (Phase 1)
**Expected**:
- Correlation |r|: 0.05-0.15 (5-15x improvement)
- Efficiency Δ/π: 0.017-0.053 (5-15x improvement)
- R²: 5-15% (realistic for team sport)

**Rationale**: 
- NFL showed 1,403x improvement with real names
- MLB pitcher = NFL QB (most important individual)
- MLB manager = NFL coach (strategy leader)

### With Full Phase 1-2 (Pitcher + Manager + Hitter + Context)
**Expected**:
- Correlation |r|: 0.10-0.25 (10-25x improvement)
- Efficiency Δ/π: 0.035-0.088 (10-25x improvement)
- R²: 10-25% (competitive with NFL's 14%)

---

## Data Collection Strategy

### MLB Stats API Endpoints Needed

1. **Game Details** (`/game/{gamePk}`):
   - Starting pitchers (home/away)
   - Managers (home/away)
   - Lineups (key hitters)
   - Umpires
   - Attendance
   - Weather

2. **People** (`/people/{id}`):
   - Resolve pitcher/manager/hitter IDs to names
   - Get player stats, career milestones

3. **Schedule** (`/schedule`):
   - Historical matchups (head-to-head records)
   - Streak calculations

4. **Transactions** (`/transactions`):
   - Injury data
   - Trade data
   - Roster moves

---

## Narrative Enhancement Examples

### Current Narrative (Team-Level Only)
```
"The legendary Yankees-Red Sox rivalry continues as Boston visits New York 
for this classic matchup. The Yankees enter with a 45-30 record, while the 
Red Sox stand at 40-35. Both teams are in the thick of the playoff race."
```

### Enhanced Narrative (With Individual Names)
```
"The legendary Yankees-Red Sox rivalry continues as Boston visits New York 
for this classic matchup. Yankees ace Gerrit Cole takes the mound against 
Red Sox veteran Chris Sale in a marquee pitching duel. Manager Aaron Boone's 
Yankees, led by slugger Aaron Judge, enter with a 45-30 record, while manager 
Alex Cora's Red Sox, featuring Rafael Devers, stand at 40-35. Both teams are 
in the thick of the playoff race, with Cole seeking his 10th win of the season 
and Sale making his return from injury."
```

**Difference**: Individual names (Cole, Sale, Judge, Devers, Boone, Cora) create rich nominative content for transformers to extract.

---

## Conclusion

**Critical Gap**: MLB currently collects **team-level data only**, missing the **individual-level narratives** that made NFL analysis successful.

**Solution**: Collect starting pitcher names, manager names, and key hitter names (Phase 1) to unlock nominative signal similar to NFL's 1,403x improvement.

**Expected Result**: 5-25x improvement in correlation and efficiency, bringing MLB from current weak state (|r|=0.0097) to competitive with NFL (|r|=0.10-0.25).

