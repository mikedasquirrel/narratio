#!/usr/bin/env python3
"""
Today's Games Scraper - All Sports
Scrapes ESPN schedules for NHL, NFL, NBA

No API keys required - just web scraping
Date: November 17, 2025
"""

import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import json
from typing import List, Dict, Optional
import re

def scrape_nhl_games(date: Optional[str] = None) -> List[Dict]:
    """
    Scrape NHL games from ESPN
    
    Parameters
    ----------
    date : str, optional
        Date in YYYYMMDD format, defaults to today
    
    Returns
    -------
    games : list of dict
        List of NHL games
    """
    if date is None:
        date = datetime.now().strftime('%Y%m%d')
    
    url = f"https://www.espn.com/nhl/schedule/_/date/{date}"
    
    print(f"  Scraping NHL: {url}")
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        games = []
        
        # Find all game rows
        game_rows = soup.find_all('tr', class_=re.compile('Table__TR'))
        
        for row in game_rows:
            try:
                # Extract team names
                teams = row.find_all('span', class_='Table__Team')
                if len(teams) < 2:
                    continue
                
                away_team = teams[0].text.strip()
                home_team = teams[1].text.strip()
                
                # Extract time
                time_elem = row.find('td', class_='date__col')
                game_time = time_elem.text.strip() if time_elem else 'TBD'
                
                game = {
                    'sport': 'NHL',
                    'away_team': away_team,
                    'home_team': home_team,
                    'date': datetime.strptime(date, '%Y%m%d').strftime('%Y-%m-%d'),
                    'time': game_time,
                    'source': 'ESPN'
                }
                
                games.append(game)
                
            except Exception as e:
                continue
        
        print(f"    Found {len(games)} NHL games")
        return games
        
    except Exception as e:
        print(f"    Error scraping NHL: {e}")
        return []

def scrape_nfl_games(date: Optional[str] = None) -> List[Dict]:
    """
    Scrape NFL games from ESPN
    
    Parameters
    ----------
    date : str, optional
        Date in YYYYMMDD format, defaults to today
    
    Returns
    -------
    games : list of dict
        List of NFL games
    """
    if date is None:
        date = datetime.now().strftime('%Y%m%d')
    
    # NFL typically uses week-based schedule, but try date-based first
    url = f"https://www.espn.com/nfl/schedule/_/date/{date}"
    
    print(f"  Scraping NFL: {url}")
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        games = []
        
        # Find all game rows
        game_rows = soup.find_all('tr', class_=re.compile('Table__TR'))
        
        for row in game_rows:
            try:
                # Extract team names
                teams = row.find_all('span', class_='Table__Team')
                if len(teams) < 2:
                    continue
                
                away_team = teams[0].text.strip()
                home_team = teams[1].text.strip()
                
                # Extract spread if available
                spread_elem = row.find('td', class_='line')
                spread = spread_elem.text.strip() if spread_elem else None
                
                # Extract time
                time_elem = row.find('td', class_='date__col')
                game_time = time_elem.text.strip() if time_elem else 'TBD'
                
                game = {
                    'sport': 'NFL',
                    'away_team': away_team,
                    'home_team': home_team,
                    'date': datetime.strptime(date, '%Y%m%d').strftime('%Y-%m-%d'),
                    'time': game_time,
                    'spread': spread,
                    'source': 'ESPN'
                }
                
                games.append(game)
                
            except Exception as e:
                continue
        
        print(f"    Found {len(games)} NFL games")
        return games
        
    except Exception as e:
        print(f"    Error scraping NFL: {e}")
        return []

def scrape_nba_games(date: Optional[str] = None) -> List[Dict]:
    """
    Scrape NBA games from ESPN
    
    Parameters
    ----------
    date : str, optional
        Date in YYYYMMDD format, defaults to today
    
    Returns
    -------
    games : list of dict
        List of NBA games
    """
    if date is None:
        date = datetime.now().strftime('%Y%m%d')
    
    url = f"https://www.espn.com/nba/schedule/_/date/{date}"
    
    print(f"  Scraping NBA: {url}")
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        games = []
        
        # Find all game rows
        game_rows = soup.find_all('tr', class_=re.compile('Table__TR'))
        
        for row in game_rows:
            try:
                # Extract team names
                teams = row.find_all('span', class_='Table__Team')
                if len(teams) < 2:
                    continue
                
                away_team = teams[0].text.strip()
                home_team = teams[1].text.strip()
                
                # Extract spread if available
                spread_elem = row.find('td', class_='line')
                spread = spread_elem.text.strip() if spread_elem else None
                
                # Extract time
                time_elem = row.find('td', class_='date__col')
                game_time = time_elem.text.strip() if time_elem else 'TBD'
                
                game = {
                    'sport': 'NBA',
                    'away_team': away_team,
                    'home_team': home_team,
                    'date': datetime.strptime(date, '%Y%m%d').strftime('%Y-%m-%d'),
                    'time': game_time,
                    'spread': spread,
                    'source': 'ESPN'
                }
                
                games.append(game)
                
            except Exception as e:
                continue
        
        print(f"    Found {len(games)} NBA games")
        return games
        
    except Exception as e:
        print(f"    Error scraping NBA: {e}")
        return []

def scrape_all_sports(date: Optional[str] = None) -> Dict:
    """
    Scrape all sports for given date
    
    Parameters
    ----------
    date : str, optional
        Date in YYYYMMDD format, defaults to today
    
    Returns
    -------
    all_games : dict
        Dictionary with games by sport
    """
    if date is None:
        date = datetime.now().strftime('%Y%m%d')
    
    date_display = datetime.strptime(date, '%Y%m%d').strftime('%Y-%m-%d')
    
    print(f"\n{'='*80}")
    print(f"  SCRAPING GAMES FOR {date_display}")
    print('='*80)
    
    nhl_games = scrape_nhl_games(date)
    nfl_games = scrape_nfl_games(date)
    nba_games = scrape_nba_games(date)
    
    all_games = {
        'date': date_display,
        'scraped_at': datetime.now().isoformat(),
        'nhl': nhl_games,
        'nfl': nfl_games,
        'nba': nba_games,
        'total_games': len(nhl_games) + len(nfl_games) + len(nba_games)
    }
    
    print(f"\n  SUMMARY:")
    print(f"    NHL: {len(nhl_games)} games")
    print(f"    NFL: {len(nfl_games)} games")
    print(f"    NBA: {len(nba_games)} games")
    print(f"    Total: {all_games['total_games']} games")
    print(f"\n{'='*80}\n")
    
    return all_games

def main():
    """Main scraper execution"""
    import sys
    
    # Get date from command line or use today
    if len(sys.argv) > 1:
        date = sys.argv[1]
    else:
        date = None
    
    # Scrape all games
    all_games = scrape_all_sports(date)
    
    # Save to file
    output_file = f"data/scraped_games/games_{all_games['date'].replace('-', '')}.json"
    from pathlib import Path
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(all_games, f, indent=2)
    
    print(f"Saved to: {output_file}\n")

if __name__ == '__main__':
    main()

