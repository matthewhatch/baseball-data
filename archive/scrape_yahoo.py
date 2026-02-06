"""
Scrape MLB game data from Yahoo Sports using BeautifulSoup.
Extracts JSON data embedded in the HTML page.
"""
import pandas as pd
import time
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import json
import re
from datetime import datetime, timedelta


def scrape_yahoo_sports_season(year: int, verbose: bool = True) -> pd.DataFrame:
    """
    Scrape game schedule and results from Yahoo Sports.
    Extracts game data from embedded JSON in the HTML.
    
    Args:
        year: MLB season year (e.g., 2023)
        verbose: Print progress
        
    Returns:
        DataFrame with columns: date, away_team, away_score, home_team, home_score
    """
    if verbose:
        print(f"Fetching {year} schedule from Yahoo Sports...")
    
    games = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    # MLB season typically runs from late March through November
    start_date = datetime(year, 3, 20)
    end_date = datetime(year, 11, 1)
    
    current_date = start_date
    dates_processed = 0
    
    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        url = f"https://sports.yahoo.com/mlb/schedule/?season={year}&date={date_str}"
        
        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all script tags and look for game data
            scripts = soup.find_all('script')
            
            for script in scripts:
                if script.string:
                    script_text = script.string
                    
                    # Look for game data patterns in the script
                    # Yahoo embeds data as JSON objects
                    if '"games"' in script_text or 'gameCards' in script_text:
                        try:
                            # Try to extract JSON from script content
                            # Look for patterns like: {"games": [...]}
                            json_matches = re.findall(r'\{[^{}]*"games"[^{}]*\}', script_text)
                            
                            for json_str in json_matches:
                                try:
                                    data = json.loads(json_str)
                                    if 'games' in data:
                                        for game in data['games']:
                                            parsed_game = parse_yahoo_game(game, date_str)
                                            if parsed_game:
                                                games.append(parsed_game)
                                except json.JSONDecodeError:
                                    continue
                                    
                        except Exception as e:
                            continue
            
            # Also try to extract from data attributes
            # Look for game containers with team/score data
            game_divs = soup.find_all('div', class_=re.compile(r'game|Game|schedule|Schedule', re.I))
            
            for game_div in game_divs:
                # Try to extract team names and scores from the div
                text = game_div.get_text(strip=True)
                
                # Pattern: "TeamA vs TeamB" with scores
                # Try to find score patterns like "3-2" or similar
                score_match = re.search(r'(\d+)\s*[-:]\s*(\d+)', text)
                if score_match:
                    # Extract team names from nearby elements
                    teams = re.findall(r'([A-Z]{2,3})', text)
                    if len(teams) >= 2:
                        # This is a very basic extraction, may need refinement
                        pass
            
            if verbose:
                dates_processed += 1
                if dates_processed % 30 == 0:
                    print(f"  Processed {dates_processed} dates, found {len(games)} games so far...")
            
        except requests.exceptions.RequestException as e:
            if verbose:
                print(f"  Warning: Network error on {date_str}: {str(e)[:40]}")
        except Exception as e:
            if verbose:
                print(f"  Warning: Parse error on {date_str}: {str(e)[:40]}")
        
        current_date += timedelta(days=1)
        # Rate limit to be respectful
        time.sleep(0.5)
    
    if games:
        result_df = pd.DataFrame(games)
        result_df['date'] = pd.to_datetime(result_df['date'], errors='coerce')
        result_df = result_df.dropna(subset=['date'])
        result_df['date'] = result_df['date'].dt.strftime('%Y-%m-%d')
        result_df = result_df.sort_values('date').reset_index(drop=True)
        
        if verbose:
            print(f"  ✓ Scraped {len(result_df)} games for {year}")
        
        return result_df
    else:
        if verbose:
            print(f"  ✗ No games parsed for {year}")
        return pd.DataFrame(columns=['date', 'away_team', 'away_score', 'home_team', 'home_score'])


def parse_yahoo_game(game_data, date_str):
    """
    Parse a single game from Yahoo Sports game object.
    
    Args:
        game_data: Game object from Yahoo API/HTML
        date_str: Date string for the game
        
    Returns:
        Dictionary with game info or None if parsing fails
    """
    try:
        # Try different possible structures for game data
        if isinstance(game_data, dict):
            # Standard structure
            if 'awayTeam' in game_data and 'homeTeam' in game_data:
                away_team = game_data['awayTeam'].get('abbreviation', '')
                home_team = game_data['homeTeam'].get('abbreviation', '')
                away_score = game_data['awayTeam'].get('score')
                home_score = game_data['homeTeam'].get('score')
                
                if away_score is not None and home_score is not None:
                    return {
                        'date': date_str,
                        'away_team': away_team,
                        'away_score': int(away_score),
                        'home_team': home_team,
                        'home_score': int(home_score)
                    }
            
            # Alternative structure
            if 'away' in game_data and 'home' in game_data:
                away_data = game_data['away']
                home_data = game_data['home']
                
                away_team = away_data.get('abbreviation', away_data.get('name', ''))
                home_team = home_data.get('abbreviation', home_data.get('name', ''))
                away_score = away_data.get('score')
                home_score = home_data.get('score')
                
                if away_score is not None and home_score is not None:
                    return {
                        'date': date_str,
                        'away_team': away_team,
                        'away_score': int(away_score),
                        'home_team': home_team,
                        'home_score': int(home_score)
                    }
    
    except (KeyError, TypeError, ValueError):
        pass
    
    return None


def scrape_multiple_seasons(years: list, output_dir: str = "./data/raw", verbose: bool = True) -> dict:
    """
    Scrape multiple seasons and save to CSV files.
    
    Args:
        years: List of years to scrape (e.g., [2020, 2021, 2022])
        output_dir: Directory to save CSV files
        verbose: Print progress
        
    Returns:
        Dictionary mapping year to DataFrame
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for year in years:
        df = scrape_yahoo_sports_season(year, verbose=verbose)
        
        if df is not None and len(df) > 0:
            csv_path = output_path / f"games_{year}.csv"
            df.to_csv(csv_path, index=False)
            results[year] = df
            
            if verbose:
                print(f"  Saved {csv_path}\n")
        else:
            if verbose:
                print(f"  ✗ No data for {year}\n")
            results[year] = pd.DataFrame()
        
        # Be respectful to Yahoo Sports
        time.sleep(2)
    
    return results


if __name__ == "__main__":
    # Scrape 2020-2024 seasons
    years_to_scrape = [2020, 2021, 2022, 2023, 2024]
    
    print("=" * 60)
    print("MLB Game Data Scraper")
    print("Source: Yahoo Sports (sports.yahoo.com/mlb)")
    print("=" * 60)
    print()
    
    results = scrape_multiple_seasons(
        years=years_to_scrape,
        output_dir="./data/raw",
        verbose=True
    )
    
    print()
    print("=" * 60)
    print("Summary:")
    print("=" * 60)
    
    total_games = 0
    for year, df in results.items():
        count = len(df)
        total_games += count
        status = "✓" if count > 0 else "✗"
        print(f"{status} {year}: {count:,} games")
    
    print(f"\nTotal games scraped: {total_games:,}")
    print()
