"""
Fetch MLB game data from StatsAPI (official MLB API - free and no authentication needed).
"""
import requests
import pandas as pd
import time
from pathlib import Path
from datetime import datetime


def fetch_season_games_statsapi(year: int, verbose: bool = True) -> pd.DataFrame:
    """
    Fetch game schedule and results from MLB StatsAPI.
    
    Args:
        year: MLB season year (e.g., 2023)
        verbose: Print progress
        
    Returns:
        DataFrame with columns: date, home_team, away_team, home_score, away_score
    """
    if verbose:
        print(f"Fetching {year} games from MLB StatsAPI...")
    
    games = []
    
    try:
        # Fetch season schedule - regular season games
        url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&season={year}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        schedule = data if isinstance(data, list) else data.get('dates', [])
        
        if not schedule:
            print(f"  No data returned for {year}")
            return None
        
        for game_day in (schedule if isinstance(schedule, list) and isinstance(schedule[0], dict) and 'games' in schedule[0] else [{'games': schedule}]):
            games_list = game_day.get('games', []) if isinstance(game_day, dict) else [game_day]
            
            for game in games_list:
                if not isinstance(game, dict):
                    continue
                
                try:
                    # Skip games that haven't been played
                    status = game.get('status', {})
                    if isinstance(status, dict):
                        if status.get('abstractGameState') != 'Final':
                            continue
                    else:
                        continue
                    
                    # Extract game information
                    game_date = game.get('gameDateTime', '').split('T')[0]  # YYYY-MM-DD format
                    
                    away_team = game.get('teams', {}).get('away', {}).get('team', {}).get('teamCode', '').upper()
                    home_team = game.get('teams', {}).get('home', {}).get('team', {}).get('teamCode', '').upper()
                    
                    away_score = game.get('teams', {}).get('away', {}).get('score')
                    home_score = game.get('teams', {}).get('home', {}).get('score')
                    
                    # Validate data
                    if not all([game_date, away_team, home_team, away_score is not None, home_score is not None]):
                        continue
                    
                    games.append({
                        'date': game_date,
                        'away_team': away_team,
                        'away_score': int(away_score),
                        'home_team': home_team,
                        'home_score': int(home_score),
                    })
                except (KeyError, ValueError, TypeError):
                    continue
        
        if games:
            df = pd.DataFrame(games)
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date'])
            df['date'] = df['date'].dt.strftime('%Y-%m-%d')
            df = df.sort_values('date').reset_index(drop=True)
            
            if verbose:
                print(f"  ✓ Fetched {len(df)} games for {year}")
            
            return df
        else:
            print(f"  ✗ No completed games found for {year}")
            return None
            
    except requests.RequestException as e:
        print(f"  Error: {type(e).__name__}: {str(e)[:100]}")
        return None
    finally:
        time.sleep(1)  # Be respectful


def fetch_multiple_seasons(years: list, output_dir: str = "./data/raw", verbose: bool = True) -> dict:
    """
    Fetch multiple seasons of data.
    
    Args:
        years: List of years to fetch (e.g., [2021, 2022, 2023])
        output_dir: Directory to save CSV files
        verbose: Print progress
        
    Returns:
        Dict with year: dataframe pairs
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for year in years:
        df = fetch_season_games_statsapi(year, verbose=verbose)
        
        if df is not None and len(df) > 0:
            filepath = f"{output_dir}/games_{year}.csv"
            df.to_csv(filepath, index=False)
            results[year] = df
            if verbose:
                print(f"  Saved to {filepath}\n")
        else:
            if verbose:
                print(f"  Skipped {year}\n")
    
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("MLB STATSAPI FETCHER")
    print("=" * 60)
    
    # Fetch last 5 seasons
    years_to_fetch = [2020, 2021, 2022, 2023, 2024]
    
    print(f"\nFetching seasons: {years_to_fetch}")
    print("(Using official MLB StatsAPI - may take 1-2 minutes)\n")
    
    results = fetch_multiple_seasons(years_to_fetch)
    
    print("=" * 60)
    if results:
        total_games = sum(len(df) for df in results.values())
        print(f"✓ Success! Fetched {len(results)} seasons ({total_games} games total)")
        print(f"  Data saved to ./data/raw/")
        print(f"\n  Next: run 'python train.py'")
    else:
        print("✗ Could not fetch data from StatsAPI")
        print("  This could mean:")
        print("  - All seasons have no completed games yet")
        print("  - Network connectivity issue")
        print("\n  Alternative: Use the sample data generator")
        print("  python generate_sample_data.py")
    print("=" * 60)
