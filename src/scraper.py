"""
Scrape MLB game data using the MLB Stats API (statsapi.mlb.com).
Free, no authentication required, reliable JSON response.
"""
import pandas as pd
import time
from pathlib import Path
import requests
from datetime import datetime, timedelta
from tqdm import tqdm


def scrape_mlb_statsapi_season(year: int, team_abbr_map: dict, verbose: bool = True) -> pd.DataFrame:
    """
    Scrape game schedule and results using MLB Stats API.
    
    Args:
        year: MLB season year (e.g., 2023)
        team_abbr_map: Dictionary mapping team ID to abbreviation
        verbose: Print progress
        
    Returns:
        DataFrame with columns: date, away_team, away_score, home_team, home_score
    """
    if verbose:
        print(f"Fetching {year} schedule from MLB Stats API...")
    
    games = []
    
    # Get season schedule from MLB Stats API
    # API returns games in batches
    start_date = datetime(year, 3, 20)
    end_date = datetime(year, 11, 1)
    
    current_date = start_date
    processed = 0
    
    # Calculate total days for progress bar
    total_days = (end_date - start_date).days + 1
    
    with tqdm(total=total_days, desc=f"  {year}", unit="day", disable=not verbose) as pbar:
        while current_date <= end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            
            # MLB Stats API endpoint for schedule
            url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={date_str}&endDate={date_str}"
            
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                
                # Parse games from response
                if 'dates' in data and data['dates']:
                    for date_entry in data['dates']:
                        if 'games' in date_entry:
                            for game in date_entry['games']:
                                # Extract game information
                                try:
                                    # Only include completed games (Final status)
                                    status = game.get('status', {}).get('abstractGameState', '')
                                    
                                    if status != 'Final':
                                        continue
                                    
                                    away_team_id = game.get('teams', {}).get('away', {}).get('team', {}).get('id')
                                    home_team_id = game.get('teams', {}).get('home', {}).get('team', {}).get('id')
                                    away_score = game.get('teams', {}).get('away', {}).get('score')
                                    home_score = game.get('teams', {}).get('home', {}).get('score')
                                    
                                    # Get abbreviations from map
                                    away_abbr = team_abbr_map.get(away_team_id, 'UNK')
                                    home_abbr = team_abbr_map.get(home_team_id, 'UNK')
                                    
                                    # Get game ID and additional info
                                    game_id = game.get('gamePk')
                                    game_type = game.get('gameType', '')
                                    venue = game.get('venue', {}).get('name', '')
                                    game_status = game.get('status', {}).get('abstractGameState', '')
                                    double_header = game.get('doubleHeader', 'N')
                                    
                                    # Only include games with valid scores
                                    if away_score is not None and home_score is not None:
                                        games.append({
                                            'game_id': game_id,
                                            'game_type': game_type,
                                            'date': date_str,
                                            'away_team': away_abbr,
                                            'away_score': int(away_score),
                                            'home_team': home_abbr,
                                            'home_score': int(home_score),
                                            'venue': venue,
                                            'status': game_status,
                                            'double_header': double_header
                                        })
                                        processed += 1
                                except (KeyError, TypeError, ValueError):
                                    # Skip malformed game entries
                                    continue
                    
            except requests.exceptions.RequestException as e:
                if verbose:
                    pbar.write(f"    ⚠ Error fetching {date_str}: {str(e)[:40]}")
                # Continue to next date
            except Exception as e:
                if verbose:
                    pbar.write(f"    ⚠ Parse error on {date_str}: {str(e)[:40]}")
            
            current_date += timedelta(days=1)
            pbar.update(1)
            pbar.set_postfix({'games': processed})
    
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
        return pd.DataFrame(columns=['game_id', 'game_type', 'date', 'away_team', 'away_score', 'home_team', 'home_score', 'venue', 'status', 'double_header'])


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
    
    # Load team abbreviations once
    if verbose:
        print("Loading team information...")
    try:
        teams_resp = requests.get('https://statsapi.mlb.com/api/v1/teams?sportId=1', timeout=10)
        teams_data = teams_resp.json()
        team_abbr_map = {}
        for team in teams_data.get('teams', []):
            team_abbr_map[team['id']] = team.get('abbreviation', team['name'][:3].upper())
        if verbose:
            print(f"✓ Loaded {len(team_abbr_map)} teams\n")
    except Exception as e:
        if verbose:
            print(f"✗ Failed to load teams: {str(e)}\n")
        team_abbr_map = {}
    
    results = {}
    
    for year in years:
        df = scrape_mlb_statsapi_season(year, team_abbr_map, verbose=verbose)
        
        if df is not None and len(df) > 0:
            csv_path = output_path / f"games_{year}.csv"
            df.to_csv(csv_path, index=False)
            results[year] = df
            
            if verbose:
                print(f"  Saved {csv_path}")
        else:
            if verbose:
                print(f"  ✗ No data for {year}")
            results[year] = pd.DataFrame()
        
        # Be respectful to the API
        time.sleep(1)
    
    return results


if __name__ == "__main__":
    # Scrape 2020-2024 seasons
    years_to_scrape = [2020, 2021, 2022, 2023, 2024]
    
    print("=" * 60)
    print("MLB Game Data Scraper")
    print("Source: MLB Stats API (statsapi.mlb.com)")
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
        print(f"{status} {year}: {count} games")
    
    print(f"\nTotal games scraped: {total_games}")
    print()
