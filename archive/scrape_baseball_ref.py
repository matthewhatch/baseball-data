"""
Scrape MLB game data using the MLB Stats API (statsapi.mlb.com).
Free, no authentication required.
"""
import pandas as pd
import time
from pathlib import Path
import requests
from datetime import datetime, timedelta


def scrape_mlb_statsapi_season(year: int, verbose: bool = True) -> pd.DataFrame:
    """
    Scrape game schedule and results using MLB Stats API.
    
    Args:
        year: MLB season year (e.g., 2023)
        verbose: Print progress
        
    Returns:
        DataFrame with columns: date, away_team, away_score, home_team, home_score
    """
    if verbose:
        print(f"Fetching {year} schedule from MLB Stats API...")
    
    games = []
    
    # Get season schedule from MLB Stats API
    # API returns games in batches, iterate through dates
    start_date = datetime(year, 3, 20)
    end_date = datetime(year, 11, 1)
    
    current_date = start_date
    
    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        
        # MLB Stats API endpoint for schedule
        url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={date_str}&endDate={date_str}"
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Parse games from response
                if verbose:
                    print(f"  Fetching {date_str}...", end=" ")
                
                # Look for game containers/rows on the page
                # Yahoo Sports game structure varies, try multiple selectors
                game_elements = soup.find_all(['div', 'tr'], class_=re.compile(r'game|Game|score|Score', re.I))
                
                # Try alternative: look for elements with team abbreviations and scores
                # Example pattern: "NYY 5 BOS 3"
                text_content = soup.get_text()
                
                # Extract games from text (simplified pattern matching)
                # This is a placeholder - we'll need to refine based on actual HTML structure
                
                if game_elements:
                    if verbose:
                        print(f"found {len(game_elements)} elements")
                else:
                    if verbose:
                        print("no games")
                
            except Exception as e:
                if verbose:
                    print(f"error: {str(e)[:30]}")
                continue
            finally:
                current_date += timedelta(days=1)
        
        if games:
            result_df = pd.DataFrame(games)
            result_df['date'] = pd.to_datetime(result_df['date'], errors='coerce')
            result_df = result_df.dropna(subset=['date'])
            result_df['date'] = result_df['date'].dt.strftime('%Y-%m-%d')
            
            if verbose:
                print(f"  ✓ Scraped {len(result_df)} games for {year}")
            
            return result_df
        else:
            if verbose:
                print(f"  ✗ No games parsed for {year} from Yahoo Sports")
            return None
            
    except Exception as e:
        if verbose:
            print(f"  Error: {type(e).__name__}: {str(e)[:100]}")
        return None
    finally:
        time.sleep(2)


def scrape_multiple_seasons(years: list, output_dir: str = "./data/raw", verbose: bool = True) -> dict:
    """
    Scrape multiple seasons of data from Yahoo Sports.
    
    Args:
        years: List of years to scrape (e.g., [2021, 2022, 2023])
        output_dir: Directory to save CSV files
        verbose: Print progress
        
    Returns:
        Dict with year: dataframe pairs
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for year in years:
        df = scrape_yahoo_sports_season(year, verbose=verbose)
        
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
    import sys
    
    print("=" * 60)
    print("MLB GAME SCRAPER - Yahoo Sports")
    print("=" * 60)
    
    # Scrape last 5 seasons
    years_to_scrape = [2020, 2021, 2022, 2023, 2024]
    
    print(f"\nScraping seasons: {years_to_scrape}")
    print("(This may take a minute...)\n")
    results = scrape_multiple_seasons(years_to_scrape)
    
    print("=" * 60)
    if results:
        print(f"✓ Complete! Scraped {len(results)} seasons")
        print(f"  Data saved to ./data/raw/")
        print(f"\n  Next: run 'python train.py'")
    else:
        print("✗ No data scraped. Try:")
        print("  1. Check internet connection")
        print("  2. Verify Yahoo Sports is accessible")
        print("  3. Download CSV from Kaggle manually")
        print("  https://www.kaggle.com/search?q=MLB+games")
    print("=" * 60)
