"""
Quick guide: Download real baseball data and use it with the model.

OPTION 1: Download from Kaggle (Easiest - 2 minutes)
======================================================
1. Go to: https://www.kaggle.com/datasets
2. Search: "MLB games" or "baseball games"
3. Popular datasets:
   - "MLB Scores Data" by daverosenman
   - "Baseball Dataset" by open-source
   - "MLB Games Data" by search results
4. Download the CSV file
5. Save as: ./data/raw/games_YYYY.csv
   (where YYYY is the season year)
6. Run: python train.py

Required columns in CSV:
- date (format: YYYY-MM-DD or similar)
- home_team (team abbreviation: NYY, BOS, LAD, etc.)
- away_team (team abbreviation)
- home_score (final score)
- away_score (final score)

OPTION 2: Use already scraped data from GitHub
===============================================
1. Download from: https://github.com/chadwickbureau/retrosheet
2. Or: https://github.com/jldbc/pybaseball (has example data)
3. Convert to CSV format with required columns
4. Save to ./data/raw/games_YYYY.csv
5. Run: python train.py

OPTION 3: Use sample data (for testing)
========================================
1. Run: python generate_sample_data.py
   (Creates realistic synthetic data)
2. Then: python train.py
   
OPTION 4: APIs (More complex but free)
=======================================
a) Baseball Reference (via requests + BeautifulSoup):
   - See: scrape_baseball_ref.py
   - May have rate limiting/blocking
   
b) MLB StatsAPI (best free API):
   - Requires game day tracking
   - See: fetch_statsapi.py

CSV FILE FORMAT EXAMPLE:
=======================
date,home_team,away_team,home_score,away_score
2024-03-28,NYY,BOS,3,2
2024-03-29,NYY,BOS,5,3
2024-03-30,BOS,NYY,2,6
2024-03-31,TB,LAD,4,1

STEPS TO GET STARTED:
======================
1. [ ] Download CSV from Kaggle (or option above)
2. [ ] Check it has required columns
3. [ ] Rename/save to ./data/raw/games_YYYY.csv
4. [ ] Repeat for multiple years if available
5. [ ] Run: python train.py

That's it! The model will:
- Load all CSV files from ./data/raw/
- Engineer features (team stats, rolling averages)
- Split into train/validation/test
- Train XGBoost model
- Evaluate and save model
"""

if __name__ == "__main__":
    print(__doc__)
    print("\nFiles in data/raw/:")
    import os
    from pathlib import Path
    
    raw_dir = Path("./data/raw")
    if raw_dir.exists():
        files = list(raw_dir.glob("*.csv"))
        if files:
            for f in files:
                size = f.stat().st_size / 1024  # KB
                lines = len(open(f).readlines())
                print(f"  ✓ {f.name} ({lines} lines, {size:.1f} KB)")
        else:
            print("  (empty - add CSV files here)")
    else:
        print("  (directory not found)")
    
    print("\nQuick commands:")
    print("  python generate_sample_data.py   # Create test data")
    print("  python train.py                  # Train model")
    print("  python scrape_baseball_ref.py    # Scrape data (may fail)")
