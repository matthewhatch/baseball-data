#!/usr/bin/env python3
"""
Quick reference for common tasks
"""

COMMANDS = {
    "Fetch Data": "python -m src.scraper",
    "Analyze Data": "python -m src.eda",
    "Train Models": "python -m src.train_model",
    "Make Predictions": "python -m src.predict",
}

FILES = {
    "Main Entry": "main.py",
    "Data Scraper": "src/scraper.py",
    "Data Analysis": "src/eda.py",
    "Model Training": "src/train_model.py",
    "Predictions": "src/predict.py",
}

DOCS = {
    "Quick Start": "README.md",
    "Project Structure": "docs/PROJECT_STRUCTURE.md",
    "Model Summary": "docs/OUTCOME_MODEL_SUMMARY.md",
    "Scraper Details": "docs/SCRAPER_SUMMARY.md",
    "Data Guide": "docs/DATA_GUIDE.md",
}

if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════════════╗
║         MLB GAME OUTCOME PREDICTION - QUICK REFERENCE             ║
╚════════════════════════════════════════════════════════════════════╝

COMMANDS:
""")
    for name, cmd in COMMANDS.items():
        print(f"  {name:20s} → {cmd}")

    print("""
FILES:
""")
    for name, path in FILES.items():
        print(f"  {name:20s} → {path}")

    print("""
DOCUMENTATION:
""")
    for name, path in DOCS.items():
        print(f"  {name:20s} → {path}")

    print("""
DATA:
  Raw Game Data           → data/raw/games_*.csv (11,486 games)
  
MODELS:
  Trained Models          → models/*.pkl
  Performance Chart       → models/model_performance.png
  
ARCHIVED CODE:
  Deprecated Scrapers     → archive/scrape_*.py
  Legacy Scripts          → archive/*.py
  Logs & Temp Files       → archive/

╔════════════════════════════════════════════════════════════════════╗
║ NEW USERS: Start with README.md                                    ║
║ DEVELOPERS: See docs/PROJECT_STRUCTURE.md for detailed org         ║
╚════════════════════════════════════════════════════════════════════╝
""")
