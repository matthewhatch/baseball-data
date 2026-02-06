#!/usr/bin/env python3
"""
Main entry point for MLB Game Outcome Prediction System
Provides command-line interface to all major functions.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def main():
    """Show available commands"""
    print("""
╔════════════════════════════════════════════════════════════════════╗
║         MLB GAME OUTCOME PREDICTION SYSTEM                        ║
║                                                                    ║
║  Available Commands:                                               ║
║                                                                    ║
║  1. SCRAPING:                                                      ║
║     python -m src.scraper                                          ║
║     → Fetch and save MLB game data from statsapi.mlb.com           ║
║                                                                    ║
║  2. EXPLORATION:                                                   ║
║     python -m src.eda                                              ║
║     → Analyze data distributions and patterns                      ║
║                                                                    ║
║  3. MODEL TRAINING:                                                ║
║     python -m src.train_model                                      ║
║     → Train game outcome prediction models                         ║
║                                                                    ║
║  4. PREDICTIONS:                                                   ║
║     python -m src.predict                                          ║
║     → Make game outcome predictions                                ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
    """)

if __name__ == '__main__':
    main()
