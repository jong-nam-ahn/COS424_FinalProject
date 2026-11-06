# COS424_FinalProject

## Critics vs. The Public: a tale of two ratings

Jongnam Ahn, Sawyer Dilks, Kevin Weng

## Repository structure

```
main_analysis.ipynb        Milestone 2 jupyter notebook
data/
  pitchfork.csv            Pitchfork reviews with Spotify audio features (input)
  discogs.csv              Discogs ratings for each Pitchfork album (scraped)
  pitchfork_discogs.csv    merged and filtered dataset used by the notebook
src/
  eda.py                   loading, summary tables and plots called by the notebook
  regression.py            baseline score-difference regression
  scraping/
    collect_discogs.py     looks up each album on the Discogs API, saves data/discogs.csv
    build_dataset.py       joins and filters the data into data/pitchfork_discogs.csv
    text_norm.py           album name matching
```

All commands below are run from the repository root.

## Quick start

### 0. Install dependencies

```bash
pip install -r requirements.txt
```

### 1. Scrape Discogs and build the dataset

The scraped and merged data are already in `data/`, so this step is only needed to reproduce them. The scraper skips albums already in its output file, so delete `data/discogs.csv` first to scrape everything from scratch.

```bash
# 1a. Set a Discogs API token (https://www.discogs.com/settings/developers)
export DISCOGS_TOKEN=your_token_here

# 1b. Scrape Discogs ratings for the Pitchfork albums into data/discogs.csv
python src/scraping/collect_discogs.py --limit 20 --output data/discogs.csv   # quick test
python src/scraping/collect_discogs.py --hours 4  --output data/discogs.csv   # stops cleanly after 4 hours

# 1c. Join with Pitchfork and save data/pitchfork_discogs.csv
python src/scraping/build_dataset.py
```

### 2. Run the EDA

```bash
# 2a. Open the notebook and run all cells
jupyter notebook main_analysis.ipynb

```
