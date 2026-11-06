"""Build data/pitchfork_discogs.csv, the dataset used by main_analysis.ipynb.

Joins the Pitchfork reviews (data/pitchfork.csv) with the Discogs ratings (data/discogs.csv,
built by combine_discogs.py) and keeps one row per album that passes these filters:
  1. matched on Discogs with at least 10 community ratings (fewer is too noisy)
  2. has Spotify audio features (-1 in every feature = missing)
  3. reviewed within 2 years of release (later reviews are retrospective reissue reviews)
  4. a studio album (no compilations, live albums, box sets, deluxe/remastered editions)
Then adds percentile ranks, each artist's earlier average ratings, and sonic atypicality.

Usage (from the repository root): python src/scraping/build_dataset.py
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

PITCHFORK = "data/pitchfork.csv"
DISCOGS = "data/discogs.csv"
OUTPUT = "data/pitchfork_discogs.csv"

FEATURES = ["danceability", "energy", "loudness", "speechiness", "acousticness",
            "instrumentalness", "liveness", "valence", "tempo"]
MIN_RATINGS = 10
MAX_REVIEW_LAG = 2  # years between release and review
# Compilations, live albums and reissue packages: collectors rate them as objects, critics as records
NOT_STUDIO = (r"\b(?:live|sessions?|remix(?:es)?|deluxe|box|collection|anthology|greatest hits|best of|singles|"
              r"b-sides|rarities|reissue|remaster(?:ed)?|edition|compilation|in concert|peel)\b|\d{2,4}\s*[-–]\s*\d{2,4}")


def report(label, d):
    print(f"  {label:55s} {len(d):>7,} rows")


def load_and_filter():
    full = pd.read_csv(PITCHFORK)
    full["date"] = pd.to_datetime(full["reviewdate"], errors="coerce")
    full["genre"] = full["genre"].replace("none", pd.NA).fillna("Unknown")
    report("Pitchfork reviews", full)

    discogs = pd.read_csv(DISCOGS).drop_duplicates(["artist", "album"])
    d = full.merge(discogs[["artist", "album", "rating_avg", "rating_count", "have", "want", "match_quality"]],
                   on=["artist", "album"])
    report("joined with Discogs lookups", d)
    d = d[d["rating_count"] >= MIN_RATINGS]
    report(f"with >= {MIN_RATINGS} Discogs ratings", d)
    d = d[~(d[FEATURES] == -1).all(axis=1)]
    report("with Spotify audio features", d)
    d["lag"] = d["date"].dt.year - d["releaseyear"]
    d = d[d["lag"] <= MAX_REVIEW_LAG]
    report(f"reviewed within {MAX_REVIEW_LAG} years of release", d)
    d = d.drop_duplicates(["artist", "album"])
    report("one row per album", d)
    d = d[~d["album"].str.contains(NOT_STUDIO, case=False, regex=True)].reset_index(drop=True)
    report("studio albums only", d)
    return full, d


def add_derived_columns(d, full):
    d["pP"] = d["score"].rank(pct=True)        # critic percentile
    d["pD"] = d["rating_avg"].rank(pct=True)   # public percentile
    d["gap"] = d["pP"] - d["pD"]               # > 0: critics liked it more than the public

    # Each artist's average rating over their *earlier* albums in the sample (NaN for a first album)
    s = d.sort_values("date")
    for col, name in [("score", "P"), ("rating_avg", "D")]:
        d[f"prior_{name}"] = s.groupby("artist")[col].transform(lambda x: x.shift().expanding().mean())

    # Sonic atypicality: mean distance to the 10 most similar albums released the same year
    pool = full[~(full[FEATURES] == -1).all(axis=1)].drop_duplicates(["artist", "album"])
    scaler = StandardScaler().fit(pool[FEATURES])
    d["atypicality"] = np.nan
    for year, x in d.groupby("releaseyear"):
        ref = pool[pool["releaseyear"] == year]
        if len(ref) < 30:
            continue
        nn = NearestNeighbors(n_neighbors=11).fit(scaler.transform(ref[FEATURES]))
        dist, _ = nn.kneighbors(scaler.transform(x[FEATURES]))
        d.loc[x.index, "atypicality"] = dist[:, 1:].mean(axis=1)  # first neighbor is the album itself

    d["log_have"] = np.log1p(d["have"])
    return d


def main():
    print("Building the merged dataset:")
    full, d = load_and_filter()
    d = add_derived_columns(d, full)
    d.to_csv(OUTPUT, index=False)
    print(f"Saved {OUTPUT} ({len(d):,} albums, {d.shape[1]} columns)")


if __name__ == "__main__":
    main()
