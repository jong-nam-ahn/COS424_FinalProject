"""Combine the per-person Discogs collections (data/discogs-<name>.csv) into data/discogs.csv.

Usage (from the repository root): python src/scraping/combine_discogs.py

Handles files written on Windows, where early runs of collect_discogs.py saved some lines
as Windows-1252 instead of UTF-8: each line is decoded with whichever encoding it uses.
Albums looked up twice (after a resume) are kept once.
"""
import glob
import io

import pandas as pd

OUTPUT = "data/discogs.csv"


def read_mixed_encoding(path):
    lines = []
    for raw in open(path, "rb").read().split(b"\n"):
        try:
            lines.append(raw.decode("utf-8"))
        except UnicodeDecodeError:
            lines.append(raw.decode("cp1252"))
    text = "\n".join(lines).replace("\r", "")
    return pd.read_csv(io.StringIO(text))


def main():
    parts = []
    for path in sorted(glob.glob("data/discogs-*.csv")):
        part = read_mixed_encoding(path)
        part["collected_by"] = path.split("discogs-")[1].removesuffix(".csv")
        print(f"{path}: {len(part):,} rows, {part.duplicated(['artist', 'album']).sum()} repeated lookups")
        parts.append(part)
    if not parts:
        raise SystemExit("No data/discogs-<name>.csv files found. Run collect_discogs.py with "
                         "--output data/discogs-<name>.csv first.")
    combined = pd.concat(parts, ignore_index=True).drop_duplicates(["artist", "album"])

    # Every album in the Pitchfork data should have exactly one row
    pf = pd.read_csv("data/pitchfork.csv").drop_duplicates(["artist", "album"])
    pf = pf[pf["artist"].str.lower() != "various artists"]
    expected, got = set(zip(pf["artist"], pf["album"])), set(zip(combined["artist"], combined["album"]))
    print(f"\nCombined: {len(combined):,} albums | expected {len(expected):,} | "
          f"missing {len(expected - got)} | unexpected {len(got - expected)}")
    print("Status:", combined["status"].value_counts().to_dict())
    print(f"With >= 10 ratings: {(combined['enough_ratings'].astype(str) == 'True').sum():,}")

    combined.to_csv(OUTPUT, index=False, encoding="utf-8")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
