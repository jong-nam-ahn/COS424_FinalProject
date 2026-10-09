"""Collect Discogs community ratings for the albums in output-data.csv (Pitchfork).

For each album: search Discogs masters by artist + title, pick the best match,
then read the community rating (average + count) of that master's main release.

Usage:
    export DISCOGS_TOKEN=...            # https://www.discogs.com/settings/developers
    python collect_discogs.py --limit 20             # quick test
    python collect_discogs.py --hours 4              # stop cleanly after 4 hours
    python collect_discogs.py --genres Rock Rap Electronic Experimental Pop/R&B
    python collect_discogs.py                        # everything (~3 requests per album)

Progress is appended to the output CSV after every album, so the script can be
stopped and re-run at any time; albums already in the output are skipped.
Albums are processed in a seeded random order, so a partial run is a random sample.
"""
import argparse
import csv
import os
import re
import sys
import time
from difflib import SequenceMatcher

import pandas as pd
import requests

from text_norm import normalize, strip_format_suffix

API = "https://api.discogs.com"
USER_AGENT = "COS424-FinalProject/0.1 (Princeton course project)"
MIN_RATINGS = 10  # below this, the Discogs average is too noisy to use
FIELDS = ["artist", "album", "releaseyear", "status", "match_quality", "enough_ratings",
          "discogs_master_id", "discogs_release_id", "discogs_artist", "discogs_title", "discogs_year",
          "rating_avg", "rating_count", "have", "want"]


class Discogs:
    """Minimal Discogs client that stays under the rate limit (60/min with a token, 25/min without)."""

    def __init__(self, token):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        if token:
            self.session.headers["Authorization"] = f"Discogs token={token}"
        self.min_interval = 60 / (60 if token else 25)
        self.last_call = 0.0

    def get(self, path, **params):
        for attempt in range(5):
            wait = self.min_interval - (time.time() - self.last_call)
            if wait > 0:
                time.sleep(wait)
            self.last_call = time.time()
            try:
                r = self.session.get(f"{API}{path}", params=params, timeout=30)
            except requests.RequestException as e:
                print(f"  network error ({e}); retrying", file=sys.stderr)
                time.sleep(10 * (attempt + 1))
                continue
            if r.status_code == 429:  # rate limited: the window is a moving 60 seconds
                print("  rate limited; sleeping 60s", file=sys.stderr)
                time.sleep(60)
                continue
            if r.status_code == 404:
                return None
            if r.status_code >= 500:
                time.sleep(10 * (attempt + 1))
                continue
            r.raise_for_status()
            if r.headers.get("X-Discogs-Ratelimit-Remaining") == "0":
                time.sleep(60)
            return r.json()
        raise RuntimeError(f"giving up on {path} {params}")


def split_discogs_title(title):
    """Discogs search titles look like 'Artist - Album'; artists may carry ' (2)' or '*' suffixes."""
    artist, _, album = title.partition(" - ")
    artist = re.sub(r"\s*\(\d+\)$", "", artist).rstrip("*").strip()
    return artist, album.strip()


def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def pick_best(results, artist, album, year):
    """Return (result, match_quality) for the best candidate, or (None, None)."""
    want_artist, want_album = normalize(artist), normalize(album)
    exact, fuzzy = [], []
    for res in results:
        d_artist, d_album = split_discogs_title(res.get("title", ""))
        sa, sb = similarity(want_artist, normalize(d_artist)), similarity(want_album, normalize(d_album))
        if sa == 1 and sb == 1:
            exact.append(res)
        elif sa >= 0.85 and sb >= 0.85:
            fuzzy.append((sa + sb, res))
    if exact:
        # Several masters can share a title (bootlegs, multiple self-titled albums): prefer the
        # ones released within a year of Pitchfork's year, then the most-collected one.
        def year_ok(res):
            return pd.notna(year) and str(res.get("year", "")).isdigit() and abs(int(res["year"]) - year) <= 1
        pool = [r for r in exact if year_ok(r)] or exact
        return max(pool, key=lambda r: r.get("community", {}).get("have", 0)), "exact"
    if fuzzy:
        return max(fuzzy, key=lambda t: t[0])[1], "fuzzy"
    return None, None


def lookup(client, artist, album, year):
    row = {"artist": artist, "album": album, "releaseyear": year}
    # Pitchfork often appends the format ("STFU II EP") where Discogs doesn't ("STFU II")
    titles = list(dict.fromkeys([album, strip_format_suffix(album)]))
    queries = [{"artist": artist, "release_title": t} for t in titles]
    queries += [{"q": f"{artist} {t}"} for t in titles]  # free text tolerates naming differences

    # Prefer masters (one per album, across all pressings); some albums only exist as releases.
    best, quality, kind = None, None, None
    for search_type in ("master", "release"):
        for query in queries:
            data = client.get("/database/search", type=search_type, per_page=25, **query)
            best, quality = pick_best((data or {}).get("results", []), artist, album, year)
            if best is not None:
                break
        if best is not None:
            kind = search_type
            break
    if best is None:
        return {**row, "status": "not_found"}

    d_artist, d_title = split_discogs_title(best["title"])
    row.update(match_quality=quality, discogs_artist=d_artist, discogs_title=d_title, discogs_year=best.get("year"))
    if kind == "master":
        master = client.get(f"/masters/{best['id']}")
        if not master or not master.get("main_release"):
            return {**row, "status": "no_main_release", "discogs_master_id": best["id"]}
        row["discogs_master_id"] = best["id"]
        release_id = master["main_release"]
    else:
        release_id = best["id"]

    release = client.get(f"/releases/{release_id}")
    if not release:
        return {**row, "status": "release_missing", "discogs_release_id": release_id}
    community = release.get("community", {})
    rating = community.get("rating", {})
    count = rating.get("count") or 0
    row.update(status="ok", discogs_release_id=release_id,
               rating_avg=rating.get("average") if count else None,  # Discogs reports 0.0 when unrated
               rating_count=count, enough_ratings=count >= MIN_RATINGS,
               have=community.get("have"), want=community.get("want"))
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="data/output-data.csv")
    ap.add_argument("--output", default="data/discogs_ratings.csv")
    ap.add_argument("--genres", nargs="*", help="only these Pitchfork genres, e.g. Rock Rap 'Pop/R&B'")
    ap.add_argument("--limit", type=int, help="stop after this many new albums (for testing)")
    ap.add_argument("--hours", type=float, help="stop cleanly after this many hours (e.g. 4)")
    ap.add_argument("--seed", type=int, default=0, help="seed for the processing order")
    args = ap.parse_args()

    token = os.environ.get("DISCOGS_TOKEN")
    if not token:
        print("DISCOGS_TOKEN not set: running unauthenticated at 25 requests/min.", file=sys.stderr)

    pf = pd.read_csv(args.input)
    if args.genres:
        pf = pf[pf["genre"].isin(args.genres)]
    pf = pf[pf["artist"].str.lower() != "various artists"]
    # One lookup per album: reissue reviews of the same album share a Discogs entry
    pf = pf.drop_duplicates(["artist", "album"]).sample(frac=1, random_state=args.seed)

    prev = pd.read_csv(args.output, encoding="utf-8") if os.path.exists(args.output) else pd.DataFrame(columns=FIELDS)
    done = set(zip(prev["artist"], prev["album"]))
    todo = [r for r in pf.itertuples() if (r.artist, r.album) not in done]
    total = len(pf)  # albums in scope for this run (after --genres)
    progress = Progress(total, already=total - len(todo), prev=prev)
    if args.limit is not None:
        todo = todo[:args.limit]
    progress.summary(f"Starting: {len(todo):,} to look up this run "
                     f"(~{len(todo) * 3 / (60 if token else 25) / 60:.1f} hours at most)")

    client = Discogs(token)
    new_file = not os.path.exists(args.output)
    with open(args.output, "a", newline="", encoding="utf-8") as f:  # Windows defaults to cp1252
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        deadline = time.time() + args.hours * 3600 if args.hours else None
        try:
            for r in todo:
                if deadline and time.time() >= deadline:
                    progress.summary(f"Reached the {args.hours:g}-hour limit. Re-run the same command to continue")
                    return
                row = lookup(client, r.artist, r.album, r.releaseyear)
                writer.writerow(row)
                f.flush()
                progress.add(row)
        except KeyboardInterrupt:
            progress.summary("Stopped. Re-run the same command to continue")
            return
    progress.summary("Done")


class Progress:
    """Running tally of what has been collected, printed as one updating line."""

    def __init__(self, total, already, prev):
        self.total, self.collected = total, already
        self.ok = int((prev["status"] == "ok").sum())
        self.usable = int((prev["enough_ratings"].astype(str) == "True").sum())
        self.not_found = int((prev["status"] == "not_found").sum())
        self.new, self.start = 0, time.time()
        self.tty = sys.stdout.isatty()

    def line(self):
        pct = 100 * self.collected / self.total if self.total else 100
        text = (f"Collected {self.collected:,}/{self.total:,} ({pct:.1f}%) | matched {self.ok:,} | "
                f">={MIN_RATINGS} ratings {self.usable:,} | not found {self.not_found:,}")
        if self.new:
            rate = self.new / (time.time() - self.start)  # albums per second this run
            text += f" | ~{(self.total - self.collected) / rate / 3600:.1f}h left"
        return text

    def add(self, row):
        self.collected += 1
        self.new += 1
        self.ok += row["status"] == "ok"
        self.usable += bool(row.get("enough_ratings"))
        self.not_found += row["status"] == "not_found"
        if self.tty:  # overwrite one line in a terminal; log every 25 albums when redirected
            print("\r" + self.line().ljust(120), end="", flush=True)
        elif self.new % 25 == 0:
            print(self.line(), flush=True)

    def summary(self, message):
        if self.tty and self.new:  # end the live progress line first
            print()
        print(f"{message}\n  {self.line()}", flush=True)

if __name__ == "__main__":
    main()
