"""Shared name normalization for matching albums across Pitchfork, RYM and Discogs."""
import re
import unicodedata

import pandas as pd


def normalize(s, drop_brackets=True):
    if pd.isna(s):
        return ""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))  # strip accents
    s = s.lower()
    s = re.sub(r"[‘’´`']", "", s)                  # drop apostrophes of any kind
    s = s.replace("&", " and ").replace("+", " and ")
    if drop_brackets:
        stripped = re.sub(r"\s*[\(\[].*?[\)\]]", "", s)  # drop "(Deluxe Edition)", "[EP]", ...
        s = stripped if stripped.strip() else s          # ...unless that empties it ("[Untitled]")
    s = re.sub(r"[\W_]+", " ", s)                  # punctuation -> space (keeps non-Latin letters)
    s = re.sub(r"^the\s+", "", s.strip())          # leading "the"
    s = re.sub(r"(?<=\S)\s+(ep|lp)$", "", s)       # format suffix: "STFU II EP" -> "stfu ii"
    return s.strip()


def strip_format_suffix(title):
    """Raw title without a trailing ' EP' / ' LP' (for search queries); unchanged if nothing is left."""
    stripped = re.sub(r"\s+(EP|LP)\s*$", "", str(title), flags=re.IGNORECASE).strip()
    return stripped or str(title)
