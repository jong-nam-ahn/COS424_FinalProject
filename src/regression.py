"""Baseline score-difference regression (proposal section 3.1).

    Score_ij = b0 + b1 * IsCritic_ij + e_ij,   j in {Pitchfork, Discogs}

Each album contributes one row per source, so the data are stacked to long format and
standard errors are clustered by album (the two rows of an album are not independent).
Discogs ratings (1-5 stars) must be put on Pitchfork's 0-10 scale first; b1 depends on that
choice, so two rescalings are reported:
  "x2"     : 2 * rating               (5 stars -> 10, 1 star -> 2)
  "linear" : (rating - 1) * 10 / 4    (1 star -> 0, 5 stars -> 10; matches both scale ends)
"""
import numpy as np
import pandas as pd

RESCALE = {
    "x2": lambda r: 2 * r,
    "linear": lambda r: (r - 1) * 10 / 4,
}


def to_long(d, rescale="x2"):
    """One row per (album, source) with columns: album_id, source, is_critic, score, genre."""
    base = d.reset_index(drop=True).assign(album_id=lambda x: x.index)
    critic = base.assign(source="Pitchfork", is_critic=1, score_10=base["score"])
    crowd = base.assign(source="Discogs", is_critic=0, score_10=RESCALE[rescale](base["rating_avg"]))
    cols = ["album_id", "artist", "album", "genre", "source", "is_critic", "score_10"]
    return pd.concat([critic[cols], crowd[cols]], ignore_index=True)


def ols_clustered(y, X, groups):
    """OLS with cluster-robust (CR1) standard errors. X must include the intercept column."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    meat = np.zeros((X.shape[1], X.shape[1]))
    codes, uniq = pd.factorize(np.asarray(groups))
    for g in range(len(uniq)):
        s = X[codes == g].T @ resid[codes == g]
        meat += np.outer(s, s)
    n, k, G = len(y), X.shape[1], len(uniq)
    V = XtX_inv @ meat @ XtX_inv * (G / (G - 1)) * ((n - 1) / (n - k))
    return beta, np.sqrt(np.diag(V))


def baseline(d, rescale="x2"):
    """Fit eq. (1). Returns a one-row-per-coefficient table."""
    long = to_long(d, rescale)
    X = np.column_stack([np.ones(len(long)), long["is_critic"]])
    beta, se = ols_clustered(long["score_10"], X, long["album_id"])
    return pd.DataFrame({"coef": beta, "se": se, "ci_lo": beta - 1.96 * se, "ci_hi": beta + 1.96 * se},
                        index=["b0 (Discogs mean)", "b1 IsCritic"]).round(3)


def by_genre(d, rescale="x2", min_n=40):
    """Eq. (1) fit separately within each genre: b1 = average critic - crowd gap in that genre."""
    rows = []
    for g, x in d.groupby("genre"):
        if len(x) < min_n:
            continue
        t = baseline(x, rescale).loc["b1 IsCritic"]
        rows.append({"genre": g, "n_albums": len(x), **t.to_dict()})
    return pd.DataFrame(rows).set_index("genre").sort_values("coef")


if __name__ == "__main__":  # run from the repo root: python -m src.regression
    d = pd.read_csv("data/pitchfork_discogs.csv")
    print(f"n = {len(d):,} albums -> {2 * len(d):,} album-source rows\n")
    for r in RESCALE:
        print(f"--- Discogs rescaled '{r}' ---")
        print(baseline(d, r).to_string(), "\n")
    print("--- b1 by genre ('x2' rescaling) ---")
    print(by_genre(d).to_string())
