"""Exploratory data analysis helpers for main_analysis.ipynb.

Every function either returns a table (DataFrame/Series) or draws one figure, so the notebook
only needs one-line calls. Discogs ratings (1-5 stars) are put on Pitchfork's 0-10 scale by
multiplying by 2 (column `rating_10`).
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/pitchfork_discogs.csv"

AUDIO_FEATURES = ["danceability", "energy", "key", "loudness", "speechiness",
                  "acousticness", "instrumentalness", "liveness", "valence", "tempo"]
# PCA leaves out `key`: it is a category (C, C#, ..., B coded 0-11), not a quantity to standardize
PCA_FEATURES = [f for f in AUDIO_FEATURES if f != "key"]
NUMERIC_COLUMNS = ["releaseyear", "score", "rating_avg", "rating_count"] + AUDIO_FEATURES
SUMMARY_STATS = ["mean", "std", "min", "median", "max", "count"]


# ----------------------------------------------------------------------------- setup and loading
def setup_style():
    """Default look for every plot and table in the notebook."""
    sns.set_theme(style="whitegrid", context="notebook")
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 120)


def load_data(path=DATA_PATH):
    """Load the merged Pitchfork x Discogs dataset.

    Parses `reviewdate` as a date and adds `rating_10`, the Discogs average rating on a 0-10 scale.
    """
    df = pd.read_csv(path)
    df["reviewdate"] = pd.to_datetime(df["reviewdate"])
    df["rating_10"] = df["rating_avg"] * 2
    return df


# ----------------------------------------------------------------------------- tables
def describe_numeric(df, columns=NUMERIC_COLUMNS):
    """Descriptive statistics of the numeric columns."""
    return df[columns].describe().round(3)


def summarize(df, by, value="score", sort_by=None, ascending=False):
    """mean / std / min / median / max / count of `value` for each group in `by`.

    `by` is a column name or a Series (e.g. review year). `sort_by` is a statistic name;
    by default the table is sorted by group.
    """
    table = df.groupby(by, observed=True)[value].agg(SUMMARY_STATS).round(3)
    return table.sort_values(sort_by, ascending=ascending) if sort_by else table.sort_index()


def feature_correlations(df, target="score", features=AUDIO_FEATURES):
    """Pearson correlation of each audio feature with `target`, sorted ascending."""
    return df[features + [target]].corr()[target].drop(target).sort_values()


def count_late_reviews(df, max_gap_years=1):
    """Number of reviews published more than `max_gap_years` after the album's release year."""
    return int((df["reviewdate"].dt.year - df["releaseyear"] > max_gap_years).sum())


def add_review_era(df, bins=(1999, 2005, 2010, 2015, 2020),
                   labels=("1999-2004", "2005-2009", "2010-2014", "2015-2019")):
    """Add `years_bin`: the review year grouped into roughly half-decade ranges."""
    df["years_bin"] = pd.cut(df["reviewdate"].dt.year, bins=list(bins), labels=list(labels), right=False)
    return df


def add_count_bins(df, q=5):
    """Add `counts_bin` (0 = fewest ratings): rating_count split into `q` equal-sized bins.

    Returns the bin edges.
    """
    df["counts_bin"], edges = pd.qcut(df["rating_count"], q, labels=False, retbins=True)
    return edges


# ----------------------------------------------------------------------------- plots
def _finish(title, xlabel=None, ylabel=None):
    plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()


def plot_distribution(values, title):
    """Histogram with KDE, plus mean (solid) and median (dashed) lines."""
    plt.figure(figsize=(8, 6))
    sns.histplot(values, bins=20, kde=True, color="cornflowerblue")
    plt.axvline(values.mean(), color="tomato", linestyle="-", label=f"Mean: {values.mean():.2f}")
    plt.axvline(values.median(), color="darkorange", linestyle="--", label=f"Median: {values.median():.2f}")
    plt.legend()
    _finish(title)


def plot_feature_correlations(correlations, target_label="Pitchfork Score"):
    """Horizontal bars: red for positive, blue for negative correlations."""
    plt.figure(figsize=(8, 6))
    colors = ["cornflowerblue" if v < 0 else "tomato" for v in correlations]
    correlations.plot(kind="barh", color=colors)
    plt.axvline(0, color="black", linewidth=1)
    _finish(f"Musical Feature Correlations with {target_label}s",
            xlabel=f"Pearson Correlation with {target_label}")


def plot_box_by(df, value, group, order, title, xlabel, ylabel, color, figsize=(10, 6)):
    """Horizontal boxplot of `value` for each `group`, in the given order (outliers hidden)."""
    plt.figure(figsize=figsize)
    sns.boxplot(data=df, x=value, y=group, order=order, color=color, orient="h", showfliers=False)
    plt.xlim(left=0)
    _finish(title, xlabel, ylabel)


def plot_reviews_per_year(year_summary):
    """Number of reviews per review year (from `summarize(df, review_year)`)."""
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=year_summary.index, y=year_summary["count"], color="darkorange")
    plt.xlim(left=1998)
    plt.xticks(range(2000, 2020, 5))
    _finish("Number of Albums Pitchfork Reviewed by Review Year", "Review Year", "Number of Albums Reviewed")


def plot_mean_median_per_year(year_summary):
    """Mean and median score per review year (from `summarize(df, review_year)`)."""
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=year_summary.index, y=year_summary["mean"], color="darkorange", label="Mean")
    sns.lineplot(x=year_summary.index, y=year_summary["median"], color="slateblue", label="Median")
    plt.xlim(left=1998)
    plt.xticks(range(2000, 2020, 5))
    plt.ylim(0, 10)
    _finish("Mean and Median Scores by Review Year", "Review Year", "Score")


def plot_reviews_per_reviewer(reviewer_summary):
    """Review count of each reviewer, ranked from most to fewest reviews."""
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=range(len(reviewer_summary)), y=reviewer_summary["count"].values, color="mediumpurple")
    _finish("Number of Reviews by Reviewer", "Reviewer (ranked by review count)", "Number of Reviews")


def plot_rating_count_hist(df):
    """Histogram of the number of Discogs ratings per album (log y-axis)."""
    plt.figure(figsize=(8, 6))
    sns.histplot(df["rating_count"], bins=80, color="cadetblue")
    plt.yscale("log")
    _finish("Histogram of rating counts", "Rating count", "Number of Albums with the rating count (log)")


def umap_embedding(df, n_genres=5, features=AUDIO_FEATURES, random_state=424):
    """2-D UMAP of the standardized audio features for albums in the `n_genres` largest genres.

    Returns (embedding array, genre Series aligned with it).
    """
    top = df["genre"].value_counts().head(n_genres).index
    subset = df[df["genre"].isin(top)]
    scaled = StandardScaler().fit_transform(subset[features].values)

    # Silence two harmless UMAP warnings: tqdm asking for ipywidgets (progress bars), and
    # "n_jobs overridden to 1" (a fixed random_state makes UMAP single-threaded, as intended).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import umap  # imported here: slow to import and only needed for this plot
        embedding = umap.UMAP(random_state=random_state, n_jobs=1).fit_transform(scaled)
    return embedding, subset["genre"]


def plot_umap(embedding, genres):
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=genres.values,
                    alpha=0.5, s=20, linewidth=0, palette="Dark2")
    plt.legend(title="Genre", loc="upper right")
    _finish("UMAP of Musicality Features by Genre")


def fit_pca(df, features=PCA_FEATURES):
    """Fit a PCA (all components) on the standardized audio features and return it."""
    from sklearn.decomposition import PCA

    return PCA().fit(StandardScaler().fit_transform(df[features]))


def pca_variance_table(pca):
    """Share of the total variance explained by each component, and cumulatively."""
    ratio = pca.explained_variance_ratio_
    return pd.DataFrame({"explained": ratio, "cumulative": ratio.cumsum()},
                        index=[f"PC{i + 1}" for i in range(len(ratio))]).round(3)






def plot_pca_variance(pca):
    """Bars: variance explained by each component; line: cumulative share."""
    table = pca_variance_table(pca)
    x = range(1, len(table) + 1)
    plt.figure(figsize=(9, 5))
    plt.bar(x, table["explained"], color="cornflowerblue", label="Individual component")
    plt.plot(x, table["cumulative"], marker="o", color="tomato", label="Cumulative")
    plt.xticks(list(x))
    plt.ylim(0, 1.05)
    plt.legend(loc="center right")
    _finish("PCA of Audio Features: Explained Variance", "Principal component", "Share of variance explained")






def _overlay_kde(ax, scores, ratings):
    sns.kdeplot(scores, color="cornflowerblue", fill=True, alpha=0.5, label="Pitchfork", ax=ax)
    ax.axvline(scores.mean(), color="royalblue", linestyle="--")
    sns.kdeplot(ratings, color="darkorange", fill=True, alpha=0.5, label="Discogs", ax=ax)
    ax.axvline(ratings.mean(), color="tomato", linestyle="--")


def plot_pitchfork_vs_discogs(df):
    """Overlaid KDEs of Pitchfork scores and Discogs ratings (0-10), dashed lines at the means."""
    fig, ax = plt.subplots(figsize=(8, 6))
    _overlay_kde(ax, df["score"], df["rating_10"])
    ax.legend(loc="upper left")
    _finish("Distribution of Pitchfork Scores vs Discogs Ratings")


def rank_agreement(df, n_boot=1000, seed=0):
    """How well do Pitchfork and Discogs agree on the *ordering* of albums?

    Returns the Spearman rank correlation with a 95% bootstrap CI, and the share of Pitchfork's
    top-25% albums that are also in Discogs' top 25% (25% would mean no relation).
    """
    score, rating = df["score"].to_numpy(), df["rating_avg"].to_numpy()

    def spearman(i):
        return pd.Series(score[i]).corr(pd.Series(rating[i]), method="spearman")

    rng = np.random.default_rng(seed)
    boot = [spearman(rng.integers(0, len(df), len(df))) for _ in range(n_boot)]

    return pd.Series({
        "n_albums": len(df),
        "spearman": spearman(np.arange(len(df))),
        "ci_lo": np.percentile(boot, 2.5),
        "ci_hi": np.percentile(boot, 97.5),
    }).round(3)


def plot_percentile_scatter(df):
    """Each album's Pitchfork percentile vs its Discogs percentile; dashed line = perfect agreement."""
    pct_p = df["score"].rank(pct=True) * 100
    pct_d = df["rating_avg"].rank(pct=True) * 100
    rho = df["score"].corr(df["rating_avg"], method="spearman")
    plt.figure(figsize=(7, 7))
    plt.scatter(pct_p, pct_d, s=5, alpha=0.3, color="cornflowerblue", linewidths=0)
    plt.plot([0, 100], [0, 100], color="tomato", linestyle="--", label="Perfect agreement")
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.legend(loc="upper left")
    _finish(f"Pitchfork vs Discogs Percentile Rank (Spearman ρ = {rho:.2f}, n = {len(df):,})",
            "Pitchfork score (percentile)", "Discogs rating (percentile)")


def plot_pitchfork_vs_discogs_by_genre(df, legend_genre="Electronic"):
    """One overlaid-KDE panel per genre (2 x 5 grid)."""
    fig, axes = plt.subplots(2, 5, figsize=(14, 7), sharex=True, sharey=True)
    for ax, (genre, group) in zip(axes.flatten(), df.groupby("genre")):
        _overlay_kde(ax, group["score"], group["rating_10"])
        ax.set_title(genre)
        ax.set_xlabel("Score/Rating")
        if genre == legend_genre:
            ax.legend(loc="upper left")
    fig.suptitle("Distribution of Pitchfork Scores vs Discogs Ratings by Genre")
    fig.tight_layout()
    plt.show()
