"""
Best-Selling Nintendo Switch Games - Exploratory Data Analysis
=============================================================

Improved English version of the data-analysis project for the
Programming Fundamentals course (Universidad EAFIT, 2022-1).

The dataset lists the 43 Nintendo Switch games with more than one million
copies sold (Kaggle "Top Selling Switch Games", compiled from Wikipedia).
The script:

1. Loads and cleans the data (dates, regional publisher prefixes, duplicated
   developer names) and derives new variables (years on market, sales per
   year, publisher ecosystem).
2. Computes descriptive statistics by genre, developer and publisher.
3. Tests the four hypotheses of the original report with explicit criteria
   and, where useful, a non-parametric test (Mann-Whitney U, Spearman).
4. Saves every table (CSV / JSON) and figure (PNG) and opens a window with
   the summary, the data table and one interactive tab per figure.

Usage
-----
    python switch_sales_analysis.py            (or press "Run" in VS Code)
    python switch_sales_analysis.py --no-gui   (only write the outputs)

If numpy, pandas, matplotlib or scipy are missing, the first run creates a
local environment (.venv in the project folder), installs them there and
re-launches itself. Works with uv-managed / externally managed Pythons.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENV_DIR = PROJECT_ROOT / ".venv"
REQUIRED = ["numpy", "pandas", "matplotlib", "scipy"]


# ---------------------------------------------------------------------------
# 0. One-click dependency bootstrap
# ---------------------------------------------------------------------------
def _missing_modules() -> list[str]:
    return [m for m in REQUIRED if importlib.util.find_spec(m) is None]


def _venv_python() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def _install(python: str, packages: list[str]) -> None:
    uv = shutil.which("uv")
    if uv:
        subprocess.check_call([uv, "pip", "install", "--python", python, *packages])
    else:
        subprocess.check_call([python, "-m", "pip", "install", *packages])


def _ensure_dependencies() -> None:
    """Create .venv, install the libraries and re-launch if anything is missing."""
    missing = _missing_modules()
    if not missing:
        return

    in_venv = Path(sys.prefix).resolve() == VENV_DIR.resolve()
    if in_venv or os.environ.get("SWITCH_EDA_BOOTSTRAPPED"):
        print(f"Installing missing libraries: {', '.join(missing)} ...")
        _install(sys.executable, missing)
        importlib.invalidate_caches()
        return

    py = _venv_python()
    if not py.exists():
        if VENV_DIR.exists():
            # Incomplete environment or one created by another operating system.
            print(f"Removing unusable environment in {VENV_DIR} ...")
            shutil.rmtree(VENV_DIR, ignore_errors=True)
        print(f"Creating a local Python environment in {VENV_DIR} ...")
        uv = shutil.which("uv")
        if uv:
            subprocess.check_call([uv, "venv", str(VENV_DIR), "--python", sys.executable])
        else:
            subprocess.check_call([sys.executable, "-m", "venv", str(VENV_DIR)])

    check = subprocess.run([str(py), "-c", "import " + ", ".join(REQUIRED)], capture_output=True)
    if check.returncode != 0:
        print(f"Installing {', '.join(REQUIRED)} (first run only, may take a minute) ...")
        _install(str(py), REQUIRED)
        print("Libraries installed.\n")

    env = dict(os.environ, SWITCH_EDA_BOOTSTRAPPED="1")
    raise SystemExit(subprocess.call([str(py), str(Path(__file__).resolve()), *sys.argv[1:]], env=env))


_ensure_dependencies()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy import stats  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA_FILE = PROJECT_ROOT / "01_Data" / "switch_top_selling_games.csv"
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Plot style (validated categorical palette: blue / orange, gray neutral)
# ---------------------------------------------------------------------------
BLUE = "#2a78d6"      # Nintendo ecosystem / highlighted series
ORANGE = "#eb6834"    # third-party
GREEN = "#1baf7a"
NEUTRAL = "#b9b8b0"   # "other" / below reference
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"

plt.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": INK_2,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "axes.titlelocation": "left",
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "legend.frameon": False,
})

# Nintendo-owned or Nintendo-affiliated publishers. The Pokemon Company is
# co-owned by Nintendo, so its games count as first/second party.
ECOSYSTEM_PUBLISHERS = {"Nintendo", "The Pokémon Company"}

# Genre families used as a sensitivity check for hypothesis 2.
GENRE_FAMILY = {
    "Role-playing": "Role-playing (all sub-genres)",
    "Action role-playing": "Role-playing (all sub-genres)",
    "Tactical role-playing": "Role-playing (all sub-genres)",
}


# ---------------------------------------------------------------------------
# 1. Load and clean
# ---------------------------------------------------------------------------
def load_data(path: Path = DATA_FILE) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.rename(columns={
        "No.": "rank", "Title": "title", "Copies sold": "sales_m", "As of": "as_of",
        "Release date": "release", "Genre(s)": "genre", "Developer(s)": "developer",
        "Publisher(s)": "publisher_raw",
    })
    df["as_of"] = pd.to_datetime(df["as_of"], format="%d-%b-%y")
    df["release"] = pd.to_datetime(df["release"], format="%d-%b-%y")

    # "JP: Konami" means Konami published the game in Japan; the dataset keeps
    # only the first regional publisher, so the prefix is removed and flagged.
    df["regional_publisher"] = df["publisher_raw"].str.contains(r"^[A-Z/]+:\s", regex=True)
    df["publisher"] = df["publisher_raw"].str.replace(r"^[A-Z/]+:\s*", "", regex=True)

    # "Nintendo" (Super Mario 3D All-Stars) and "Nintendo EPD" are the same studio group.
    df["developer"] = df["developer"].replace({"Nintendo": "Nintendo EPD"})

    df["ecosystem"] = np.where(df["publisher"].isin(ECOSYSTEM_PUBLISHERS),
                               "First/second party", "Third party")
    df["years_on_market"] = (df["as_of"] - df["release"]).dt.days / 365.25
    df["sales_per_year_m"] = df["sales_m"] / df["years_on_market"]
    df["release_year"] = df["release"].dt.year
    df["genre_family"] = df["genre"].map(GENRE_FAMILY).fillna(df["genre"])
    return df.sort_values("sales_m", ascending=False).reset_index(drop=True)


def group_table(df: pd.DataFrame, key: str) -> pd.DataFrame:
    total = df["sales_m"].sum()
    g = (df.groupby(key)["sales_m"]
         .agg(titles="count", sales_m="sum", mean_m="mean", best_m="max")
         .sort_values(["sales_m", "titles"], ascending=False))
    g["sales_share_pct"] = 100 * g["sales_m"] / total
    g["title_share_pct"] = 100 * g["titles"] / len(df)
    return g.round(3)


# ---------------------------------------------------------------------------
# 2. Hypotheses
# ---------------------------------------------------------------------------
def evaluate(df: pd.DataFrame, genres: pd.DataFrame, devs: pd.DataFrame,
             pubs: pd.DataFrame, eco: pd.DataFrame) -> dict:
    total = float(df["sales_m"].sum())
    mean_title = float(df["sales_m"].mean())
    median_title = float(df["sales_m"].median())
    mean_dev = float(devs["sales_m"].mean())

    # H1 - Nintendo EPD is the top developer and above the developer mean
    top_dev = devs.index[0]
    h1_ok = top_dev == "Nintendo EPD" and devs.loc["Nintendo EPD", "sales_m"] > mean_dev
    above_dev = devs[devs["sales_m"] > mean_dev].index.tolist()

    # H2 - platformer leads sales, role-playing leads title count
    top_sales_genre = genres.index[0]
    top_count_genre = genres["titles"].idxmax()
    rpg_titles = int(genres.loc["Role-playing", "titles"])
    families = group_table(df, "genre_family")
    rpg_family = families.loc["Role-playing (all sub-genres)"]

    # H3 - first/second party dominate titles; compare sales per title
    first = df.loc[df["ecosystem"] == "First/second party", "sales_m"]
    third = df.loc[df["ecosystem"] == "Third party", "sales_m"]
    mw = stats.mannwhitneyu(first, third, alternative="greater")

    # H4 - most titles above the mean
    above = int((df["sales_m"] > mean_title).sum())
    skew = float(stats.skew(df["sales_m"]))

    # Extra: time on market vs sales
    rho, p_rho = stats.spearmanr(df["years_on_market"], df["sales_m"])
    top10_share = float(df["sales_m"].head(10).sum() / total)

    hypotheses = [
        {
            "id": "H1",
            "statement": "Nintendo EPD is the developer with the most sales and is above the mean sales per developer.",
            "result": "Supported" if h1_ok else "Rejected",
            "evidence": (f"Nintendo EPD: {devs.loc['Nintendo EPD', 'sales_m']:.2f} M copies in "
                         f"{int(devs.loc['Nintendo EPD', 'titles'])} titles ({devs.loc['Nintendo EPD', 'sales_share_pct']:.1f} % of all sales), "
                         f"{devs.loc['Nintendo EPD', 'sales_m'] / mean_dev:.1f} x the developer mean of {mean_dev:.2f} M. "
                         f"Only {len(above_dev)} of {len(devs)} developers are above the mean: {', '.join(above_dev)}."),
        },
        {
            "id": "H2",
            "statement": "Platformer is the genre with the most sales and role-playing the genre with the most titles.",
            "result": "Partly supported",
            "evidence": (f"Top genre by sales: {top_sales_genre} ({genres.loc[top_sales_genre, 'sales_m']:.2f} M, "
                         f"{genres.loc[top_sales_genre, 'sales_share_pct']:.1f} %). Top genre by titles: {top_count_genre} "
                         f"({int(genres.loc[top_count_genre, 'titles'])}) vs {rpg_titles} labelled 'Role-playing'. "
                         f"If action and tactical RPGs are grouped, the role-playing family has "
                         f"{int(rpg_family['titles'])} titles and {rpg_family['sales_m']:.2f} M copies."),
        },
        {
            "id": "H3",
            "statement": "Third-party companies have far fewer best-selling titles than first/second-party companies.",
            "result": "Supported",
            "evidence": (f"First/second party: {int(eco.loc['First/second party', 'titles'])} titles "
                         f"({eco.loc['First/second party', 'title_share_pct']:.1f} %) and "
                         f"{eco.loc['First/second party', 'sales_share_pct']:.1f} % of sales; third party: "
                         f"{int(eco.loc['Third party', 'titles'])} titles. Median sales per title "
                         f"{first.median():.2f} M vs {third.median():.2f} M "
                         f"(Mann-Whitney U = {mw.statistic:.0f}, one-sided p = {mw.pvalue:.4f})."),
        },
        {
            "id": "H4",
            "statement": "Most titles sell more than the mean of the list.",
            "result": "Rejected",
            "evidence": (f"Only {above} of {len(df)} titles ({100 * above / len(df):.1f} %) exceed the mean of "
                         f"{mean_title:.2f} M. The distribution is right-skewed (skewness {skew:.2f}); "
                         f"the median is {median_title:.2f} M and the top 10 titles hold {100 * top10_share:.1f} % of all sales."),
        },
    ]

    summary = {
        "dataset": {
            "titles": int(len(df)),
            "total_sales_m": round(total, 2),
            "mean_sales_per_title_m": round(mean_title, 4),
            "median_sales_per_title_m": round(median_title, 4),
            "std_sales_per_title_m": round(float(df["sales_m"].std()), 3),
            "skewness": round(skew, 3),
            "genres": int(len(genres)),
            "developers": int(len(devs)),
            "publishers": int(len(pubs)),
            "mean_sales_per_genre_m": round(float(genres["sales_m"].mean()), 3),
            "mean_sales_per_developer_m": round(mean_dev, 3),
            "top10_share_pct": round(100 * top10_share, 2),
            "regional_publisher_rows": int(df["regional_publisher"].sum()),
            "as_of_range": [str(df["as_of"].min().date()), str(df["as_of"].max().date())],
        },
        "time_on_market": {"spearman_rho": round(float(rho), 3), "p_value": round(float(p_rho), 4)},
        "mann_whitney": {"U": float(mw.statistic), "p_value_one_sided": round(float(mw.pvalue), 5),
                         "median_first_second_m": float(first.median()),
                         "median_third_m": float(third.median())},
        "hypotheses": hypotheses,
    }
    return summary


# ---------------------------------------------------------------------------
# 3. Figures
# ---------------------------------------------------------------------------
def _barh_labels(ax, bars, values, fmt="{:.1f}", pad=0.6):
    for b, v in zip(bars, values):
        ax.text(b.get_width() + pad, b.get_y() + b.get_height() / 2, fmt.format(v),
                va="center", ha="left", fontsize=7.5, color=INK_2,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.4, alpha=0.9))


def _legend(ax, items, lines=(), **kw):
    handles = [Patch(facecolor=c, label=l) for l, c in items]
    handles += [Line2D([], [], color=c, ls=ls, lw=1.2, label=l) for l, c, ls in lines]
    ax.legend(handles=handles, **kw)


def fig_top_titles(df):
    top = df.head(20).iloc[::-1]
    colors = [BLUE if e == "First/second party" else ORANGE for e in top["ecosystem"]]
    fig, ax = plt.subplots(figsize=(8, 6.4))
    bars = ax.barh(top["title"], top["sales_m"], color=colors, height=0.68)
    _barh_labels(ax, bars, top["sales_m"], "{:.2f}", pad=0.3)
    ax.set_xlabel("Copies sold (millions)")
    ax.set_title("Top 20 best-selling Nintendo Switch games")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, top["sales_m"].max() * 1.12)
    _legend(ax, [("First/second party", BLUE), ("Third party", ORANGE)], loc="lower right")
    fig.set_layout_engine("tight")
    return fig


def fig_genres(genres, families):
    g = genres.iloc[::-1]
    highlight = {"Platformer", "Role-playing"}
    colors = [BLUE if n in highlight else NEUTRAL for n in g.index]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 6.2), sharey=True,
                                 gridspec_kw={"width_ratios": [1.5, 1]})
    b1 = a1.barh(g.index, g["sales_m"], color=colors, height=0.7)
    _barh_labels(a1, b1, g["sales_m"], pad=0.5)
    a1.set_xlabel("Copies sold (millions)")
    a1.set_title("Sales by genre")
    a1.set_xlim(0, g["sales_m"].max() * 1.15)
    b2 = a2.barh(g.index, g["titles"], color=colors, height=0.7)
    _barh_labels(a2, b2, g["titles"], "{:.0f}", pad=0.1)
    a2.set_xlabel("Number of titles")
    a2.set_title("Titles by genre")
    a2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    a2.set_xlim(0, g["titles"].max() * 1.2)
    for a in (a1, a2):
        a.grid(axis="y", visible=False)
    rpg = families.loc["Role-playing (all sub-genres)"]
    fig.text(0.01, 0.01,
             f"Genres sorted by sales. Grouping role-playing, action RPG and tactical RPG gives "
             f"{int(rpg['titles'])} titles and {rpg['sales_m']:.1f} M copies.",
             fontsize=7.5, color=MUTED)
    fig.set_layout_engine("tight", rect=(0, 0.03, 1, 1))
    return fig


def fig_genre_share(genres):
    total = genres["sales_m"].sum()
    top = genres.head(7)
    other = total - top["sales_m"].sum()
    labels = list(top.index) + [f"Other ({len(genres) - 7} genres)"]
    values = list(top["sales_m"]) + [other]
    fig, ax = plt.subplots(figsize=(8, 2.6))
    left = 0.0
    shades = ["#1c5aa3", "#2a78d6", "#4f93e0", "#76abe8", "#9cc3ef", "#bdd6f4", "#d8e7f8", NEUTRAL]
    for lab, v, c in zip(labels, values, shades):
        ax.barh([0], [v], left=left, color=c, edgecolor="white", linewidth=2, height=0.5)
        pct = 100 * v / total
        if pct >= 6:
            ax.text(left + v / 2, 0, f"{pct:.0f} %", ha="center", va="center", fontsize=8,
                    color="white" if c in shades[:4] else INK)
        left += v
    ax.set_xlim(0, total)
    ax.set_yticks([])
    ax.set_xticks([total * q for q in (0, 0.25, 0.5, 0.75, 1)])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{100 * x / total:.0f} %"))
    ax.set_title("Share of total sales by genre")
    ax.grid(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    _legend(ax, list(zip(labels, shades)), loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, fontsize=7.5)
    fig.set_layout_engine("tight")
    return fig


def fig_developers(devs):
    mean_dev = devs["sales_m"].mean()
    median_dev = devs["sales_m"].median()
    d = devs.iloc[::-1]
    colors = [BLUE if v > mean_dev else NEUTRAL for v in d["sales_m"]]
    fig, ax = plt.subplots(figsize=(8.5, 6.6))
    bars = ax.barh(d.index, d["sales_m"], color=colors, height=0.7)
    _barh_labels(ax, bars, d["sales_m"], "{:.2f}", pad=1)
    ax.axvline(mean_dev, color=INK, lw=1.2, ls="--")
    ax.axvline(median_dev, color=INK_2, lw=1.2, ls=":")
    ax.set_xlabel("Copies sold (millions)")
    ax.set_title("Total sales by developer")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, d["sales_m"].max() * 1.12)
    ax.set_yticks(range(len(d)), [f"{i} ({n})" for i, n in zip(d.index, d["titles"])])
    _legend(ax, [("Above developer mean", BLUE), ("Below developer mean", NEUTRAL)],
            lines=[(f"Mean {mean_dev:.2f} M", INK, "--"), (f"Median {median_dev:.2f} M", INK_2, ":")],
            loc="lower right")
    fig.text(0.01, 0.01, "Number of titles in parentheses.", fontsize=7.5, color=MUTED)
    fig.set_layout_engine("tight", rect=(0, 0.02, 1, 1))
    return fig


def fig_developer_hist(devs):
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.arange(0, 180, 10)
    counts, edges, patches = ax.hist(devs["sales_m"], bins=bins, color=BLUE, edgecolor="white", linewidth=2)
    for c, e in zip(counts, edges[:-1]):
        if c:
            ax.text(e + 5, c + 0.3, f"{int(c)}", ha="center", fontsize=8, color=INK_2)
    ax.set_xticks(bins[::2])
    ax.set_xlabel("Total copies sold by the developer (millions, 10 M bins)")
    ax.set_ylabel("Number of developers")
    ax.set_title("Distribution of developers by total sales")
    ax.annotate("Nintendo EPD (162 M)", xy=(162, 1.1), xytext=(120, 6), fontsize=8, color=INK_2,
                arrowprops=dict(arrowstyle="->", color=MUTED, shrinkB=6))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_ylim(0, counts.max() * 1.1)
    ax.grid(axis="x", visible=False)
    fig.set_layout_engine("tight")
    return fig


def fig_publishers(pubs):
    p = pubs.iloc[::-1]
    colors = [BLUE if n in ECOSYSTEM_PUBLISHERS else ORANGE for n in p.index]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 4.6), sharey=True)
    b1 = a1.barh(p.index, p["titles"], color=colors, height=0.7)
    _barh_labels(a1, b1, p["titles"], "{:.0f}", pad=0.4)
    a1.set_xlabel("Number of titles")
    a1.set_title("Titles by publisher")
    a1.set_xlim(0, p["titles"].max() * 1.15)
    b2 = a2.barh(p.index, p["sales_m"], color=colors, height=0.7)
    _barh_labels(a2, b2, p["sales_m"], pad=3)
    a2.set_xlabel("Copies sold (millions)")
    a2.set_title("Sales by publisher")
    a2.set_xlim(0, p["sales_m"].max() * 1.18)
    for a in (a1, a2):
        a.grid(axis="y", visible=False)
    _legend(a2, [("First/second party", BLUE), ("Third party", ORANGE)], loc="lower right")
    fig.text(0.01, 0.01, "Regional prefixes (JP:, NA/PAL:) removed; the source keeps only one publisher per game.",
             fontsize=7.5, color=MUTED)
    fig.set_layout_engine("tight", rect=(0, 0.03, 1, 1))
    return fig


def fig_ecosystem(df, eco, summary):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 4.2), gridspec_kw={"width_ratios": [1.2, 1]})
    rows = [("Titles", "title_share_pct"), ("Sales", "sales_share_pct")]
    for y, (name, col) in enumerate(rows):
        f = eco.loc["First/second party", col]
        t = eco.loc["Third party", col]
        a1.barh(y, f, color=BLUE, edgecolor="white", linewidth=2, height=0.55)
        a1.barh(y, t, left=f, color=ORANGE, edgecolor="white", linewidth=2, height=0.55)
        a1.text(f / 2, y, f"{f:.1f} %", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
        if t >= 10:
            a1.text(f + t / 2, y, f"{t:.1f} %", ha="center", va="center", color="white", fontsize=8.5)
        else:
            a1.text(101.5, y, f"{t:.1f} %", ha="left", va="center", color=INK_2, fontsize=8.5, clip_on=False)
    a1.set_yticks([0, 1], [r[0] for r in rows])
    a1.invert_yaxis()
    a1.set_xlim(0, 100)
    a1.xaxis.set_major_formatter(mticker.PercentFormatter())
    a1.set_title("Share of best-sellers")
    a1.grid(axis="y", visible=False)
    _legend(a1, [("First/second party", BLUE), ("Third party", ORANGE)],
            loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)

    rng = np.random.default_rng(7)
    for x, (grp, c) in enumerate((("First/second party", BLUE), ("Third party", ORANGE))):
        v = df.loc[df["ecosystem"] == grp, "sales_m"]
        jitter = rng.uniform(-0.16, 0.16, len(v))
        a2.scatter(x + jitter, v, s=36, color=c, edgecolor="white", linewidth=1, zorder=3)
        med = v.median()
        a2.plot([x - 0.3, x + 0.3], [med, med], color=INK, lw=1.6, zorder=4)
        a2.text(x + 0.33, med, f"median {med:.2f}", va="center", fontsize=7.5, color=INK_2)
    a2.set_yscale("log")
    a2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    a2.set_xticks([0, 1], ["First/second\nparty", "Third\nparty"])
    a2.set_xlim(-0.5, 1.9)
    a2.set_ylabel("Copies sold per title (millions, log scale)")
    a2.set_title("Sales per title")
    a2.grid(axis="x", visible=False)
    mw = summary["mann_whitney"]
    a2.text(0.98, 0.98, f"Mann-Whitney U, one-sided\np = {mw['p_value_one_sided']:.3f}", transform=a2.transAxes,
            va="top", ha="right", fontsize=7.5, color=MUTED)
    fig.set_layout_engine("tight")
    return fig


def fig_title_distribution(df):
    mean = df["sales_m"].mean()
    median = df["sales_m"].median()
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.arange(0, 36, 2)
    n, edges, patches = ax.hist(df["sales_m"], bins=bins, edgecolor="white", linewidth=2, color=NEUTRAL)
    for p, e in zip(patches, edges[:-1]):
        if e + 1 > mean:          # bin centre above the mean
            p.set_facecolor(BLUE)
    ax.axvline(mean, color=INK, lw=1.2, ls="--")
    ax.axvline(median, color=INK_2, lw=1.2, ls=":")
    above = int((df["sales_m"] > mean).sum())
    ax.set_xticks(bins[::2])
    ax.set_xlabel("Copies sold per title (millions, 2 M bins)")
    ax.set_ylabel("Number of titles")
    ax.set_title("Distribution of sales per title")
    ax.grid(axis="x", visible=False)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    _legend(ax, [(f"Above the mean ({above} titles, {100 * above / len(df):.1f} %)", BLUE),
                 (f"Below the mean ({len(df) - above} titles)", NEUTRAL)],
            lines=[(f"Mean {mean:.2f} M", INK, "--"), (f"Median {median:.2f} M", INK_2, ":")],
            loc="upper right")
    fig.set_layout_engine("tight")
    return fig


def fig_pareto(df):
    s = df["sales_m"].to_numpy()
    cum = 100 * np.cumsum(s) / s.sum()
    x = np.arange(1, len(s) + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, cum, color=BLUE, lw=2, marker="o", ms=4, mfc=BLUE, mec="white")
    ax.plot([0, len(s)], [0, 100], color=MUTED, lw=1, ls=":")
    ax.text(len(s) * 0.62, 55, "equal sales per title", color=MUTED, fontsize=8, rotation=20)
    for k in (5, 10, 20):
        ax.annotate(f"Top {k}: {cum[k - 1]:.1f} %", xy=(k, cum[k - 1]), xytext=(k + 2.2, cum[k - 1] - 9),
                    fontsize=8, color=INK_2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set_xlim(0, len(s) + 1)
    ax.set_ylim(0, 102)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_xlabel("Number of titles (ranked by sales)")
    ax.set_ylabel("Cumulative share of total sales")
    ax.set_title("Sales concentration (Pareto curve)")
    fig.set_layout_engine("tight")
    return fig


def fig_time_on_market(df, summary):
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for grp, c in (("First/second party", BLUE), ("Third party", ORANGE)):
        d = df[df["ecosystem"] == grp]
        ax.scatter(d["years_on_market"], d["sales_m"], s=40, color=c, edgecolor="white", linewidth=1,
                   label=grp, zorder=3)
    offsets = {  # title: (dx, dy) in points; negative dx puts the label on the left
        "Mario Kart 8 Deluxe": (-6, 0),
        "Animal Crossing: New Horizons": (6, 0),
        "Super Smash Bros. Ultimate": (-6, 3),
        "The Legend of Zelda: Breath of the Wild": (-4, 13),
        "Pokémon Sword and Shield": (6, 0),
        "Super Mario Odyssey": (-6, -7),
    }
    for _, r in df.head(6).iterrows():
        dx, dy = offsets.get(r["title"], (6, 0))
        ax.annotate(r["title"], (r["years_on_market"], r["sales_m"]), xytext=(dx, dy),
                    textcoords="offset points", fontsize=7, color=INK_2, va="center",
                    ha="right" if dx < 0 else "left")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("Years between release and the sales figure")
    ax.set_ylabel("Copies sold (millions, log scale)")
    ax.set_title("Sales vs time on the market")
    t = summary["time_on_market"]
    ax.set_xlim(-0.1, df["years_on_market"].max() + 0.3)
    ax.set_ylim(0.85, 45)
    ax.legend(loc="lower right", title=f"Spearman rho = {t['spearman_rho']:.2f} (p = {t['p_value']:.3f})",
              title_fontsize=8, alignment="right")
    fig.set_layout_engine("tight")
    return fig


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------
def main() -> None:
    show = "--no-gui" not in sys.argv
    df = load_data()
    genres = group_table(df, "genre")
    families = group_table(df, "genre_family")
    devs = group_table(df, "developer")
    pubs = group_table(df, "publisher")
    eco = group_table(df, "ecosystem")
    summary = evaluate(df, genres, devs, pubs, eco)

    # ---- tables ----------------------------------------------------------
    out = df.copy()
    for c in ("as_of", "release"):
        out[c] = out[c].dt.date
    out.round(3).to_csv(RESULTS / "clean_dataset.csv", index=False)
    genres.to_csv(RESULTS / "sales_by_genre.csv")
    families.to_csv(RESULTS / "sales_by_genre_family.csv")
    devs.to_csv(RESULTS / "sales_by_developer.csv")
    pubs.to_csv(RESULTS / "sales_by_publisher.csv")
    eco.to_csv(RESULTS / "sales_by_ecosystem.csv")
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- figures ---------------------------------------------------------
    figures = {
        "Top titles": (fig_top_titles(df), "01_top_titles"),
        "Genres": (fig_genres(genres, families), "02_genres"),
        "Genre share": (fig_genre_share(genres), "03_genre_share"),
        "Developers": (fig_developers(devs), "04_developers"),
        "Developer histogram": (fig_developer_hist(devs), "05_developer_histogram"),
        "Publishers": (fig_publishers(pubs), "06_publishers"),
        "1st/2nd vs 3rd party": (fig_ecosystem(df, eco, summary), "07_ecosystem"),
        "Sales distribution": (fig_title_distribution(df), "08_title_distribution"),
        "Concentration": (fig_pareto(df), "09_pareto"),
        "Time on market": (fig_time_on_market(df, summary), "10_time_on_market"),
    }
    for fig, fname in figures.values():
        fig.savefig(FIGURES / f"{fname}.png", bbox_inches="tight")

    # ---- console report --------------------------------------------------
    d = summary["dataset"]
    print("=" * 72)
    print("BEST-SELLING NINTENDO SWITCH GAMES - DATA ANALYSIS")
    print("=" * 72)
    print(f"Titles: {d['titles']}   Total sales: {d['total_sales_m']:.2f} M copies")
    print(f"Mean / median per title: {d['mean_sales_per_title_m']:.2f} / {d['median_sales_per_title_m']:.2f} M")
    print(f"Genres: {d['genres']}   Developers: {d['developers']}   Publishers: {d['publishers']}")
    print(f"Top-10 share of sales: {d['top10_share_pct']:.1f} %\n")
    for h in summary["hypotheses"]:
        print(f"{h['id']} [{h['result']}] {h['statement']}")
        print(f"    {h['evidence']}\n")
    print(f"Tables saved in:  {RESULTS}")
    print(f"Figures saved in: {FIGURES}")

    if show:
        try:
            show_gui({k: v[0] for k, v in figures.items()}, summary, df, genres, devs)
        except Exception as exc:  # no display available, tkinter missing, ...
            print(f"\nThe interface could not be opened ({exc}). Figures are in {FIGURES}.")


# ---------------------------------------------------------------------------
# 5. Graphical interface (Tkinter, included with Python)
# ---------------------------------------------------------------------------
def show_gui(figures, summary, df, genres, devs) -> None:
    import tkinter as tk
    from tkinter import ttk

    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

    BG = "#f9f9f7"
    CARD = "#fcfcfb"
    BORDER = "#e1e0d9"
    RESULT_COLOR = {"Supported": "#0f7a52", "Partly supported": "#9a6700", "Rejected": "#c2332f"}

    root = tk.Tk()
    root.title("Nintendo Switch Best-Sellers - Data Analysis")
    root.geometry("1200x820")
    root.minsize(940, 640)
    root.configure(bg=BG)

    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("Treeview", rowheight=22, font=("Segoe UI", 9))
    style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"))
    style.configure("TNotebook.Tab", padding=(12, 6), font=("Segoe UI", 9))

    header = tk.Frame(root, bg=BG)
    header.pack(fill="x", padx=16, pady=(14, 6))
    tk.Label(header, text="Best-Selling Nintendo Switch Games", bg=BG, fg=INK,
             font=("Segoe UI", 16, "bold")).pack(anchor="w")
    tk.Label(header, text="Exploratory data analysis of the 43 titles with more than one million copies sold",
             bg=BG, fg=INK_2, font=("Segoe UI", 10)).pack(anchor="w")

    d = summary["dataset"]
    nep = devs.loc["Nintendo EPD"]
    tiles = [
        ("Total sales", f"{d['total_sales_m']:.1f} M", f"{d['titles']} titles"),
        ("Mean / median per title", f"{d['mean_sales_per_title_m']:.2f} / {d['median_sales_per_title_m']:.2f} M",
         f"skewness {d['skewness']:.2f}"),
        ("Top 10 share", f"{d['top10_share_pct']:.1f} %", "of all copies sold"),
        ("Nintendo EPD", f"{nep['sales_share_pct']:.1f} %", f"{int(nep['titles'])} titles, {nep['sales_m']:.1f} M"),
        ("Top genre", genres.index[0], f"{genres.iloc[0]['sales_m']:.1f} M, {int(genres.iloc[0]['titles'])} titles"),
    ]
    row = tk.Frame(root, bg=BG)
    row.pack(fill="x", padx=16, pady=(4, 10))
    for i, (name, value, note) in enumerate(tiles):
        card = tk.Frame(row, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        card.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 8, 0))
        row.columnconfigure(i, weight=1)
        tk.Label(card, text=name, bg=CARD, fg=INK_2, font=("Segoe UI", 9)).pack(anchor="w", padx=10, pady=(8, 0))
        tk.Label(card, text=value, bg=CARD, fg=INK, font=("Segoe UI", 16, "bold")).pack(anchor="w", padx=10)
        tk.Label(card, text=note, bg=CARD, fg=MUTED, font=("Segoe UI", 8)).pack(anchor="w", padx=10, pady=(0, 8))

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=16, pady=(0, 8))

    def table(parent, columns, rows, widths, height=12):
        frame = tk.Frame(parent, bg=CARD)
        tree = ttk.Treeview(frame, columns=columns, show="headings", height=height)
        for c, w in zip(columns, widths):
            tree.heading(c, text=c)
            tree.column(c, width=w, anchor="e" if w < 110 else "w", stretch=w >= 110)
        for r in rows:
            tree.insert("", "end", values=r)
        sb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        return frame, tree

    # ---- hypotheses tab ----------------------------------------------------
    tab = tk.Frame(notebook, bg=CARD)
    notebook.add(tab, text="Hypotheses")
    canvas = tk.Canvas(tab, bg=CARD, highlightthickness=0)
    sb = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
    inner = tk.Frame(canvas, bg=CARD)
    inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    win = canvas.create_window((0, 0), window=inner, anchor="nw")
    canvas.bind("<Configure>", lambda e: canvas.itemconfigure(win, width=e.width))
    canvas.configure(yscrollcommand=sb.set)
    canvas.pack(side="left", fill="both", expand=True)
    sb.pack(side="right", fill="y")
    inner.columnconfigure(1, weight=1)
    for i, h in enumerate(summary["hypotheses"]):
        tk.Label(inner, text=h["id"], bg=CARD, fg=INK, font=("Segoe UI", 13, "bold")).grid(
            row=2 * i, column=0, sticky="nw", padx=(14, 10), pady=(14, 0))
        box = tk.Frame(inner, bg=CARD)
        box.grid(row=2 * i, column=1, sticky="ew", padx=(0, 14), pady=(14, 0))
        top = tk.Frame(box, bg=CARD)
        top.pack(fill="x")
        tk.Label(top, text=h["statement"], bg=CARD, fg=INK, font=("Segoe UI", 10, "bold"),
                 wraplength=760, justify="left").pack(side="left", anchor="w")
        tk.Label(top, text=f"  {h['result']}  ", bg=RESULT_COLOR[h["result"]], fg="white",
                 font=("Segoe UI", 9, "bold")).pack(side="right", anchor="n")
        ev = tk.Label(box, text=h["evidence"], bg=CARD, fg=INK_2, font=("Segoe UI", 9),
                      wraplength=900, justify="left")
        ev.pack(anchor="w", pady=(4, 0))
        box.bind("<Configure>", lambda e, lbl=ev: lbl.configure(wraplength=max(300, e.width - 10)))
        ttk.Separator(inner, orient="horizontal").grid(row=2 * i + 1, column=0, columnspan=2,
                                                        sticky="ew", padx=14, pady=(12, 0))

    # ---- tables tab ---------------------------------------------------------
    tab2 = tk.Frame(notebook, bg=CARD)
    notebook.add(tab2, text="Genres & developers")
    tab2.columnconfigure(0, weight=1)
    tab2.columnconfigure(1, weight=1)
    tab2.rowconfigure(1, weight=1)
    for col, (ttl, tbl) in enumerate((("Sales by genre", genres), ("Sales by developer", devs))):
        tk.Label(tab2, text=ttl, bg=CARD, fg=INK, font=("Segoe UI", 11, "bold")).grid(
            row=0, column=col, sticky="w", padx=10, pady=(10, 4))
        f, _ = table(tab2, ("Name", "Titles", "Sales (M)", "Share %", "Mean (M)"),
                     [(n, int(r.titles), f"{r.sales_m:.2f}", f"{r.sales_share_pct:.1f}", f"{r.mean_m:.2f}")
                      for n, r in tbl.iterrows()], (190, 60, 80, 70, 80))
        f.grid(row=1, column=col, sticky="nsew", padx=10, pady=(0, 10))

    # ---- dataset tab (click a header to sort) --------------------------------
    tab3 = tk.Frame(notebook, bg=CARD)
    notebook.add(tab3, text="Dataset")
    cols = ("#", "Title", "Sales (M)", "Genre", "Developer", "Publisher", "Released", "Group")
    rows = [(i + 1, r.title, f"{r.sales_m:.2f}", r.genre, r.developer, r.publisher,
             r.release.date().isoformat(), r.ecosystem) for i, r in enumerate(df.itertuples())]
    f, tree = table(tab3, cols, rows, (40, 280, 80, 150, 170, 170, 90, 130), height=20)
    f.pack(fill="both", expand=True, padx=10, pady=10)

    def sort_by(col, reverse=False):
        items = [(tree.set(k, col), k) for k in tree.get_children("")]
        try:
            items.sort(key=lambda t: float(t[0]), reverse=reverse)
        except ValueError:
            items.sort(key=lambda t: t[0].lower(), reverse=reverse)
        for idx, (_, k) in enumerate(items):
            tree.move(k, "", idx)
        tree.heading(col, command=lambda: sort_by(col, not reverse))

    for c in cols:
        tree.heading(c, command=lambda c=c: sort_by(c))

    # ---- figures tab: list on the left, interactive plot on the right ------
    tab4 = tk.Frame(notebook, bg=CARD)
    notebook.add(tab4, text="Figures")
    notebook.insert(0, tab4)
    side = tk.Frame(tab4, bg=CARD)
    side.pack(side="left", fill="y", padx=(10, 0), pady=10)
    tk.Label(side, text="Figures", bg=CARD, fg=INK, font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 6))
    names = list(figures)
    listbox = tk.Listbox(side, activestyle="none", exportselection=False, font=("Segoe UI", 10),
                         width=22, height=len(names), bd=0, highlightthickness=1,
                         highlightbackground=BORDER, selectbackground=BLUE, selectforeground="white",
                         bg="white", fg=INK)
    for i, n in enumerate(names, 1):
        listbox.insert("end", f" {i:>2}. {n}")
    listbox.pack(anchor="nw")
    tk.Label(side, text="Use the toolbar below each plot\nto zoom, pan or save it.", bg=CARD, fg=MUTED,
             font=("Segoe UI", 8), justify="left").pack(anchor="w", pady=(10, 0))
    area = tk.Frame(tab4, bg="white", highlightbackground=BORDER, highlightthickness=1)
    area.pack(side="left", fill="both", expand=True, padx=10, pady=10)

    panels: dict[str, tk.Frame] = {}

    def show_figure(name):
        for p in panels.values():
            p.pack_forget()
        if name not in panels:
            frame = tk.Frame(area, bg="white")
            fig = figures[name]
            fig.set_dpi(100)
            cv = FigureCanvasTkAgg(fig, master=frame)
            tb = NavigationToolbar2Tk(cv, frame, pack_toolbar=False)
            tb.update()
            tb.pack(side="bottom", fill="x")
            cv.get_tk_widget().pack(side="top", fill="both", expand=True)
            cv.draw()
            panels[name] = frame
        panels[name].pack(fill="both", expand=True)

    def on_select(_event=None):
        sel = listbox.curselection()
        if sel:
            show_figure(names[sel[0]])

    listbox.bind("<<ListboxSelect>>", on_select)
    listbox.selection_set(0)
    show_figure(names[0])
    notebook.select(0)

    footer = tk.Frame(root, bg=BG)
    footer.pack(fill="x", padx=16, pady=(0, 12))
    tk.Label(footer, bg=BG, fg=INK_2, font=("Segoe UI", 9),
             text=f"Source: Kaggle / Wikipedia, sales figures dated {d['as_of_range'][0]} to {d['as_of_range'][1]}"
             ).pack(side="left")

    def open_folder(path):
        if sys.platform.startswith("win"):
            os.startfile(path)  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(path)])

    ttk.Button(footer, text="Close", command=root.destroy).pack(side="right")
    ttk.Button(footer, text="Open figures folder", command=lambda: open_folder(FIGURES)).pack(side="right", padx=6)
    ttk.Button(footer, text="Open results folder", command=lambda: open_folder(RESULTS)).pack(side="right")

    root.protocol("WM_DELETE_WINDOW", lambda: (root.quit(), root.destroy()))
    print("\nInterface open - close the window to finish.")
    root.mainloop()


if __name__ == "__main__":
    main()
