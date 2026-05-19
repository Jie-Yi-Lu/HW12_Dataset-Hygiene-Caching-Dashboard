import time
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = "./data/messy_stroop_homework.csv"

# ── Cached helpers ──────────────────────────────────────────────────────────

@st.cache_data(ttl=600, show_spinner="Loading data…")
# ttl=600: re-read the CSV at most once per 10 minutes so a TA refreshing the
# page doesn't pay I/O on every interaction, but stale data never stays cached
# beyond a single lab session.
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
# clean() is a pure, deterministic function of its input; caching it means the
# expensive string-parsing / type-coercion runs exactly once per unique raw df.
def clean(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    # 1. rt_ms: coerce string sentinels ("missing", "--") → NaN, then cast float
    df["rt_ms"] = pd.to_numeric(df["rt_ms"], errors="coerce")

    # 2. rt_ms: replace numeric sentinels (-1, 9999) → NaN
    df["rt_ms"] = df["rt_ms"].replace({-1: np.nan, 9999: np.nan})

    # 3. rt_ms: keep only physiologically plausible range 200–2000 ms
    df = df[df["rt_ms"].isna() | df["rt_ms"].between(200, 2000)]

    # 4. Drop rows where rt_ms is still NaN (sentinels + range outliers gone)
    df = df.dropna(subset=["rt_ms"])

    # 5. condition: strip whitespace, lower-case, expand abbreviations
    df["condition"] = df["condition"].str.strip().str.lower()
    df["condition"] = df["condition"].replace({"con": "congruent", "incong.": "incongruent"})
    df["condition"] = df["condition"].astype("category")

    # 6. accuracy: map "True"→1, "False"→0, then cast int
    df["accuracy"] = df["accuracy"].replace({"True": 1, "False": 0})
    df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce").astype("Int64")

    # 7. age: replace sentinels (-1, 888) → NaN
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["age"] = df["age"].replace({-1: np.nan, 888: np.nan})

    # 8. Remove exact duplicate rows (subject-merging error)
    df = df.drop_duplicates()

    return df.reset_index(drop=True)


def analyse(df: pd.DataFrame, *, outlier_sd: float = 3.0) -> dict:
    """Compute Stroop-effect statistics with optional SD-based outlier trimming.

    outlier_sd lives here (not in clean) because it is an analysis decision,
    not a data-quality decision.  Different research questions may use different
    thresholds on the same cleaned dataset.
    """
    result = {}
    for cond in ["congruent", "incongruent"]:
        sub = df[df["condition"] == cond]["rt_ms"]
        if outlier_sd:
            mu, sigma = sub.mean(), sub.std()
            sub = sub[(sub - mu).abs() <= outlier_sd * sigma]
        result[cond] = {"mean": sub.mean(), "sd": sub.std(), "n": len(sub)}
    result["stroop_effect"] = result["incongruent"]["mean"] - result["congruent"]["mean"]
    return result


# ── Layout ───────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Stroop Dashboard", layout="wide")
st.title("Stroop Effect Dashboard")

# ── Load & clean (with cache timing) ────────────────────────────────────────

t0 = time.perf_counter()
raw = load_data(DATA_PATH)
load_ms = (time.perf_counter() - t0) * 1000

t1 = time.perf_counter()
df_clean = clean(raw)
clean_ms = (time.perf_counter() - t1) * 1000

# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.header("Filters")

all_subjects = sorted(df_clean["subject_id"].dropna().unique().tolist())
selected_subjects = st.sidebar.multiselect(
    "Subject ID", options=all_subjects, default=all_subjects
)

rt_min_val = float(df_clean["rt_ms"].min())
rt_max_val = float(df_clean["rt_ms"].max())
rt_range = st.sidebar.slider(
    "RT range (ms)",
    min_value=rt_min_val,
    max_value=rt_max_val,
    value=(rt_min_val, rt_max_val),
    step=10.0,
)

outlier_sd = st.sidebar.slider("Outlier SD threshold (analyse)", 1.0, 5.0, 3.0, 0.5)

if st.sidebar.button("Clear cache"):
    st.cache_data.clear()
    st.rerun()

# ── Boolean mask ─────────────────────────────────────────────────────────────

mask = (
    df_clean["subject_id"].isin(selected_subjects)
    & df_clean["rt_ms"].between(rt_range[0], rt_range[1])
)
df_view = df_clean[mask]

# ── Cache timing display ──────────────────────────────────────────────────────

with st.expander("Cache timing (demo)", expanded=False):
    col1, col2 = st.columns(2)
    col1.metric("load_data() time (ms)", f"{load_ms:.1f}")
    col2.metric("clean() time (ms)", f"{clean_ms:.1f}")
    st.caption(
        "First run: full I/O + computation. Subsequent runs: near-zero (cache hit). "
        "Click 'Clear cache' in the sidebar to reset."
    )

# ── KPI metrics ───────────────────────────────────────────────────────────────

stats = analyse(df_view, outlier_sd=outlier_sd)

st.subheader("KPI Metrics")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Trials (filtered)", len(df_view))
k2.metric("Mean RT — Congruent (ms)", f"{stats['congruent']['mean']:.1f}")
k3.metric("Mean RT — Incongruent (ms)", f"{stats['incongruent']['mean']:.1f}")
k4.metric("Stroop Effect (ms)", f"{stats['stroop_effect']:.1f}")

# ── Chart: Mean RT by condition × subject ─────────────────────────────────────

st.subheader("Mean RT by Condition × Subject")

if not df_view.empty:
    grouped = (
        df_view.groupby(["subject_id", "condition"])["rt_ms"]
        .mean()
        .reset_index()
        .pivot(index="subject_id", columns="condition", values="rt_ms")
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    grouped.plot(kind="bar", ax=ax)
    ax.set_xlabel("Subject ID")
    ax.set_ylabel("Mean RT (ms)")
    ax.set_title("Mean RT by Subject and Condition")
    ax.legend(title="Condition")
    plt.xticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
else:
    st.info("No data matches the current filter selection.")

# ── Data table ────────────────────────────────────────────────────────────────

st.subheader("Cleaned Data (filtered, 50 data point only)")
st.dataframe(df_view.head(50), use_container_width=True)
