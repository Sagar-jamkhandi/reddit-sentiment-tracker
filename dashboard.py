"""
Streamlit dashboard for the real-time sentiment tracker.

Fully self-contained: fetches live data itself via the sidebar button
(no need to run sentiment_tracker.py separately first). Defaults to the
Hacker News API, which requires no authentication and works reliably on
any host, including Streamlit Cloud. Reddit is available as an optional
source if you've set up official API credentials (see README).

Usage:
    streamlit run dashboard.py
"""

import glob
import os

import pandas as pd
import plotly.express as px
import streamlit as st

from sentiment_tracker import analyze_subreddit

st.set_page_config(page_title="Real-Time Sentiment Tracker", page_icon="💬", layout="wide")

st.title("💬 Real-Time Sentiment Tracker")
st.caption("Sentiment analysis (VADER) on live posts, sourced from a public API in real time.")

OUTPUT_DIR = "output"


@st.cache_data
def load_latest_posts() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "posts_*.csv")))
    if not files:
        return pd.DataFrame()
    return pd.read_csv(files[-1], parse_dates=["created_dt"])


def get_reddit_credentials():
    """Read Reddit API credentials from Streamlit secrets, if configured."""
    try:
        return st.secrets["REDDIT_CLIENT_ID"], st.secrets["REDDIT_CLIENT_SECRET"]
    except Exception:
        return None, None


def fetch_and_save(source: str, labels: list[str], limit: int) -> pd.DataFrame:
    """Fetch fresh posts for the given source/labels and cache to CSV."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    client_id, client_secret = get_reddit_credentials()
    use_praw = source == "reddit" and bool(client_id and client_secret)

    dfs = []
    progress = st.progress(0.0, text="Starting fetch...")
    for i, label in enumerate(labels):
        progress.progress(i / len(labels), text=f"Fetching {label} ...")
        try:
            dfs.append(
                analyze_subreddit(
                    label, limit=limit, use_praw=use_praw,
                    client_id=client_id, client_secret=client_secret, source=source,
                )
            )
        except Exception as e:
            st.error(f"Failed to fetch {label}: {e}")
    progress.progress(1.0, text="Done.")
    if not dfs:
        return pd.DataFrame()
    combined = pd.concat(dfs, ignore_index=True)
    timestamp = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    combined.to_csv(os.path.join(OUTPUT_DIR, f"posts_{timestamp}.csv"), index=False)
    return combined


# --- Sidebar: fetch controls --------------------------------------------
_client_id, _client_secret = get_reddit_credentials()
_has_reddit_creds = bool(_client_id and _client_secret)

with st.sidebar:
    st.header("Fetch Live Data")
    source = st.radio(
        "Data source",
        options=["hackernews", "reddit"],
        format_func=lambda s: "Hacker News (no setup needed)" if s == "hackernews" else "Reddit (needs API credentials)",
    )

    if source == "hackernews":
        st.caption("✅ No auth needed — pulls current top stories from Hacker News.")
        label_input = "top"
        fetch_limit = st.slider("Number of stories", 10, 100, 50)
    else:
        if _has_reddit_creds:
            st.caption("✅ Using official Reddit API (credentials found)")
        else:
            st.caption(
                "⚠️ No Reddit API credentials found in Secrets. The public "
                "endpoint fallback is blocked by Reddit on most cloud hosts. "
                "See README to add REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET."
            )
        label_input = st.text_input("Subreddits (comma-separated)", value="technology, datascience")
        fetch_limit = st.slider("Posts per subreddit", 10, 100, 50)

    fetch_clicked = st.button("🔄 Fetch Live Data", type="primary", use_container_width=True)

if fetch_clicked:
    labels = [s.strip() for s in label_input.split(",") if s.strip()]
    with st.spinner("Pulling live posts and scoring sentiment..."):
        df = fetch_and_save(source, labels, fetch_limit)
    st.cache_data.clear()
    if df.empty and source == "reddit" and not _has_reddit_creds:
        st.error(
            "Fetched 0 posts — Reddit likely blocked the request (common on cloud "
            "hosts). Add REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET in Secrets, or "
            "switch the source to Hacker News above."
        )
    else:
        st.success(f"Fetched {len(df)} posts.")
else:
    df = load_latest_posts()

if df.empty:
    st.info(
        "👈 Click **Fetch Live Data** in the sidebar to pull live posts and "
        "run sentiment analysis. First fetch takes a few seconds."
    )
    st.stop()

# --- Sidebar filters ------------------------------------------------
with st.sidebar:
    st.divider()
    st.header("Filters")
    subs = st.multiselect(
        "Source / subreddit", options=sorted(df["subreddit"].unique()), default=list(df["subreddit"].unique())
    )

filtered = df[df["subreddit"].isin(subs)]

# --- KPIs -------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Posts", len(filtered))
col2.metric("Avg. Sentiment (compound)", f"{filtered['compound'].mean():.3f}")
col3.metric("% Positive", f"{(filtered['sentiment_label'] == 'positive').mean() * 100:.1f}%")
col4.metric("% Negative", f"{(filtered['sentiment_label'] == 'negative').mean() * 100:.1f}%")

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Sentiment Distribution")
    dist = filtered.groupby(["subreddit", "sentiment_label"]).size().reset_index(name="count")
    fig = px.bar(
        dist, x="subreddit", y="count", color="sentiment_label", barmode="group",
        color_discrete_map={"positive": "green", "neutral": "gray", "negative": "red"},
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Sentiment vs. Score")
    ts = filtered.sort_values("created_dt")
    fig2 = px.scatter(
        ts, x="created_dt", y="compound", color="subreddit", size="score",
        hover_data=["title"], labels={"compound": "Sentiment (compound)", "created_dt": "Posted at"},
    )
    fig2.add_hline(y=0, line_dash="dot", line_color="gray")
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("Most Positive & Most Negative Posts")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**🙂 Most Positive**")
    st.dataframe(filtered.nlargest(5, "compound")[["subreddit", "title", "compound"]], use_container_width=True, hide_index=True)
with c2:
    st.markdown("**🙁 Most Negative**")
    st.dataframe(filtered.nsmallest(5, "compound")[["subreddit", "title", "compound"]], use_container_width=True, hide_index=True)

st.subheader("Raw Data")
st.dataframe(filtered, use_container_width=True)
