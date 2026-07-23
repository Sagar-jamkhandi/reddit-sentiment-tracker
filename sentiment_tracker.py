"""
Real-Time Sentiment Tracker
------------------------------
Pulls live posts/comments and runs sentiment analysis (VADER, tuned for
short/informal text), producing a time-trend sentiment report with
visualizations.

Two data sources are supported:
  1. Hacker News (default, recommended) - Firebase-backed public API, no
     key required, works reliably from any server including cloud hosts.
  2. Reddit - requires official API credentials (PRAW). Reddit's public
     JSON endpoint blocks cloud-server IPs, so this needs REDDIT_CLIENT_ID
     / REDDIT_CLIENT_SECRET to work when deployed. See README for setup,
     including Reddit's newer "Responsible Builder Policy" requirements.

Author: Sagar-jamkhandi
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

import pandas as pd
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

USER_AGENT = "sentiment-tracker-script/1.0 (by u/your_username)"

HN_BASE = "https://hacker-news.firebaseio.com/v0"
HN_TOPSTORIES_URL = f"{HN_BASE}/topstories.json"
HN_ITEM_URL = f"{HN_BASE}/item/{{item_id}}.json"


@dataclass
class Post:
    id: str
    subreddit: str  # kept as "subreddit" for schema compatibility; holds source/category label
    title: str
    selftext: str
    score: int
    num_comments: int
    created_utc: float
    compound: float
    sentiment_label: str


def classify(compound: float) -> str:
    """VADER's standard thresholding for compound score -> label."""
    if compound >= 0.05:
        return "positive"
    elif compound <= -0.05:
        return "negative"
    return "neutral"


def fetch_posts_hackernews(category: str = "top", limit: int = 50) -> list[dict]:
    """
    Fetch live stories from Hacker News's public Firebase API.
    No authentication required; works from any IP, including cloud hosts.
    `category` is a free-form label (e.g. "top") used only for grouping in
    the output - HN doesn't have subreddit-style channels, so we tag every
    story with this label.
    """
    ids_resp = requests.get(HN_TOPSTORIES_URL, timeout=10)
    ids_resp.raise_for_status()
    story_ids = ids_resp.json()[:limit]

    posts = []
    for story_id in story_ids:
        item_resp = requests.get(HN_ITEM_URL.format(item_id=story_id), timeout=10)
        if item_resp.status_code != 200 or not item_resp.json():
            continue
        item = item_resp.json()
        posts.append(
            {
                "id": str(item.get("id", story_id)),
                "title": item.get("title", ""),
                "selftext": item.get("text", "") or "",
                "score": item.get("score", 0),
                "num_comments": item.get("descendants", 0),
                "created_utc": item.get("time", 0),
            }
        )
    return posts


def fetch_posts_public_json(subreddit: str, limit: int = 100) -> list[dict]:
    """
    Reddit fallback fetch using the public read-only JSON endpoint.
    No API key needed, but Reddit blocks this from most cloud-server IPs
    (works locally, commonly fails when deployed).
    """
    url = f"https://www.reddit.com/r/{subreddit}/new.json"
    headers = {"User-Agent": USER_AGENT}
    params = {"limit": min(limit, 100)}
    response = requests.get(url, headers=headers, params=params, timeout=10)
    response.raise_for_status()
    children = response.json()["data"]["children"]
    return [c["data"] for c in children]


def fetch_posts_praw(
    subreddit: str,
    limit: int = 100,
    client_id: str | None = None,
    client_secret: str | None = None,
) -> list[dict]:
    """
    Preferred fetch using PRAW + official Reddit API credentials.
    Pass client_id/client_secret directly, or set these environment
    variables before running:
        REDDIT_CLIENT_ID
        REDDIT_CLIENT_SECRET
    """
    import praw

    reddit = praw.Reddit(
        client_id=client_id or os.environ["REDDIT_CLIENT_ID"],
        client_secret=client_secret or os.environ["REDDIT_CLIENT_SECRET"],
        user_agent=USER_AGENT,
    )
    posts = []
    for submission in reddit.subreddit(subreddit).new(limit=limit):
        posts.append(
            {
                "id": submission.id,
                "title": submission.title,
                "selftext": submission.selftext,
                "score": submission.score,
                "num_comments": submission.num_comments,
                "created_utc": submission.created_utc,
            }
        )
    return posts


def analyze_subreddit(
    subreddit: str,
    limit: int = 100,
    use_praw: bool = False,
    client_id: str | None = None,
    client_secret: str | None = None,
    source: str = "hackernews",
) -> pd.DataFrame:
    """
    Fetch posts and attach sentiment scores; returns a tidy DataFrame.
    `source` is "hackernews" (default, no auth needed) or "reddit".
    For "reddit", `subreddit` is the subreddit name; for "hackernews" it's
    just used as a display label.
    """
    if source == "hackernews":
        raw_posts = fetch_posts_hackernews(category=subreddit, limit=limit)
    elif use_praw:
        raw_posts = fetch_posts_praw(subreddit, limit=limit, client_id=client_id, client_secret=client_secret)
    else:
        raw_posts = fetch_posts_public_json(subreddit, limit=limit)

    records: list[Post] = []
    for p in raw_posts:
        text = f"{p.get('title', '')} {p.get('selftext', '')}".strip()
        scores = analyzer.polarity_scores(text)
        records.append(
            Post(
                id=p.get("id", ""),
                subreddit=subreddit,
                title=p.get("title", ""),
                selftext=(p.get("selftext") or "")[:200],
                score=p.get("score", 0),
                num_comments=p.get("num_comments", 0),
                created_utc=p.get("created_utc", 0),
                compound=scores["compound"],
                sentiment_label=classify(scores["compound"]),
            )
        )

    df = pd.DataFrame([asdict(r) for r in records])
    if not df.empty:
        df["created_dt"] = pd.to_datetime(df["created_utc"], unit="s", utc=True)
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sentiment counts and average compound score per subreddit."""
    if df.empty:
        return pd.DataFrame()
    return (
        df.groupby("subreddit")
        .agg(
            posts=("id", "count"),
            avg_compound=("compound", "mean"),
            pct_positive=("sentiment_label", lambda s: (s == "positive").mean() * 100),
            pct_negative=("sentiment_label", lambda s: (s == "negative").mean() * 100),
            pct_neutral=("sentiment_label", lambda s: (s == "neutral").mean() * 100),
        )
        .round(2)
        .reset_index()
    )


def main():
    parser = argparse.ArgumentParser(description="Track live sentiment across sources.")
    parser.add_argument(
        "--source",
        choices=["hackernews", "reddit"],
        default="hackernews",
        help="Data source: 'hackernews' (default, no auth needed) or 'reddit' (needs API credentials).",
    )
    parser.add_argument(
        "--subreddits",
        nargs="+",
        default=["top"],
        help="For hackernews: label(s) for the batch (e.g. 'top'). "
        "For reddit: subreddit names (no r/ prefix).",
    )
    parser.add_argument("--limit", type=int, default=50, help="Posts to fetch per label/subreddit.")
    parser.add_argument("--use-praw", action="store_true", help="Use official Reddit API via PRAW (reddit source only).")
    parser.add_argument("--out", default="output", help="Output directory for CSV/report.")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    all_dfs = []
    for sub in args.subreddits:
        print(f"Fetching ({args.source}: {sub}) ...")
        try:
            df = analyze_subreddit(sub, limit=args.limit, use_praw=args.use_praw, source=args.source)
            all_dfs.append(df)
            time.sleep(1)  # be polite to the API
        except Exception as e:
            print(f"  Failed to fetch {sub}: {e}")

    if not all_dfs:
        print("No data fetched. Exiting.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    posts_path = os.path.join(args.out, f"posts_{timestamp}.csv")
    combined.to_csv(posts_path, index=False)
    print(f"Saved {len(combined)} posts -> {posts_path}")

    summary = summarize(combined)
    summary_path = os.path.join(args.out, f"summary_{timestamp}.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary -> {summary_path}")
    print("\n=== Sentiment Summary ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
