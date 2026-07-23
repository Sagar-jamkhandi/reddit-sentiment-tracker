# 💬 Real-Time Sentiment Tracker

Pulls live posts from a public API, scores sentiment using **VADER**
(a lexicon tuned for short, informal text), and visualizes sentiment
trends through an interactive dashboard.

Built to demonstrate: working with a real-time API, NLP-based sentiment
scoring, data aggregation, and interactive visualization.

## ✨ Data Sources
- **Hacker News (default)** — no authentication required, works reliably
  everywhere including cloud hosts like Streamlit Cloud. Pulls current
  top stories via HN's public Firebase API.
- **Reddit (optional)** — requires official Reddit API credentials. Reddit
  recently tightened API access behind a "Responsible Builder Policy" and
  blocks its old no-auth public endpoint from most cloud-server IPs, so
  this only works reliably with real credentials set up (see below).

The dashboard has a source toggle in the sidebar — pick whichever works
for you. Hacker News needs zero setup and is the recommended default.

## ✨ Features
- Fetches live posts from the public API of your choice
- Scores every post with VADER sentiment (compound, positive/negative/neutral)
- Self-contained dashboard — click "Fetch Live Data," no separate script run needed
- Sentiment distribution chart, sentiment-vs-score scatter, top positive/negative posts
- Also usable as a standalone CLI script for scheduled/bulk collection

## 🛠️ Tech Stack
| Layer | Tool |
|---|---|
| Language | Python 3.10+ |
| Data source | [Hacker News API](https://github.com/HackerNews/API) (no key) or Reddit API |
| Sentiment analysis | [VADER](https://github.com/cjhutto/vaderSentiment) |
| Data handling | pandas |
| Dashboard | Streamlit + Plotly |

## 📦 Project Structure
```
project2-reddit-sentiment-tracker/
├── sentiment_tracker.py   # Fetches posts + runs sentiment analysis -> CSV
├── dashboard.py           # Self-contained Streamlit dashboard
├── requirements.txt
└── README.md
```

## ⚙️ Setup
```bash
git clone https://github.com/<your-username>/sentiment-tracker.git
cd sentiment-tracker
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Run the dashboard (recommended path — zero setup)
```bash
streamlit run dashboard.py
```
In the sidebar, leave the source as **Hacker News**, click **🔄 Fetch Live
Data**, and you're done. This is exactly what you'd deploy to Streamlit
Cloud — no secrets or credentials needed.

### Optional: use Reddit instead
Reddit now requires official API credentials for reliable access (see
[Reddit's Responsible Builder Policy](https://support.reddithelp.com/hc/en-us/articles/42728983564564-Responsible-Builder-Policy)).
If you've obtained credentials at https://www.reddit.com/prefs/apps:

1. **Local run:**
   ```bash
   export REDDIT_CLIENT_ID=your_client_id
   export REDDIT_CLIENT_SECRET=your_client_secret
   python sentiment_tracker.py --source reddit --subreddits technology --use-praw
   ```
2. **Streamlit Cloud:** go to your app → **Settings → Secrets**, paste:
   ```toml
   REDDIT_CLIENT_ID = "your_client_id"
   REDDIT_CLIENT_SECRET = "your_client_secret"
   ```
   Save and reboot. Then pick "Reddit" as the source in the dashboard sidebar.

### Standalone CLI (for scheduled/bulk collection)
```bash
python sentiment_tracker.py --source hackernews --limit 100
```
Writes `output/posts_<timestamp>.csv`, which the dashboard also reads.

## 📊 What This Project Demonstrates (for your resume/interview)
- Consuming a real-time public REST API and handling pagination/rate limits
- Applying NLP sentiment scoring to unstructured text at scale
- Data aggregation with pandas `groupby`
- Building a self-contained, deployable interactive dashboard
- Adapting a project when an external dependency changes (Reddit's policy
  shift) — a realistic engineering scenario, not just following a tutorial

## 🔮 Possible Extensions
- Schedule the script hourly (cron / GitHub Actions) to build a historical
  sentiment-trend dataset over weeks/months
- Add Hacker News comment-level sentiment, not just story titles
- Swap VADER for a transformer-based sentiment model and compare results

## 📝 Resume Bullet (copy/adapt this)
> Built a real-time sentiment tracking dashboard in Python, ingesting live
> data from a public API, applying NLP sentiment scoring (VADER) across
> hundreds of posts, and surfacing trends through an interactive Streamlit
> dashboard with a pluggable data-source architecture.

## 📄 License
MIT — free to use and adapt.
