# GamePulse Analytics — Community Health Dashboard

MSBA (Quantic) capstone. An interactive Streamlit dashboard that scores the Telegram
chats of **GIG (Gaming Italian Group)**, an Italian gaming community network, on two
pillars:

- **Engagement** — how much and how regularly members take part (messages per active
  user, active-member ratio, returning users, stickiness, lifetime, bot usage).
- **Structural quality** — the *shape* of the conversation, not its content (reply rate,
  mention density, message length, sticker ratio, participation inequality, ban rate,
  newcomer rejection).

The dashboard ships with a **frozen, pseudonymised snapshot** of the network's activity
telemetry, so it runs fully offline and always shows the same data.

## Quick start

You need **Python 3.11 or newer** (3.12 is what was tested).

### Windows

Double-click **`launch_windows.bat`**. The first launch creates a local `.venv` and
installs the dependencies (a few minutes); later launches open straight away. The
dashboard opens in your browser at <http://localhost:8501>.

### macOS / Linux / any terminal

```bash
git clone https://github.com/Morelato17/MSBAQuanticCapstone_GamePulseAnalytics.git
cd MSBAQuanticCapstone_GamePulseAnalytics
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/dashboard_app.py
```

Running it locally for evaluation is the only use the [license](#license) allows; please
do not redeploy it as a public service.

## Using the dashboard

The sidebar sets the **rolling score window** (7, 28 or 90 days), the **"ranking as of"**
date and the **evolution range**. The five tabs:

| Tab | What it shows |
|-----|---------------|
| Overview | Engagement × quality map (bubble size = active users) and the full chat ranking with period-over-period change. |
| Evolution | Weekly-sampled score history for up to five chats side by side. |
| Attention queue | Rule-based flags (e.g. *busy but unhealthy*, *rapid quality deterioration*, *retention risk*) ranked by severity and audience size. |
| Chat detail | Scores, raw metric evidence and most active user IDs for one chat. |
| Top contributors | User leaderboard for the whole network or one chat, by messages, active days, replies, mentions or cross-chat presence. |

Chats below the evidence floor (too few messages or active users in the window) are shown
as **unrated**, not as bad.

## The data

`data/` holds the snapshot as Parquet files, queried in-process with
[DuckDB](https://duckdb.org) — no database server needed.

| File | Grain | Columns |
|------|-------|---------|
| `user_activity.parquet` | user × chat × day × message type | `user_id`, `chat_id`, `activity_date`, `message_type`, `message_count`, `word_count`, `reply_count`, `mention_count` |
| `membership_history.parquet` | one membership event | `user_id`, `chat_id`, `event_type` (join / leave / ban / restrict), `event_date` (UTC) |
| `chats.parquet` | one chat | `id`, `telegram_id`, `title`, `member_count`, `status`, `type`, `staff_chat_id` |
| `manifest.json` | — | freeze date, period covered, row counts |

**Period covered:** 2026-05-23 → 2026-10-09 (frozen on 2026-10-10).
The snapshot is static by design: the dashboard cannot refresh or regenerate it.

### Privacy

The data comes from a live community, so it was pseudonymised before publication:

- every chat id and user id was replaced by a synthetic integer, drawn from a random
  permutation that was discarded after the export — the mapping cannot be recovered;
- chat titles were replaced by `Chat NNN`;
- only aggregate activity counts are included — no message text, names, usernames or
  profile data were ever collected into this dataset.

## Repository layout

```
app/
  dashboard_app.py     Streamlit UI
  scoring_model.py     metric extraction, scorers, shrinkage, weights, frozen anchors
data/                  frozen pseudonymised snapshot (Parquet) + manifest
launch_windows.bat     double-click launcher for Windows
requirements.txt       pinned dependencies
```

## Limitations

- **Structural, not content quality.** Scores describe how people interact, never what
  they say.
- **Short history.** The snapshot starts on 2026-05-23, so user lifetimes are truncated
  and early windows have less history behind them.
- **Fixed anchors.** Metric normalisation uses anchors frozen from the network
  distribution (`frozen_anchors()` in `scoring_model.py`); they are not re-derived from
  the period you select.

## License

**All rights reserved — evaluation only.** You may view, download and run this project
locally to evaluate or review the capstone. Redistribution, modification, commercial
use, any other use of the data and any attempt to re-identify people or chats are not
permitted. See [`LICENSE`](LICENSE) for the full terms.
