"""
GIG — Engagement & Quality scoring model (reference implementation).

Pure, dependency-light implementation of the chat scoring design. No notebook
or plotting dependencies here on purpose.

Runs against the frozen, pseudonymised Parquet snapshot in `../data/`, queried
in-process through DuckDB.

Design points this file implements:
  * masked + renormalised weights   — a metric that is NULL for a chat drops out
                                      and the remaining weights renormalise
  * anchored-absolute normalisation — anchors derived from the network
                                      distribution once, then frozen
  * three scorer shapes             — higher_better / lower_better / band
  * per-metric shrinkage            — toward the network median, by sample size
  * eligibility floor               — under-sampled chats are UNRATED, not bad

Nothing here is calibrated. Anchors are DERIVED from whatever snapshot you point
it at (`derive_anchors`) and must be reviewed before being frozen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

import duckdb
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Connection
# --------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TABLES = ("user_activity", "membership_history", "chats")


def connect(data_dir: Path = DATA_DIR) -> duckdb.DuckDBPyConnection:
    """Open an in-memory DuckDB over the frozen Parquet snapshot.

    Each table is exposed as a view with the same name and columns as the
    original Postgres table, so the SQL below is unchanged apart from its
    parameter placeholders.
    """
    missing = [t for t in TABLES if not (data_dir / f"{t}.parquet").is_file()]
    if missing:
        raise FileNotFoundError(f"Frozen snapshot files missing in {data_dir}: {', '.join(missing)}")

    conn = duckdb.connect()
    # Date-vs-timestamp comparisons must cast in UTC, as they did in Postgres.
    conn.execute("SET TimeZone = 'UTC'")
    for table in TABLES:
        path = (data_dir / f"{table}.parquet").as_posix()
        conn.execute(f"CREATE VIEW {table} AS SELECT * FROM read_parquet('{path}')")
    return conn


def _df(conn, sql: str, params: dict | None = None) -> pd.DataFrame:
    df = conn.execute(sql, params or {}).df()
    # DuckDB returns DATE columns as datetime64; the callers work with dates.
    for c in df.columns:
        if pd.api.types.is_datetime64_dtype(df[c]):
            df[c] = df[c].dt.date
    return df


def snapshot_bounds(conn) -> tuple[date, date]:
    """(first_day, last_day) of activity present in the snapshot."""
    row = _df(conn, "SELECT MIN(activity_date) a, MAX(activity_date) b FROM user_activity")
    return row.at[0, "a"], row.at[0, "b"]


# --------------------------------------------------------------------------
# Extraction
#
# Chat-level aggregates come from SQL. Anything needing a *distribution*
# (Gini, the C6 median threshold, per-chat DAU/MAU) is computed in pandas from
# per-user / per-day rows — clearer and far easier to audit than window-function
# SQL, and these tables are small.
# --------------------------------------------------------------------------

_SQL_CHAT_AGG = """
SELECT ua.chat_id,
       SUM(ua.message_count)::numeric                                            AS msgs,
       COUNT(DISTINCT ua.user_id)                                                AS active_users,
       SUM(ua.reply_count)::numeric                                              AS replies,
       SUM(ua.mention_count)::numeric                                            AS mentions,
       -- COALESCE is load-bearing: SUM(...) FILTER returns NULL, not 0, when no
       -- row matches. Without it a chat that simply sent no stickers/commands
       -- drops that metric out of its weighted score entirely instead of
       -- scoring a legitimate zero. (COUNT FILTER already returns 0.)
       COALESCE(SUM(ua.message_count) FILTER (WHERE ua.message_type = 'sticker'), 0)::numeric AS sticker_msgs,
       COALESCE(SUM(ua.message_count) FILTER (WHERE ua.message_type = 'command'), 0)::numeric AS command_msgs,
       COALESCE(SUM(ua.message_count) FILTER (WHERE ua.message_type = 'text'), 0)::numeric    AS text_msgs,
       COALESCE(SUM(ua.word_count)    FILTER (WHERE ua.message_type = 'text'), 0)::numeric    AS text_words,
       COUNT(DISTINCT ua.user_id) FILTER (WHERE ua.message_type = 'command')     AS command_users
FROM user_activity ua
WHERE ua.activity_date > $lo AND ua.activity_date <= $hi
GROUP BY ua.chat_id
"""

_SQL_PER_USER = """
SELECT chat_id, user_id, SUM(message_count)::numeric AS m
FROM user_activity
WHERE activity_date > $lo AND activity_date <= $hi
GROUP BY chat_id, user_id
HAVING SUM(message_count) > 0
"""

_SQL_DAILY = """
SELECT chat_id, activity_date, COUNT(DISTINCT user_id) AS dau
FROM user_activity
WHERE activity_date > $lo AND activity_date <= $hi
GROUP BY chat_id, activity_date
"""

_SQL_MEMBERSHIP = """
SELECT user_id, chat_id, event_type, event_date
FROM membership_history
WHERE event_date > $lo AND event_date <= $hi
"""

_SQL_LIFETIME = """
SELECT chat_id, user_id,
       (MAX(activity_date) - MIN(activity_date)) AS lifetime_days
FROM user_activity
GROUP BY chat_id, user_id
"""

_SQL_CHATS = """
WITH staff AS (
    SELECT DISTINCT staff_chat_id AS id FROM chats WHERE staff_chat_id IS NOT NULL
)
SELECT c.telegram_id            AS chat_id,
       c.title,
       c.member_count,
       c.status::text           AS status,
       c.type::text             AS chat_type,
       (c.id IN (SELECT id FROM staff)) AS is_staff_chat
FROM chats c
"""

# The score describes COMMUNITY chats. Three populations must be kept out or
# they distort both the rankings and the anchors derived from them:
#   * staff / management groups (status MANAGED, and any chat referenced as
#     another chat's staff_chat_id) - tiny, private, extremely high intensity.
#     Four of them topped the engagement ranking on the first calibration run:
#     ~700 messages across 9 people is not community engagement.
#   * channels - broadcast, no conversation; reply rate and Gini are meaningless.
#   * chats not currently in the network (LEFT / KICKED / REJECTED / PENDING /
#     EVALUATION / BLACKLISTED).
# QUARANTINE counts as in-network: the lifecycle page is explicit that its
# feature eligibility is identical to ACTIVE.
COMMUNITY_STATUSES = ("ACTIVE", "QUARANTINE")


def _gini(x: np.ndarray) -> float:
    """Gini coefficient of a non-negative array. 0 = perfectly equal."""
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    total = x.sum()
    if n < 2 or total <= 0:
        return np.nan
    i = np.arange(1, n + 1)
    return float(((2 * i - n - 1) * x).sum() / (n * total))


def fetch_window(conn, anchor: date, win_days: int = 28) -> pd.DataFrame:
    """Per-chat metric frame for the window (anchor - win_days, anchor].

    Returns one row per chat with the 14 V1 metrics, plus the sample-size
    columns the shrinkage step needs. Metrics that cannot be computed for a
    chat are NaN — they are masked out at scoring time rather than imputed.
    """
    lo, hi = anchor - timedelta(days=win_days), anchor
    plo, phi = anchor - timedelta(days=2 * win_days), lo  # previous window

    agg = _df(conn, _SQL_CHAT_AGG, {"lo": lo, "hi": hi}).set_index("chat_id")
    cur_u = _df(conn, _SQL_PER_USER, {"lo": lo, "hi": hi})
    prv_u = _df(conn, _SQL_PER_USER, {"lo": plo, "hi": phi})
    daily = _df(conn, _SQL_DAILY, {"lo": lo, "hi": hi})
    chats = _df(conn, _SQL_CHATS).set_index("chat_id")

    # Membership needs a 7-day lookahead beyond the window to judge the C5
    # cohort, so pull the whole span and slice in pandas.
    mem = _df(conn, _SQL_MEMBERSHIP, {"lo": plo, "hi": hi + timedelta(days=7)})
    if not mem.empty:
        mem["event_date"] = pd.to_datetime(mem["event_date"], utc=True)

    out = pd.DataFrame(index=agg.index)
    out["title"] = chats["title"].reindex(out.index)
    out["msgs"] = agg["msgs"].astype(float)
    out["active_users"] = agg["active_users"].astype(float)
    out["member_count"] = chats["member_count"].reindex(out.index).astype(float)

    # Population flags. Kept as columns rather than filtered away here, so the
    # notebook can report what was excluded instead of silently dropping it.
    out["status"] = chats["status"].reindex(out.index)
    out["chat_type"] = chats["chat_type"].reindex(out.index)
    out["is_staff_chat"] = chats["is_staff_chat"].reindex(out.index).fillna(False).astype(bool)
    out["in_population"] = (
        out["status"].isin(COMMUNITY_STATUSES)
        & (out["chat_type"] == "SUPERGROUP")
        & ~out["is_staff_chat"]
    )
    out.loc[out["member_count"] <= 0, "member_count"] = np.nan  # 33 chats: unusable C6 denominator

    # ---- Engagement -------------------------------------------------------
    # C11 — messages per active user. Denominator is binary-active throughout
    # (brief C-3): one definition of "active user" across every metric.
    out["c11_msgs_per_active_user"] = out["msgs"] / out["active_users"]

    # C6 — active member ratio, dynamic-median variant (brief D6): share of the
    # chat's *members* posting at or above that chat's own median poster.
    med = cur_u.groupby("chat_id")["m"].median()
    at_or_above = (
        cur_u.join(med.rename("med"), on="chat_id")
        .query("m >= med")
        .groupby("chat_id")["user_id"].count()
    )
    out["c6_active_member_ratio"] = at_or_above.reindex(out.index) / out["member_count"]

    # U8 — returning user rate, per chat.
    cur_set = cur_u.groupby("chat_id")["user_id"].apply(set)
    prv_set = prv_u.groupby("chat_id")["user_id"].apply(set)
    both = prv_set.index.intersection(cur_set.index)
    u8 = pd.Series(
        {c: len(prv_set[c] & cur_set[c]) / len(prv_set[c]) for c in both if prv_set[c]},
        dtype=float,
    )
    out["u8_returning_user_rate"] = u8.reindex(out.index)
    out["prev_active_users"] = prv_set.reindex(out.index).apply(
        lambda s: len(s) if isinstance(s, set) else np.nan
    )

    # N4 — stickiness. Single-day DAU is far too noisy per chat, so this uses
    # MEAN daily active users over the window / distinct users in the window.
    # Documented deviation from the catalogue's single-day DAU.
    mean_dau = daily.groupby("chat_id")["dau"].mean()
    out["n4_stickiness"] = mean_dau.reindex(out.index) / out["active_users"]

    # U9 — median user lifetime in the chat.
    # WARNING: truncated by the snapshot's own start date. With ~81 days of
    # history nobody can show a lifetime above that. Unreliable here; see the
    # notebook's limitations cell.
    life = _df(conn, _SQL_LIFETIME)
    out["u9_median_lifetime_days"] = (
        life.groupby("chat_id")["lifetime_days"].median().reindex(out.index).astype(float)
    )

    # C9 / C13 — bot reach and bot depth.
    out["c9_bot_usage_rate"] = agg["command_users"].astype(float) / out["active_users"]
    out["c13_commands_per_user"] = agg["command_msgs"].astype(float) / out["active_users"]

    # ---- Quality ----------------------------------------------------------
    out["u7_reply_rate"] = agg["replies"].astype(float) / out["msgs"]
    out["u10_mention_density"] = agg["mentions"].astype(float) / out["msgs"]
    out["u6_avg_words"] = agg["text_words"].astype(float) / agg["text_msgs"].replace(0, np.nan)
    out["c3_sticker_ratio"] = agg["sticker_msgs"].astype(float) / out["msgs"]

    gini = cur_u.groupby("chat_id")["m"].apply(lambda s: _gini(s.values))
    out["c8_gini"] = gini.reindex(out.index)

    # C10 — ban rate per active member.
    # NB: kpi_queries.sql claims no ban events exist. That comment is STALE —
    # this snapshot carries 3,295 bans and 5,073 restricts. Verified, not assumed.
    if not mem.empty:
        w_lo = pd.Timestamp(lo, tz="UTC")
        w_hi = pd.Timestamp(hi, tz="UTC")
        in_win = mem[(mem["event_date"] > w_lo) & (mem["event_date"] <= w_hi)]
        bans = in_win[in_win["event_type"] == "ban"].groupby("chat_id").size()
        out["c10_ban_rate"] = bans.reindex(out.index).fillna(0) / out["active_users"]

        # C5 — newcomer rejection: of users joining in the window, the share who
        # left or were banned within 7 days. Joins in the final 7 days are
        # EXCLUDED — they have not had a full 7 days to churn, and including
        # them biases the rate downward.
        cohort_hi = w_hi - pd.Timedelta(days=7)
        joins = mem[
            (mem["event_type"] == "join")
            & (mem["event_date"] > w_lo)
            & (mem["event_date"] <= cohort_hi)
        ].groupby(["chat_id", "user_id"])["event_date"].min().rename("joined_at")

        exits = mem[mem["event_type"].isin(["leave", "ban"])]
        if not joins.empty and not exits.empty:
            j = joins.reset_index()
            e = exits[["chat_id", "user_id", "event_date"]].rename(columns={"event_date": "exit_at"})
            merged = j.merge(e, on=["chat_id", "user_id"], how="left")
            merged["churned"] = (
                (merged["exit_at"] > merged["joined_at"])
                & (merged["exit_at"] <= merged["joined_at"] + pd.Timedelta(days=7))
            )
            per = merged.groupby(["chat_id", "user_id"])["churned"].any().reset_index()
            c5 = per.groupby("chat_id")["churned"].mean()
            out["c5_newcomer_rejection"] = c5.reindex(out.index)
            out["cohort_size"] = per.groupby("chat_id").size().reindex(out.index)

    for col in ("c10_ban_rate", "c5_newcomer_rejection", "cohort_size"):
        if col not in out:
            out[col] = np.nan

    out["anchor"] = anchor
    out["win_days"] = win_days
    return out


# --------------------------------------------------------------------------
# Scorers  (brief §5.2)
# --------------------------------------------------------------------------

Shape = Literal["higher_better", "lower_better", "band"]


def higher_better(x, lo, hi):
    return np.clip((x - lo) / (hi - lo), 0, 1) * 100


def lower_better(x, lo, hi):
    return np.clip((hi - x) / (hi - lo), 0, 1) * 100


def higher_better_3pt(x, lo, mid, hi):
    """Piecewise-linear ramp through three anchors: lo->0, mid->50, hi->100.

    Every metric here is heavy-tailed. A straight P10..P90 ramp maps the MEDIAN
    chat to ~20/100 (c11: P10=4.7, P50=17.9, P90=70.0), so most of the network
    piles up near zero and the score stops discriminating where the chats
    actually are. Pinning the median to 50 fixes that while keeping the anchors
    absolute and frozen -- it is still "this value scores this much", not a
    rank against whoever else was measured today.
    """
    x = np.asarray(x, dtype=float)
    lower = np.clip((x - lo) / (mid - lo), 0, 1) * 50
    upper = 50 + np.clip((x - mid) / (hi - mid), 0, 1) * 50
    return np.where(x <= mid, lower, upper)


def lower_better_3pt(x, lo, mid, hi):
    """Mirror of higher_better_3pt: lo->100, mid->50, hi->0."""
    return 100 - higher_better_3pt(x, lo, mid, hi)


def band(x, zero_lo, full_lo, full_hi, zero_hi):
    """Plateau scorer: 100 across [full_lo, full_hi], 0 at/outside the zero anchors.

    For metrics where *more is not better*. Gini near 0 is not utopia — it means
    nobody talks much; a monotonic scorer would rate a flat, dead chat as perfect.
    """
    x = np.asarray(x, dtype=float)
    up = np.clip((x - zero_lo) / (full_lo - zero_lo), 0, 1)
    dn = np.clip((zero_hi - x) / (zero_hi - full_hi), 0, 1)
    return np.minimum(up, dn) * 100


# --------------------------------------------------------------------------
# Metric specification  (brief §6 — target weights over CURRENT scope)
#
# `target_w` is the weight a metric gets within the currently approved scoring
# construct. Unavailable engagement metrics remain visible as a coverage gap.
# Deferred content-quality features are kept separately as a roadmap and do not
# reduce coverage of the present structural-quality score.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    pillar: Literal["engagement", "quality"]
    facet: str
    target_w: float
    shape: Shape | None          # None => not available in V1
    sample_col: str = "msgs"     # which n drives shrinkage
    k: float = 200.0             # half-weight point
    band_anchors: tuple | None = None  # fixed (non-percentile) anchors


METRICS: list[Metric] = [
    # ---- Engagement (target 77; 73.16 available => 95% coverage) ----------
    Metric("c11_msgs_per_active_user", "Messages per active user", "engagement", "Activity", 20, "higher_better", "msgs", 100),
    Metric("c6_active_member_ratio", "Active member ratio (median)", "engagement", "Activity", 15, "higher_better", "active_users", 12.5),
    Metric("u8_returning_user_rate", "Returning user rate", "engagement", "Retention", 15, "higher_better", "prev_active_users", 12.5),
    Metric("n4_stickiness", "Stickiness (mean DAU / MAU)", "engagement", "Retention", 15, "higher_better", "active_users", 12.5),
    # Bot facet rescaled 25 -> 12 at calibration: bot reach is heavily skewed
    # (median chat 2.3% of users), easy for an admin to game, and cutting it
    # barely moved the ranking (rho > 0.98 across the whole sweep).
    Metric("c9_bot_usage_rate", "Bot usage rate", "engagement", "Bot", 4.8, "higher_better", "active_users", 12.5),
    Metric("c13_commands_per_user", "Commands per user", "engagement", "Bot", 3.36, "higher_better", "active_users", 12.5),
    Metric("feature_popularity", "Feature popularity", "engagement", "Bot", 3.84, None),  # Phase 2

    # ---- Structural quality (current target 47; 47 available => 100%) -----
    # The current model deliberately measures structural conversation quality:
    # interaction depth, participation health, signal depth, and observable
    # moderation outcomes. Content-safety, relevance, spam, deletion, and
    # response-time features are documented below as FUTURE_QUALITY_METRICS;
    # they are not part of the current target denominator.
    Metric("u7_reply_rate", "Reply rate", "quality", "Interaction depth", 12, "higher_better", "msgs", 100),
    Metric("u10_mention_density", "Mention density", "quality", "Interaction depth", 8, "higher_better", "msgs", 100),
    Metric("c8_gini", "Gini coefficient", "quality", "Participation health", 12, "band", "active_users", 12.5,
           band_anchors=(0.05, 0.30, 0.50, 0.80)),  # literature anchors, not percentiles
    Metric("u6_avg_words", "Avg words per text message", "quality", "Signal-to-noise", 5, "higher_better", "text_msgs", 100),
    Metric("c10_ban_rate", "Ban rate", "quality", "Moderation & repulsion", 5, "lower_better", "active_users", 12.5),
    Metric("c5_newcomer_rejection", "Newcomer rejection (7d)", "quality", "Moderation & repulsion", 5, "lower_better", "cohort_size", 7.5),
]

# Explicitly out of scope for the current structural-quality score. Retained as
# a transparent roadmap rather than counted as missing coverage.
FUTURE_QUALITY_METRICS: list[Metric] = [
    Metric("avg_response_time", "Avg response time", "quality", "Interaction depth", 5, None),
    Metric("conversation_balance", "Conversation balance", "quality", "Participation health", 3, None),
    Metric("spam_ratio", "Spam ratio", "quality", "Signal-to-noise", 8, None),
    Metric("link_without_reply", "Link-without-reply ratio", "quality", "Signal-to-noise", 4, None),
    Metric("off_topic_ratio", "Off-topic ratio", "quality", "Relevance", 7, None),
    Metric("topic_consistency", "Topic consistency", "quality", "Relevance", 4, None),
    Metric("game_keyword_freq", "Game keyword frequency", "quality", "Relevance", 4, None),
    Metric("toxicity_rate", "Toxicity rate", "quality", "Moderation & repulsion", 10, None),
    Metric("deletion_rate", "Deletion rate", "quality", "Moderation & repulsion", 3, None),  # Phase 2
    Metric("moderator_response_time", "Moderator response time", "quality", "Moderation & repulsion", 2, None),  # Phase 2
]

# Cut at calibration. Recorded so they are not re-proposed: these are REJECTED,
# not pending, so their weight leaves the target vector entirely rather than
# counting as a coverage gap.
REJECTED = {
    "c3_sticker_ratio": "Degenerate on this network — median 0.05% of messages, "
                        "77% of chats clipped to a flat 0 or 100. Encodes little "
                        "more than 'has this chat ever sent a sticker'.",
    "u9_median_lifetime_days": "Not measuring lifetime. Median 2 days, and "
                               "truncated by the telemetry start date, so no chat "
                               "can show a lifetime longer than the history itself. "
                               "Revisit after a year of data.",
}

AVAILABLE = [m for m in METRICS if m.shape is not None]


def coverage(pillar: str) -> float:
    """Share of the current scoped pillar target that V1 can compute."""
    tot = sum(m.target_w for m in METRICS if m.pillar == pillar)
    got = sum(m.target_w for m in AVAILABLE if m.pillar == pillar)
    return 100.0 * got / tot


def v1_weights(pillar: str) -> dict[str, float]:
    """Masked + renormalised weights for a pillar (sums to 1.0)."""
    ms = [m for m in AVAILABLE if m.pillar == pillar]
    tot = sum(m.target_w for m in ms)
    return {m.key: m.target_w / tot for m in ms}


# --------------------------------------------------------------------------
# Anchors  (brief §5.1 — derive from data, review, then FREEZE)
# --------------------------------------------------------------------------


@dataclass
class Anchors:
    lo: dict[str, float] = field(default_factory=dict)
    hi: dict[str, float] = field(default_factory=dict)
    band: dict[str, tuple] = field(default_factory=dict)
    median: dict[str, float] = field(default_factory=dict)
    p_lo: float = 10.0
    p_hi: float = 90.0


def derive_anchors(df: pd.DataFrame, p_lo: float = 10, p_hi: float = 90) -> Anchors:
    """Derive anchors from a metric frame (or several concatenated windows).

    These are CANDIDATES, not settings. Review them, sanity-check them against
    judgement, then freeze them as constants with a model version. Re-deriving
    them per run would make every score relative to that run — exactly the
    failure mode anchored-absolute scoring exists to avoid.

    With only ~43 eligible chats, P10/P90 are the 4th and 39th order statistics
    and are jumpy. Consider p_lo=20 / p_hi=80, or pool several windows.
    """
    a = Anchors(p_lo=p_lo, p_hi=p_hi)
    for m in AVAILABLE:
        if m.key not in df.columns:
            continue
        s = pd.to_numeric(df[m.key], errors="coerce").dropna()
        if s.empty:
            continue
        a.median[m.key] = float(s.median())
        if m.shape == "band":
            if m.band_anchors:
                a.band[m.key] = m.band_anchors
            else:
                # Data-derived plateau: the interquartile core is "normal",
                # falling to zero at the distribution's extremes.
                a.band[m.key] = (
                    float(s.quantile(0.02)), float(s.quantile(0.25)),
                    float(s.quantile(0.75)), float(s.quantile(0.98)),
                )
        else:
            a.lo[m.key] = float(s.quantile(p_lo / 100))
            a.hi[m.key] = float(s.quantile(p_hi / 100))
    return a


# --------------------------------------------------------------------------
# Frozen anchors — calibration v1
#
# Derived 2026-08-17 from the 2026-08-12 production snapshot: P10 / P50 / P90
# pooled across four rolling 28-day windows (2026-07-22 .. 2026-08-12),
# restricted to eligible COMMUNITY chats (209 chat-windows). Gini keeps the
# catalogue's literature band rather than percentiles.
#
# These are FROZEN ON PURPOSE. Re-deriving them per run would make every score
# relative to whoever happened to be measured that day, which defeats the whole
# point of anchored-absolute scoring. Re-calibrate deliberately, on a schedule
# or on a large shift in the network median, and bump the version when you do.
#
# Known weakness: the P90 anchors are estimated from ~52 chats and bootstrap at
# roughly +/-15-20% relative error on heavy-tailed metrics (c11 especially).
# They will move on the next calibration. The P10 and median anchors are firm.
# --------------------------------------------------------------------------

SCORING_MODEL_VERSION = "v1.0-cal-2026-08-17"

FROZEN_ANCHORS: dict[str, dict] = {
    "c11_msgs_per_active_user": {"lo": 4.6424, "mid": 16.2917, "hi": 55.5520},
    "c6_active_member_ratio":   {"lo": 0.0103, "mid": 0.0288, "hi": 0.0585},
    "u8_returning_user_rate":   {"lo": 0.2427, "mid": 0.4344, "hi": 0.6471},
    "n4_stickiness":            {"lo": 0.0859, "mid": 0.1498, "hi": 0.2385},
    "c9_bot_usage_rate":        {"lo": 0.0000, "mid": 0.0228, "hi": 0.1667},
    "c13_commands_per_user":    {"lo": 0.0000, "mid": 0.0417, "hi": 0.4317},
    "u7_reply_rate":            {"lo": 0.1120, "mid": 0.4245, "hi": 0.6973},
    "u10_mention_density":      {"lo": 0.0018, "mid": 0.0130, "hi": 0.0493},
    "u6_avg_words":             {"lo": 5.0613, "mid": 7.8464, "hi": 12.8308},
    "c10_ban_rate":             {"lo": 0.0000, "mid": 0.1444, "hi": 0.7532},
    "c5_newcomer_rejection":    {"lo": 0.0346, "mid": 0.4000, "hi": 0.7712},
    # "mid" on a band metric is not an anchor — it is the frozen shrinkage
    # target. Without it, shrinkage would fall back to the current window's
    # median and quietly make the score relative again.
    "c8_gini":                  {"band": (0.05, 0.30, 0.50, 0.80), "mid": 0.6780},
}


def frozen_anchors() -> Anchors:
    """The calibrated anchor set. Use this for scoring; use derive_anchors()
    only to study what a fresh calibration would produce."""
    a = Anchors()
    for key, spec in FROZEN_ANCHORS.items():
        if "band" in spec:
            a.band[key] = spec["band"]
        if "lo" in spec:
            a.lo[key], a.hi[key] = spec["lo"], spec["hi"]
        if "mid" in spec:
            a.median[key] = spec["mid"]
    return a


# --------------------------------------------------------------------------
# Scoring  (brief §7)
# --------------------------------------------------------------------------

MIN_MSGS = 50
MIN_ACTIVE_USERS = 5

# Pin the network median to 50 on ramp metrics (see higher_better_3pt).
# Set False to compare against the plain two-point ramp.
USE_THREE_POINT = True


def shrink(values: pd.Series, n: pd.Series, k: float, network_median: float) -> pd.Series:
    """lambda = n/(n+k); pull under-sampled chats toward the network median."""
    lam = n.astype(float) / (n.astype(float) + k)
    return lam * values + (1 - lam) * network_median


def score_pillar(
    df: pd.DataFrame,
    anchors: Anchors,
    pillar: str,
    weight_overrides: dict[str, float] | None = None,
    apply_shrinkage: bool = True,
) -> pd.DataFrame:
    """Compute a pillar score per chat.

    Weights are renormalised PER CHAT over the metrics that are non-null for
    that chat, so a chat with member_count = 0 (no C6) is scored on what it has
    rather than silently penalised.
    """
    ms = [m for m in AVAILABLE if m.pillar == pillar and m.key in df.columns]
    base_w = v1_weights(pillar)
    if weight_overrides:
        base_w = {**base_w, **weight_overrides}
        tot = sum(base_w[m.key] for m in ms)
        base_w = {k: v / tot for k, v in base_w.items()}

    sub = pd.DataFrame(index=df.index)
    wts = pd.DataFrame(index=df.index)

    for m in ms:
        raw = pd.to_numeric(df[m.key], errors="coerce")
        if apply_shrinkage and m.sample_col in df.columns:
            n = pd.to_numeric(df[m.sample_col], errors="coerce").fillna(0)
            raw = shrink(raw, n, m.k, anchors.median.get(m.key, float(raw.median())))

        if m.shape == "band":
            s = band(raw, *anchors.band[m.key])
        else:
            lo, hi = anchors.lo[m.key], anchors.hi[m.key]
            mid = anchors.median.get(m.key)
            # Three-point needs a strictly increasing lo < mid < hi; fall back
            # to the plain ramp on a degenerate distribution.
            three = USE_THREE_POINT and mid is not None and lo < mid < hi
            if m.shape == "higher_better":
                s = (higher_better_3pt(raw, lo, mid, hi) if three
                     else higher_better(raw, lo, hi))
            else:
                s = (lower_better_3pt(raw, lo, mid, hi) if three
                     else lower_better(raw, lo, hi))

        s = pd.Series(s, index=df.index).where(pd.to_numeric(df[m.key], errors="coerce").notna())
        sub[m.key] = s
        wts[m.key] = np.where(s.notna(), base_w[m.key], 0.0)

    wsum = wts.sum(axis=1)
    out = pd.DataFrame(index=df.index)
    out["title"] = df.get("title")
    out[f"{pillar}_score"] = (sub * wts).sum(axis=1) / wsum.replace(0, np.nan)
    out[f"{pillar}_metric_coverage"] = wsum  # < 1.0 => some metric was null
    for c in sub.columns:
        out[f"sub_{c}"] = sub[c]

    eligible = (
        (pd.to_numeric(df["msgs"], errors="coerce") >= MIN_MSGS)
        & (pd.to_numeric(df["active_users"], errors="coerce") >= MIN_ACTIVE_USERS)
        & df.get("in_population", pd.Series(True, index=df.index)).astype(bool)
    )
    out["eligible"] = eligible
    out.loc[~eligible, f"{pillar}_score"] = np.nan  # UNRATED, not zero
    return out


def score_all(df: pd.DataFrame, anchors: Anchors, **kw) -> pd.DataFrame:
    e = score_pillar(df, anchors, "engagement", **kw)
    q = score_pillar(df, anchors, "quality", **kw)
    out = e[["title", "engagement_score", "engagement_metric_coverage", "eligible"]].copy()
    out["quality_score"] = q["quality_score"]
    out["quality_metric_coverage"] = q["quality_metric_coverage"]
    out["msgs"] = df["msgs"]
    out["active_users"] = df["active_users"]
    return out.join(e.filter(like="sub_")).join(q.filter(like="sub_"))


def quadrant(scores: pd.DataFrame, e_cut: float | None = None, q_cut: float | None = None) -> pd.Series:
    """Label each chat with its brief §9 cell. Cuts default to the medians."""
    e = scores["engagement_score"]
    q = scores["quality_score"]
    ec = e.median() if e_cut is None else e_cut
    qc = q.median() if q_cut is None else q_cut
    lab = pd.Series(index=scores.index, dtype=object)
    lab[(e >= ec) & (q >= qc)] = "Target state"
    lab[(e >= ec) & (q < qc)] = "Busy but unhealthy"
    lab[(e < ec) & (q >= qc)] = "Good but quiet"
    lab[(e < ec) & (q < qc)] = "Merge / retire candidate"
    return lab
