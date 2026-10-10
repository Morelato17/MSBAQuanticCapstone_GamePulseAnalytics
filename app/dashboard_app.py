"""Streamlit dashboard for GIG community engagement and structural quality.

Run with:
    streamlit run app/dashboard_app.py

The app reads the frozen, pseudonymised snapshot in ``data/`` through
``scoring_model.py``. Chat titles, chat ids and user ids are synthetic; user
leaderboards contain ``user_id`` only.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scoring_model import (
    MIN_ACTIVE_USERS,
    MIN_MSGS,
    SCORING_MODEL_VERSION,
    _df,
    connect,
    fetch_window,
    frozen_anchors,
    quadrant,
    score_all,
    snapshot_bounds,
)

DEFAULT_SCORE_WINDOW = 28
MAX_TREND_POINTS = 26
ATTENTION_THRESHOLDS = {
    "low_score": 40.0,
    "healthy_score": 50.0,
    "rapid_drop": -10.0,
    "high_newcomer_rejection": 0.60,
    "high_ban_rate": 0.50,
    "low_reply_rate": 0.15,
    "low_returning_rate": 0.25,
    "low_coverage": 0.80,
}


st.set_page_config(page_title="GIG Community Health", page_icon="📊", layout="wide")


@st.cache_data(show_spinner=False)
def get_snapshot_bounds() -> tuple[date, date]:
    conn = connect()
    try:
        return snapshot_bounds(conn)
    finally:
        conn.close()


@st.cache_data(show_spinner=False)
def load_scored_window(anchor: date, window_days: int) -> pd.DataFrame:
    """Return raw metrics and scores for one rolling window."""
    conn = connect()
    try:
        raw = fetch_window(conn, anchor, window_days)
    finally:
        conn.close()

    scores = score_all(raw, frozen_anchors())
    result = raw.join(scores.drop(columns=["title", "msgs", "active_users"], errors="ignore"))
    # Fixed cuts make health labels comparable across different snapshots.
    result["health_category"] = quadrant(scores, e_cut=50, q_cut=50)
    result.index.name = "chat_id"
    return result.reset_index()


@st.cache_data(show_spinner=False)
def load_top_users(chat_id: int, start: date, end: date, limit: int = 20) -> pd.DataFrame:
    """Contributor leaderboard containing user IDs only."""
    sql = """
        SELECT user_id,
               SUM(message_count)::bigint AS messages,
               COUNT(DISTINCT activity_date)::int AS active_days,
               SUM(reply_count)::bigint AS replies,
               SUM(mention_count)::bigint AS mentions,
               COUNT(DISTINCT chat_id)::int AS chats_active_in
        FROM user_activity
        WHERE chat_id = $chat_id
          AND activity_date >= $start
          AND activity_date <= $end
        GROUP BY user_id
        ORDER BY messages DESC, active_days DESC
        LIMIT $limit
    """
    conn = connect()
    try:
        return _df(conn, sql, {"chat_id": int(chat_id), "start": start, "end": end, "limit": limit})
    finally:
        conn.close()


@st.cache_data(show_spinner=False)
def load_contributors(
    start: date,
    end: date,
    metric: str,
    limit: int,
    chat_id: int | None = None,
) -> pd.DataFrame:
    """Rank contributor IDs within the network or one chat.

    The metric is interpolated only after validation against this allow-list;
    all user-provided values remain query parameters.
    """
    order_columns = {
        "messages": "messages",
        "active_days": "active_days",
        "replies": "replies",
        "mentions": "mentions_made",
        "chats_active_in": "chats_active_in",
    }
    if metric not in order_columns:
        raise ValueError(f"Unsupported contributor metric: {metric}")

    chat_filter = "AND ua.chat_id = $chat_id" if chat_id is not None else ""
    order_column = order_columns[metric]
    sql = f"""
        WITH staff AS (
            SELECT DISTINCT staff_chat_id AS id
            FROM chats
            WHERE staff_chat_id IS NOT NULL
        ),
        community_chats AS (
            SELECT c.telegram_id AS chat_id
            FROM chats c
            WHERE c.status::text IN ('ACTIVE', 'QUARANTINE')
              AND c.type::text = 'SUPERGROUP'
              AND NOT EXISTS (SELECT 1 FROM staff s WHERE s.id = c.id)
        ),
        per_user AS (
            SELECT ua.user_id,
                   SUM(ua.message_count)::bigint AS messages,
                   COUNT(DISTINCT ua.activity_date)::int AS active_days,
                   SUM(ua.reply_count)::bigint AS replies,
                   SUM(ua.mention_count)::bigint AS mentions_made,
                   COUNT(DISTINCT ua.chat_id)::int AS chats_active_in
            FROM user_activity ua
            INNER JOIN community_chats cc ON cc.chat_id = ua.chat_id
            WHERE ua.activity_date >= $start
              AND ua.activity_date <= $end
              {chat_filter}
            GROUP BY ua.user_id
        )
        SELECT user_id,
               messages,
               active_days,
               replies,
               mentions_made,
               chats_active_in,
               SUM(messages) OVER ()::bigint AS scope_messages
        FROM per_user
        ORDER BY {order_column} DESC, messages DESC, user_id
        LIMIT $limit
    """
    params = {"start": start, "end": end, "limit": int(limit)}
    if chat_id is not None:
        params["chat_id"] = int(chat_id)

    conn = connect()
    try:
        return _df(conn, sql, params)
    finally:
        conn.close()


def build_trend(start: date, end: date, window_days: int) -> pd.DataFrame:
    """Sample rolling scores weekly, capping the number of expensive queries."""
    anchors = list(pd.date_range(start=start, end=end, freq="7D").date)
    if not anchors or anchors[-1] != end:
        anchors.append(end)
    if len(anchors) > MAX_TREND_POINTS:
        positions = pd.Series(range(len(anchors))).sample(MAX_TREND_POINTS, random_state=7).sort_values()
        anchors = [anchors[i] for i in positions]

    frames = []
    progress = st.progress(0, text="Calculating weekly score history…")
    for i, anchor in enumerate(anchors, start=1):
        frame = load_scored_window(anchor, window_days)
        frame["score_date"] = anchor
        frames.append(frame)
        progress.progress(i / len(anchors), text=f"Calculating score history: {i}/{len(anchors)}")
    progress.empty()
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def add_period_change(current: pd.DataFrame, previous: pd.DataFrame) -> pd.DataFrame:
    prior = previous.set_index("chat_id")[["engagement_score", "quality_score"]].rename(
        columns={
            "engagement_score": "previous_engagement_score",
            "quality_score": "previous_quality_score",
        }
    )
    out = current.join(prior, on="chat_id")
    out["engagement_change"] = out["engagement_score"] - out["previous_engagement_score"]
    out["quality_change"] = out["quality_score"] - out["previous_quality_score"]
    return out


def attention_queue(frame: pd.DataFrame) -> pd.DataFrame:
    """Create transparent, rule-based operational flags."""
    t = ATTENTION_THRESHOLDS
    rows: list[dict] = []

    def flag(row: pd.Series, label: str, severity: int, evidence: str, action: str) -> None:
        # Size gives otherwise-equal issues affecting more users a small boost.
        impact = min(float(row.get("active_users", 0) or 0), 200) / 200
        rows.append(
            {
                "chat_id": row.chat_id,
                "chat": row.title,
                "flag": label,
                "priority": round(severity + impact, 2),
                "evidence": evidence,
                "suggested_action": action,
            }
        )

    for _, row in frame.iterrows():
        if not bool(row.get("eligible", False)):
            flag(
                row,
                "Unrated / insufficient evidence",
                1,
                f"{int(row.msgs)} messages; {int(row.active_users)} active users",
                f"Collect at least {MIN_MSGS} messages and {MIN_ACTIVE_USERS} active users.",
            )
            continue

        e, q = row.engagement_score, row.quality_score
        if e >= t["healthy_score"] and q < t["low_score"]:
            flag(row, "Busy but unhealthy", 4, f"Engagement {e:.1f}; quality {q:.1f}", "Review the weakest quality sub-scores.")
        if e < t["low_score"] and q < t["low_score"]:
            flag(row, "Low engagement and quality", 3, f"Engagement {e:.1f}; quality {q:.1f}", "Review viability, programming, and moderation together.")
        if row.quality_change <= t["rapid_drop"]:
            flag(row, "Rapid quality deterioration", 5, f"Quality changed {row.quality_change:+.1f} points", "Inspect metric changes and recent moderation events.")
        if row.engagement_change <= t["rapid_drop"]:
            flag(row, "Rapid engagement deterioration", 4, f"Engagement changed {row.engagement_change:+.1f} points", "Inspect retention, activity, and recent community changes.")
        if row.get("c5_newcomer_rejection", 0) >= t["high_newcomer_rejection"]:
            flag(row, "High newcomer rejection", 4, f"7-day rejection {row.c5_newcomer_rejection:.1%}", "Review onboarding and first-week experience.")
        if row.get("c10_ban_rate", 0) >= t["high_ban_rate"]:
            flag(row, "High moderation pressure", 4, f"Ban rate {row.c10_ban_rate:.1%}", "Review ban causes and moderation workload.")
        if row.get("u7_reply_rate", 1) < t["low_reply_rate"]:
            flag(row, "Weak conversation depth", 2, f"Reply rate {row.u7_reply_rate:.1%}", "Try prompts or formats that encourage member-to-member replies.")
        if row.get("u8_returning_user_rate", 1) < t["low_returning_rate"]:
            flag(row, "Retention risk", 3, f"Returning-user rate {row.u8_returning_user_rate:.1%}", "Investigate why prior participants did not return.")
        coverage = min(row.get("engagement_metric_coverage", 1), row.get("quality_metric_coverage", 1))
        if coverage < t["low_coverage"]:
            flag(row, "Low metric coverage", 1, f"Minimum pillar coverage {coverage:.0%}", "Treat the score cautiously and inspect missing inputs.")

    if not rows:
        return pd.DataFrame(columns=["chat_id", "chat", "flag", "priority", "evidence", "suggested_action"])
    return pd.DataFrame(rows).sort_values(["priority", "chat"], ascending=[False, True])


def overview_page(current: pd.DataFrame) -> None:
    eligible = current[current["eligible"]].copy()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Eligible chats", f"{len(eligible):,}")
    c2.metric("Unrated chats", f"{(~current['eligible']).sum():,}")
    c3.metric("Messages", f"{current['msgs'].sum():,.0f}")
    c4.metric("Active chat-users", f"{current['active_users'].sum():,.0f}", help="Sum of per-chat active users; a person active in two chats is counted twice.")
    c5.metric("Median quality", f"{eligible['quality_score'].median():.1f}" if not eligible.empty else "—")

    if eligible.empty:
        st.warning("No chats meet the scoring eligibility threshold in this window.")
        return

    fig = px.scatter(
        eligible,
        x="engagement_score",
        y="quality_score",
        size="active_users",
        color="health_category",
        hover_name="title",
        hover_data={"chat_id": True, "msgs": ":,.0f", "active_users": ":,.0f", "engagement_change": ":+.1f", "quality_change": ":+.1f"},
        labels={"engagement_score": "Engagement score", "quality_score": "Structural-quality score", "health_category": "Health category"},
        title="Engagement and structural quality",
        size_max=42,
    )
    fig.add_vline(x=50, line_dash="dot", line_color="gray")
    fig.add_hline(y=50, line_dash="dot", line_color="gray")
    fig.update_layout(height=600)
    st.plotly_chart(fig, width="stretch")

    ranking = eligible.assign(
        engagement_rank=eligible["engagement_score"].rank(method="min", ascending=False).astype(int),
        quality_rank=eligible["quality_score"].rank(method="min", ascending=False).astype(int),
    )
    st.subheader("Chat ranking")
    st.dataframe(
        ranking[["chat_id", "title", "engagement_rank", "engagement_score", "engagement_change", "quality_rank", "quality_score", "quality_change", "active_users", "msgs", "health_category", "engagement_metric_coverage", "quality_metric_coverage"]],
        hide_index=True,
        width="stretch",
        column_config={
            "engagement_score": st.column_config.ProgressColumn("Engagement", min_value=0, max_value=100, format="%.1f"),
            "quality_score": st.column_config.ProgressColumn("Quality", min_value=0, max_value=100, format="%.1f"),
            "engagement_change": st.column_config.NumberColumn("Engagement Δ", format="%+.1f"),
            "quality_change": st.column_config.NumberColumn("Quality Δ", format="%+.1f"),
            "engagement_metric_coverage": st.column_config.NumberColumn("Engagement coverage", format="percent"),
            "quality_metric_coverage": st.column_config.NumberColumn("Quality coverage", format="percent"),
        },
    )


def evolution_page(start: date, end: date, window_days: int, current: pd.DataFrame) -> None:
    eligible_titles = current.loc[current["eligible"], "title"].dropna().sort_values().tolist()
    palette = px.colors.qualitative.Dark2
    selected = st.multiselect(
        "Chats to compare",
        eligible_titles,
        default=eligible_titles[: min(2, len(eligible_titles))],
        max_selections=5,
    )
    if not selected:
        st.info("Select at least one chat.")
        return

    # Streamlit gives every selected multiselect tag the primary accent color.
    # Match tags to the chart palette by selection order so the selector itself
    # acts as a visual key. Multiple selectors cover current BaseWeb tag markup.
    chip_rules = []
    for index, color in enumerate(palette[: len(selected)]):
        chip_rules.append(
            f"""
            [data-testid="stMultiSelectTagsContainer"]
            [data-tag][data-tag-index="{index}"] {{
                background-color: {color} !important;
                border-color: {color} !important;
                color: white !important;
            }}
            [data-testid="stMultiSelectTagsContainer"]
            [data-tag][data-tag-index="{index}"] * {{
                color: white !important;
                fill: white !important;
            }}
            """
        )
    st.markdown(f"<style>{''.join(chip_rules)}</style>", unsafe_allow_html=True)

    focus_choice = st.selectbox(
        "Highlight chat",
        ["Show all equally", *selected],
        help="Focus on one chat across both score panels while retaining the others as context.",
    )
    focused_chat = None if focus_choice == "Show all equally" else focus_choice

    trend = build_trend(start, end, window_days)
    trend = trend[trend["title"].isin(selected) & trend["eligible"]]

    if trend.empty:
        st.info("No eligible score observations are available for the selected chats and dates.")
        return

    st.info(
        f"""**How to read these charts**

- Each point summarizes the preceding **{window_days} days** and is sampled weekly.
- The date on the horizontal axis is the **end of the rolling window**.
- Select **Highlight chat** to emphasize one chat and see its latest scores and weekly changes above the charts.
- Hover over any point for its exact date and score.""",
        icon="❔",
    )

    # Score type is encoded by panel; color consistently identifies the chat.
    chat_colors = {chat: palette[i % len(palette)] for i, chat in enumerate(selected)}

    if focused_chat is not None:
        focused_data = trend[trend["title"] == focused_chat].sort_values("score_date")
        if not focused_data.empty:
            engagement_delta = focused_data["engagement_score"].diff().iloc[-1]
            quality_delta = focused_data["quality_score"].diff().iloc[-1]
            st.markdown(f"**Latest score — {focused_chat}**")
            latest_1, latest_2 = st.columns(2)
            latest_1.metric(
                "Engagement",
                f"{focused_data['engagement_score'].iloc[-1]:.1f}",
                f"{engagement_delta:+.1f} vs previous observation" if pd.notna(engagement_delta) else None,
            )
            latest_2.metric(
                "Structural quality",
                f"{focused_data['quality_score'].iloc[-1]:.1f}",
                f"{quality_delta:+.1f} vs previous observation" if pd.notna(quality_delta) else None,
            )

    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.08,
        subplot_titles=("Engagement score", "Structural quality score"),
    )

    for chat in selected:
        chat_data = trend[trend["title"] == chat].sort_values("score_date")
        if chat_data.empty:
            continue
        color = chat_colors[chat]
        is_emphasized = focused_chat is None or chat == focused_chat
        opacity = 1.0 if is_emphasized else 0.18
        width = 3 if focused_chat == chat else 2
        marker_size = 7 if focused_chat == chat else 5

        fig.add_trace(
            go.Scatter(
                x=chat_data["score_date"],
                y=chat_data["engagement_score"],
                name=chat,
                legendgroup=chat,
                showlegend=True,
                mode="lines+markers",
                opacity=opacity,
                line={"color": color, "width": width},
                marker={"color": color, "size": marker_size},
                hovertemplate=(
                    f"<b>{chat}</b><br>Date: %{{x|%d %b %Y}}<br>"
                    "Engagement: %{y:.1f}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=chat_data["score_date"],
                y=chat_data["quality_score"],
                name=chat,
                legendgroup=chat,
                showlegend=False,
                mode="lines+markers",
                opacity=opacity,
                line={"color": color, "width": width},
                marker={"color": color, "size": marker_size},
                hovertemplate=(
                    f"<b>{chat}</b><br>Date: %{{x|%d %b %Y}}<br>"
                    "Quality: %{y:.1f}<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )

    fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.7)
    for column in (1, 2):
        fig.add_annotation(
            x=0.01,
            y=50,
            xref="x domain" if column == 1 else "x2 domain",
            yref="y" if column == 1 else "y2",
            text="Threshold 50",
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            font={"size": 10, "color": "gray"},
        )
    fig.update_xaxes(title_text="Rolling-window end date", row=1, col=1)
    fig.update_xaxes(title_text="Rolling-window end date", row=1, col=2)
    fig.update_yaxes(title_text="Score", range=[0, 100], row=1, col=1)
    fig.update_yaxes(range=[0, 100], row=1, col=2)
    fig.update_layout(
        height=600,
        hovermode="x unified",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.07,
            "xanchor": "left",
            "x": 0,
        },
        margin={"t": 90, "r": 85},
    )
    st.plotly_chart(fig, width="stretch")


def attention_page(current: pd.DataFrame) -> None:
    queue = attention_queue(current)
    top_x = st.slider("Number of attention items", 5, 50, 15, 5)
    st.caption("Priority combines rule severity with a small active-user impact adjustment. Each rule and threshold is explicit in the source code.")
    st.dataframe(queue.head(top_x), hide_index=True, width="stretch")


def chat_detail_page(current: pd.DataFrame, period_start: date, period_end: date) -> None:
    options = current.dropna(subset=["title"]).sort_values("title")
    selected_title = st.selectbox("Chat", options["title"].tolist())
    row = options[options["title"] == selected_title].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Engagement", f"{row.engagement_score:.1f}" if row.eligible else "Unrated", f"{row.engagement_change:+.1f}" if row.eligible else None)
    c2.metric("Quality", f"{row.quality_score:.1f}" if row.eligible else "Unrated", f"{row.quality_change:+.1f}" if row.eligible else None)
    c3.metric("Messages", f"{row.msgs:,.0f}")
    c4.metric("Active users", f"{row.active_users:,.0f}")

    raw_metrics = {
        "Messages / active user": row.get("c11_msgs_per_active_user"),
        "Active member ratio": row.get("c6_active_member_ratio"),
        "Returning-user rate": row.get("u8_returning_user_rate"),
        "Stickiness": row.get("n4_stickiness"),
        "Reply rate": row.get("u7_reply_rate"),
        "Mention density": row.get("u10_mention_density"),
        "Participation Gini": row.get("c8_gini"),
        "Average words": row.get("u6_avg_words"),
        "Ban rate": row.get("c10_ban_rate"),
        "Newcomer rejection": row.get("c5_newcomer_rejection"),
    }
    metric_frame = pd.DataFrame({"metric": raw_metrics.keys(), "raw_value": raw_metrics.values()}).dropna()
    st.subheader("Metric evidence")
    st.dataframe(metric_frame, hide_index=True, width="stretch")

    st.subheader("Most active user IDs")
    users = load_top_users(int(row.chat_id), period_start, period_end)
    st.dataframe(users, hide_index=True, width="stretch")


def users_page(period_start: date, period_end: date, current: pd.DataFrame) -> None:
    st.caption("User IDs only; no real names or profile fields are queried.")

    chat_options = (
        current.loc[current["in_population"].fillna(False), ["chat_id", "title"]]
        .dropna(subset=["title"])
        .drop_duplicates("chat_id")
        .sort_values("title")
    )
    scope_labels = ["Entire network", *chat_options["title"].tolist()]
    scope_lookup = dict(zip(chat_options["title"], chat_options["chat_id"]))

    control_1, control_2, control_3 = st.columns([2.2, 1.4, 1])
    with control_1:
        scope = st.selectbox("Scope", scope_labels)
    with control_2:
        metric_options = {
            "Messages": "messages",
            "Active days": "active_days",
            "Replies": "replies",
            "Mentions made": "mentions",
        }
        if scope == "Entire network":
            metric_options["Active in chats"] = "chats_active_in"
        metric_label = st.selectbox("Rank by", list(metric_options))
    with control_3:
        top_n = st.selectbox("Show", [10, 20, 50], format_func=lambda value: f"Top {value}")

    metric = metric_options[metric_label]
    chat_id = None if scope == "Entire network" else int(scope_lookup[scope])
    users = load_contributors(period_start, period_end, metric, top_n, chat_id)
    if users.empty:
        st.info("No user activity was found for this period and scope.")
        return

    period_days = max((period_end - period_start).days + 1, 1)
    users = users.copy()
    users["rank"] = range(1, len(users) + 1)
    users["user_id"] = users["user_id"].astype(str)
    users["user_label"] = users["user_id"]
    users["message_share"] = users["messages"] / users["scope_messages"].replace(0, pd.NA)
    users["active_day_rate"] = users["active_days"] / period_days

    value_column = "mentions_made" if metric == "mentions" else metric
    plot_users = users.sort_values([value_column, "messages"], ascending=[False, False])
    # Plotly must be told explicitly that numeric-looking IDs are categories.
    category_order = plot_users["user_label"].tolist()[::-1]
    fig = go.Figure(
        go.Bar(
            x=plot_users[value_column],
            y=plot_users["user_label"],
            orientation="h",
            text=plot_users[value_column],
            texttemplate="%{text:,.0f}",
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(
                "<b>%{y}</b><br>"
                + metric_label
                + ": %{x:,.0f}<br>"
                "Messages: %{customdata[0]:,.0f}<br>"
                "Message share: %{customdata[1]:.1%}<extra></extra>"
            ),
            customdata=plot_users[["messages", "message_share"]],
        )
    )
    fig.update_yaxes(
        title_text="User ID",
        type="category",
        categoryorder="array",
        categoryarray=category_order,
    )
    fig.update_xaxes(title_text=metric_label, rangemode="tozero")
    fig.update_layout(
        height=max(420, len(plot_users) * 34),
        showlegend=False,
        margin={"t": 25, "r": 95, "b": 60, "l": 155},
    )
    st.subheader(f"Top contributors by {metric_label.lower()}")
    st.plotly_chart(fig, width="stretch")

    display_columns = [
        "rank",
        "user_id",
        "messages",
        "message_share",
        "active_days",
        "active_day_rate",
        "replies",
        "mentions_made",
    ]
    if scope == "Entire network":
        display_columns.append("chats_active_in")

    st.caption("❔ ⋯ Hover over a column heading for its definition and calculation.")
    st.dataframe(
        users[display_columns],
        hide_index=True,
        width="stretch",
        column_config={
            "rank": st.column_config.NumberColumn(
                "Rank ❔ ⋯",
                help="Position after sorting by the selected ranking metric. Ties are ordered by message count, then user ID.",
                format="%d",
            ),
            "user_id": st.column_config.TextColumn(
                "User ID ❔ ⋯",
                help="Stored identifier for the user. No real name or profile field is queried.",
            ),
            "messages": st.column_config.NumberColumn(
                "Messages ❔ ⋯",
                help="Total messages attributed to the user in the selected scope and period.",
                format="localized",
            ),
            "message_share": st.column_config.NumberColumn(
                "Message share ❔ ⋯",
                help="User messages divided by all messages in the selected scope and period.",
                format="percent",
            ),
            "active_days": st.column_config.NumberColumn(
                "Active days ❔ ⋯",
                help="Number of distinct calendar days on which the user was active in the selected scope and period.",
                format="%d",
            ),
            "active_day_rate": st.column_config.NumberColumn(
                "Active-day rate ❔ ⋯",
                help="Active days divided by the number of calendar days in the selected period.",
                format="percent",
            ),
            "replies": st.column_config.NumberColumn(
                "Replies ❔ ⋯",
                help="Number of the user's messages recorded as replies in the selected scope and period.",
                format="localized",
            ),
            "mentions_made": st.column_config.NumberColumn(
                "Mentions made ❔ ⋯",
                help="Number of user mentions contained in this user's messages in the selected scope and period.",
                format="localized",
            ),
            "chats_active_in": st.column_config.NumberColumn(
                "Active in chats ❔ ⋯",
                help="Number of distinct chats in which the user recorded activity during the selected period.",
                format="%d",
            ),
        },
    )


def main() -> None:
    st.title("GIG Community Health")
    st.caption(f"Scoring model {SCORING_MODEL_VERSION} · structural quality, not content quality")

    try:
        first_day, last_day = get_snapshot_bounds()
    except Exception as exc:
        st.error("The frozen snapshot could not be loaded. Check that the Parquet files are present in the data/ folder.")
        st.exception(exc)
        st.stop()

    with st.sidebar:
        st.header("Analysis period")
        score_window = st.selectbox("Rolling score window", [7, 28, 90], index=1, format_func=lambda n: f"{n} days")
        as_of = st.date_input("Ranking as of", value=last_day, min_value=first_day, max_value=last_day)
        default_start = max(first_day, as_of - timedelta(days=84))
        history = st.date_input("Evolution range", value=(default_start, as_of), min_value=first_day, max_value=last_day)
        if isinstance(history, tuple) and len(history) == 2:
            history_start, history_end = history
        else:
            history_start, history_end = default_start, as_of
        st.caption(f"Frozen, pseudonymised snapshot: {first_day} to {last_day}")

    with st.spinner("Scoring chats…"):
        current = load_scored_window(as_of, score_window)
        previous = load_scored_window(as_of - timedelta(days=score_window), score_window)
        current = add_period_change(current, previous)

    period_start = as_of - timedelta(days=score_window - 1)
    tabs = st.tabs(["Overview", "Evolution", "Attention queue", "Chat detail", "Top contributors"])
    with tabs[0]:
        overview_page(current)
    with tabs[1]:
        evolution_page(history_start, history_end, score_window, current)
    with tabs[2]:
        attention_page(current)
    with tabs[3]:
        chat_detail_page(current, period_start, as_of)
    with tabs[4]:
        users_page(period_start, as_of, current)


if __name__ == "__main__":
    main()
