"""Plotly animation of one pass play: tracked defenders joined to the targeted receiver, ball landing point marked."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from src.models.evaluate import ATHLETIC, INK, MARKET, MODEL, MUTED
from src.tracking.animate import _field
from src.tracking.data import FIELD_LENGTH, FIELD_WIDTH

KEYS = ["game_id", "play_id"]


def _traces(f: pd.DataFrame) -> list[go.Scatter]:
    def dots(df, name, color, size=11, text=None, **kw):
        return go.Scatter(
            x=df["x"], y=df["y"], mode="markers", name=name, text=text if text is not None else df["player_name"],
            hoverinfo="text", marker=dict(color=color, size=size, line=dict(color="white", width=1), **kw),
        )  # fmt: skip

    tracked = f[(f["role"] == "defense") & f["predicted"]]
    other_d = f[(f["role"] == "defense") & ~f["predicted"]]
    target = f[f["role"] == "target"]
    lines_x, lines_y = [], []
    for r in target.itertuples():
        for d in tracked.itertuples():
            lines_x += [d.x, r.x, None]
            lines_y += [d.y, r.y, None]
    land = f.iloc[0]
    coe = tracked["coe"]
    return [
        go.Scatter(
            x=lines_x,
            y=lines_y,
            mode="lines",
            line=dict(color=MUTED, width=1, dash="dot"),
            hoverinfo="skip",
            showlegend=False,
        ),  # fmt: skip
        dots(f[f["role"] == "offense"], "Offense", MODEL, 9),
        dots(other_d, "Other defense", "#f6d5c6", 9),
        dots(
            tracked,
            "Tracked defender",
            MARKET,
            13,
            text=tracked["player_name"] + " (COE " + coe.round(1).astype(str) + " yd)",
        ),  # fmt: skip
        dots(target, "Targeted receiver", ATHLETIC, 15),
        go.Scatter(
            x=[land["land_x"]],
            y=[land["land_y"]],
            mode="markers",
            name="Ball landing point",
            marker=dict(color=INK, size=11, symbol="star"),
        ),  # fmt: skip
    ]


def animate_closing(frames: pd.DataFrame, game_id: int, play_id: int) -> go.Figure:
    """Animated figure for one play from an `animation_frames` table (offense moves left to right; t=0 is the throw)."""
    play = frames[(frames["game_id"] == game_id) & (frames["play_id"] == play_id)]
    if play.empty:
        raise ValueError(f"No frames for game {game_id}, play {play_id}")
    ids = sorted(play["t"].unique())
    start = 0 if 0 in ids else ids[0]
    by_t = {i: g for i, g in play.groupby("t")}
    fig = go.Figure(data=_traces(by_t[start]), frames=[go.Frame(data=_traces(by_t[i]), name=str(i)) for i in ids])
    now = dict(frame=dict(duration=0), mode="immediate")
    play_btn = dict(label="Play", method="animate", args=[None, dict(frame=dict(duration=100), fromcurrent=True)])
    steps = [dict(label=f"{i / 10:+.1f}s", method="animate", args=[[str(i)], now]) for i in ids]
    pad = 6
    fig.update_layout(
        shapes=_field(), height=460, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white",
        xaxis=dict(range=[max(play["x"].min() - pad, 0), min(play["x"].max() + pad, FIELD_LENGTH)], visible=False),
        yaxis=dict(range=[0, FIELD_WIDTH], visible=False, scaleanchor="x"),
        legend=dict(orientation="h", y=-0.05),
        updatemenus=[dict(type="buttons", showactive=False, x=0.02, y=1.0,
                          buttons=[play_btn, dict(label="Pause", method="animate", args=[[None], now])])],
        sliders=[dict(steps=steps, active=ids.index(start), currentvalue=dict(prefix="Time vs. throw: "))],
    )  # fmt: skip
    return fig
