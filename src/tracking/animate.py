"""Plotly animation of one play: players by team, ball carrier highlighted, defenders sized by tackle probability."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.models.evaluate import ATHLETIC, INK, MARKET, MODEL, MUTED
from src.tracking.data import FIELD_LENGTH, FIELD_WIDTH, KEYS
from src.tracking.metric import active_window

COLUMNS = [*KEYS, "frameId", "nflId", "displayName", "club", "role", "x", "y", "p_tackle"]


def animation_frames(track: pd.DataFrame, plays: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
    """Tracking rows (incl. ball) for the plays in `track`, with role and the model's `proba` as p_tackle.

    `track` is standardized tracking already limited to the plays to keep; `frames` is defender_frames + proba.
    """
    p = plays[KEYS + ["ballCarrierId", "possessionTeam", "defensiveTeam"]]
    t = active_window(track.merge(p, on=KEYS))
    role = np.select(
        [t["club"] == "football", t["nflId"] == t["ballCarrierId"], t["club"] == t["possessionTeam"]],
        ["ball", "carrier", "offense"],
        "defense",
    )
    t = t.assign(role=role)
    t = t.merge(frames[KEYS + ["frameId", "nflId", "proba"]], on=[*KEYS, "frameId", "nflId"], how="left")
    return t.rename(columns={"proba": "p_tackle"})[COLUMNS].reset_index(drop=True)


def _field() -> list[dict]:
    shapes = [dict(type="rect", x0=0, x1=FIELD_LENGTH, y0=0, y1=FIELD_WIDTH, line=dict(color=MUTED), layer="below")]
    for x0 in (0, 110):
        shapes.append(
            dict(type="rect", x0=x0, x1=x0 + 10, y0=0, y1=FIELD_WIDTH, fillcolor="#f1f0ec", line_width=0, layer="below")
        )
    for x in range(10, 111, 5):
        shapes.append(
            dict(type="line", x0=x, x1=x, y0=0, y1=FIELD_WIDTH, line=dict(color="#d9d8d3", width=1), layer="below")
        )
    return shapes


def _traces(f: pd.DataFrame) -> list[go.Scatter]:
    def dots(df, name, color, size=11, **kw):
        return go.Scatter(
            x=df["x"], y=df["y"], mode="markers", name=name, text=df["displayName"], hoverinfo="text",
            marker=dict(color=color, size=size, line=dict(color="white", width=1), **kw),
        )  # fmt: skip

    d = f[f["role"] == "defense"]
    p = d["p_tackle"].fillna(0)
    return [
        dots(f[f["role"] == "offense"], "Offense", MODEL),
        go.Scatter(
            x=d["x"],
            y=d["y"],
            mode="markers",
            name="Defense",
            text=d["displayName"] + " P=" + p.round(2).astype(str),
            hoverinfo="text",
            marker=dict(
                color=p,
                cmin=0,
                cmax=1,
                colorscale=[[0, "#f6d5c6"], [1, MARKET]],
                size=9 + 26 * p,
                line=dict(color="white", width=1),
            ),
        ),  # fmt: skip
        dots(f[f["role"] == "carrier"], "Ball carrier", ATHLETIC, 16),
        go.Scatter(
            x=f[f["role"] == "ball"]["x"],
            y=f[f["role"] == "ball"]["y"],
            mode="markers",
            name="Ball",
            marker=dict(color=INK, size=6, symbol="diamond"),
        ),  # fmt: skip
    ]


def animate_play(frames: pd.DataFrame, game_id: int, play_id: int) -> go.Figure:
    """Animated figure for one play from an `animation_frames` table (offense moves left to right)."""
    play = frames[(frames["gameId"] == game_id) & (frames["playId"] == play_id)]
    if play.empty:
        raise ValueError(f"No frames for game {game_id}, play {play_id}")
    ids = sorted(play["frameId"].unique())
    by_frame = {i: g for i, g in play.groupby("frameId")}
    fig = go.Figure(
        data=_traces(by_frame[ids[0]]),
        frames=[go.Frame(data=_traces(by_frame[i]), name=str(i)) for i in ids],
    )
    now = dict(frame=dict(duration=0), mode="immediate")
    play_btn = dict(label="Play", method="animate", args=[None, dict(frame=dict(duration=100), fromcurrent=True)])
    steps = [dict(label=str(i), method="animate", args=[[str(i)], now]) for i in ids]
    fig.update_layout(
        shapes=_field(), height=460, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white",
        xaxis=dict(range=[0, FIELD_LENGTH], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[0, FIELD_WIDTH], showgrid=False, zeroline=False, visible=False, scaleanchor="x"),
        legend=dict(orientation="h", y=-0.05),
        updatemenus=[dict(type="buttons", showactive=False, x=0.02, y=1.0,
                          buttons=[play_btn, dict(label="Pause", method="animate", args=[[None], now])])],
        sliders=[dict(steps=steps, currentvalue=dict(prefix="Frame "))],
    )  # fmt: skip
    return fig
