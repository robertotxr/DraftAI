"""College <-> NFL player identity resolution.

Strategy, in order of confidence:
  1. `cfbd_draft_id`: nflverse pick (season, overall pick) joins the CFBD draft table, which carries
     the CFBD `collegeAthleteId`; accepted when that id appears in the player's pre-draft stat lines.
  2. `name_school`: normalized name + normalized school among stat lines from the five seasons
     before the draft. Ties are broken by the most recent season; ties that survive are marked
     ambiguous and left unmatched rather than guessed.
Every pick gets a row with the method used (or `unmatched`) so the match rate is auditable.
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
SCHOOL_ALIASES = {
    "miami fl": "miami",
    "miami (fl)": "miami",
    "miami oh": "miami (oh)",
    "southern cal": "usc",
    "southern california": "usc",
    "ole miss": "ole miss",
    "mississippi": "ole miss",
    "lsu": "lsu",
    "louisiana state": "lsu",
    "tcu": "tcu",
    "texas christian": "tcu",
    "smu": "smu",
    "southern methodist": "smu",
    "ucf": "ucf",
    "central florida": "ucf",
    "byu": "byu",
    "brigham young": "byu",
    "pittsburgh": "pittsburgh",
    "pitt": "pittsburgh",
    "nc state": "nc state",
    "north carolina state": "nc state",
    "utsa": "utsa",
    "texas-san antonio": "utsa",
    "uab": "uab",
    "alabama-birmingham": "uab",
    "unlv": "unlv",
    "nevada-las vegas": "unlv",
    "la-monroe": "ul monroe",
    "louisiana-monroe": "ul monroe",
    "louisiana-lafayette": "louisiana",
    "la-lafayette": "louisiana",
    "hawaii": "hawai'i",
    "san jose state": "san josé state",
}


def norm_name(name: str | None) -> str:
    if not isinstance(name, str):
        return ""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[.'`’]", "", s)
    s = re.sub(r"[^a-z ]", " ", s)
    tokens = [t for t in s.split() if t not in SUFFIXES]
    return " ".join(tokens)


def norm_school(school: str | None) -> str:
    if not isinstance(school, str):
        return ""
    s = school.lower().strip()
    s = re.sub(r"\bst\.?(?=\s|$)", "state", s)  # "Florida St." -> "florida state"
    s = s.replace("&", "and")
    return SCHOOL_ALIASES.get(s, s)


def resolve(picks: pd.DataFrame, cfbd_draft: pd.DataFrame, stat_players: pd.DataFrame) -> pd.DataFrame:
    """Link every nflverse draft pick to a CFBD college player id.

    picks:        season, pick, pfr_player_id, pfr_player_name, college
    cfbd_draft:   year, overall, collegeAthleteId, name, collegeTeam
    stat_players: player_id, player, team, season  (one row per player-season-team in CFBD stats)
    """
    p = picks[["season", "pick", "pfr_player_id", "pfr_player_name", "college"]].merge(
        cfbd_draft[["year", "overall", "collegeAthleteId", "name", "collegeTeam"]],
        left_on=["season", "pick"],
        right_on=["year", "overall"],
        how="left",
    )
    p["name_key"] = p["pfr_player_name"].map(norm_name)
    p["cfbd_name_key"] = p["name"].map(norm_name)
    # A CFBD row at the same pick with a different name means the two sources disagree; don't trust its id.
    same_person = (p["name_key"] == p["cfbd_name_key"]) | _last_name_match(p["name_key"], p["cfbd_name_key"])
    p.loc[~same_person, "collegeAthleteId"] = pd.NA
    p["school_key"] = p["collegeTeam"].fillna(p["college"]).map(norm_school)

    sp = stat_players.copy()
    sp["name_key"] = sp["player"].map(norm_name)
    sp["school_key"] = sp["team"].map(norm_school)
    by_name = dict(tuple(sp.groupby("name_key")))
    ids_by_season = sp.groupby("player_id")["season"].agg(["min", "max"])

    out = []
    for r in p.itertuples(index=False):
        lo, hi = r.season - 5, r.season - 1
        cid = r.collegeAthleteId
        if pd.notna(cid) and str(int(cid)) in ids_by_season.index:
            s_min, s_max = ids_by_season.loc[str(int(cid))]
            if s_max >= lo and s_min <= hi:
                out.append((r.season, r.pick, str(int(cid)), "cfbd_draft_id"))
                continue
        cand = by_name.get(r.name_key, sp.iloc[:0])
        cand = cand[cand.season.between(lo, hi)]
        if cand.empty:
            out.append((r.season, r.pick, None, "unmatched"))
            continue
        by_school = cand[cand.school_key == r.school_key]
        pool = by_school if not by_school.empty else cand
        ids = pool.groupby("player_id")["season"].max().sort_values(ascending=False)
        if len(ids) == 1 or (ids.iloc[0] > ids.iloc[1]):
            method = "name_school" if not by_school.empty else "name_only_recent"
            out.append((r.season, r.pick, ids.index[0], method))
        else:
            out.append((r.season, r.pick, None, "ambiguous"))
    res = pd.DataFrame(out, columns=["season", "pick", "college_player_id", "match_method"])
    return p[["season", "pick", "pfr_player_id", "pfr_player_name", "school_key", "collegeTeam"]].merge(
        res, on=["season", "pick"]
    )


def _last_name_match(a: pd.Series, b: pd.Series) -> pd.Series:
    """Accept nickname differences (e.g. 'Mike' vs 'Michael') when last names agree."""
    last = lambda s: s.fillna("").str.split().str[-1]  # noqa: E731
    first = lambda s: s.fillna("").str[:1]  # noqa: E731
    return (last(a) == last(b)) & (first(a) == first(b)) & (a.fillna("") != "")
