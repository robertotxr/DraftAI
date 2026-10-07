import pandas as pd

from src.ingest.identity import norm_name, norm_school, resolve


def test_norm_name():
    assert norm_name("Marvin Harrison Jr.") == "marvin harrison"
    assert norm_name("Odell Beckham III") == "odell beckham"
    assert norm_name("D'Andre  Swift") == "dandre swift"
    assert norm_name("Josh Jacobs-Smith") == "josh jacobs smith"
    assert norm_name("José Núñez") == "jose nunez"
    assert norm_name(None) == "" and norm_name(float("nan")) == ""


def test_norm_school():
    assert norm_school("Florida St.") == "florida state"
    assert norm_school("Florida St") == "florida state"
    assert norm_school("Southern Cal") == "usc"
    assert norm_school("Ole Miss") == norm_school("Mississippi") == "ole miss"
    assert norm_school("Miami (FL)") == "miami"
    assert norm_school("Texas & Tech") == "texas and tech"
    assert norm_school(None) == ""


def _pick(season=2020, pick=1, name="Joe Burrow", college="LSU", pid="p1"):
    return {"season": season, "pick": pick, "pfr_player_id": pid, "pfr_player_name": name, "college": college}


def _cfbd(year=2020, overall=1, cid=100, name="Joe Burrow", team="LSU"):
    return {"year": year, "overall": overall, "collegeAthleteId": cid, "name": name, "collegeTeam": team}


def _stat(pid, name, team, season=2019):
    return {"player_id": pid, "player": name, "team": team, "season": season}


def _run(picks, cfbd, stats):
    cols = ["year", "overall", "collegeAthleteId", "name", "collegeTeam"]
    return resolve(pd.DataFrame(picks), pd.DataFrame(cfbd, columns=cols), pd.DataFrame(stats)).iloc[0]


def test_cfbd_id_path():
    r = _run([_pick()], [_cfbd()], [_stat("100", "Joe Burrow", "LSU")])
    assert (r.college_player_id, r.match_method) == ("100", "cfbd_draft_id")


def test_name_school_fallback_without_cfbd_row():
    r = _run([_pick(college="Florida St.", name="Jr. Smith Jr.")], [], [_stat("7", "Jr. Smith", "Florida State")])
    assert (r.college_player_id, r.match_method) == ("7", "name_school")


def test_same_name_different_schools_resolved_by_school():
    stats = [_stat("1", "Mike Williams", "Clemson"), _stat("2", "Mike Williams", "USC")]
    r = _run([_pick(name="Mike Williams", college="USC")], [], stats)
    assert (r.college_player_id, r.match_method) == ("2", "name_school")


def test_truly_ambiguous_is_left_unmatched():
    stats = [_stat("1", "Mike Williams", "Clemson"), _stat("2", "Mike Williams", "Clemson")]
    r = _run([_pick(name="Mike Williams", college="Clemson")], [], stats)
    assert r.college_player_id is None and r.match_method == "ambiguous"


def test_disagreeing_names_at_same_pick_do_not_trust_cfbd_id():
    stats = [_stat("100", "Someone Else", "LSU"), _stat("200", "Joe Burrow", "LSU")]
    r = _run([_pick()], [_cfbd(cid=100, name="Someone Else")], stats)
    assert (r.college_player_id, r.match_method) == ("200", "name_school")


def test_unmatched():
    r = _run([_pick()], [], [_stat("9", "Other Guy", "LSU")])
    assert r.match_method == "unmatched"
