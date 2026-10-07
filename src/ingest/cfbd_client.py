"""College Football Data API client with a local response cache and call accounting.

The free tier allows 1,000 calls per month, so:
  * every response is cached as raw JSON under `paths.cfbd_cache`, keyed by endpoint + params;
  * we only pull whole seasons (never per player);
  * every live call is appended to `paths.cfbd_call_log` with the X-CallLimit-Remaining header;
  * live calls stop once the remaining budget falls below `cfbd.min_remaining`.
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import UTC, datetime

import cfbd

from src.config import cfg, path, secret

log = logging.getLogger(__name__)

# endpoint name -> (Api class, method)
ENDPOINTS = {
    "player_season_stats": ("StatsApi", "get_player_season_stats"),
    "player_usage": ("PlayersApi", "get_player_usage"),
    "draft_picks": ("DraftApi", "get_draft_picks"),
    "srs": ("RatingsApi", "get_srs"),
    "games": ("GamesApi", "get_games"),
}


class BudgetExhausted(RuntimeError):
    pass


def _cache_file(endpoint: str, params: dict):
    key = "_".join(f"{k}-{v}" for k, v in sorted(params.items()))
    return path("cfbd_cache") / endpoint / f"{key}.json"


def _remaining_from_log() -> int | None:
    log_file = path("cfbd_call_log")
    if not log_file.exists():
        return None
    rows = list(csv.DictReader(open(log_file)))
    return int(rows[-1]["remaining"]) if rows and rows[-1]["remaining"] else None


def calls_remaining() -> int | None:
    return _remaining_from_log()


def fetch(endpoint: str, **params) -> list[dict]:
    """Return the JSON payload for one endpoint call, from cache when possible."""
    file = _cache_file(endpoint, params)
    if file.exists():
        return json.loads(file.read_text())

    remaining = _remaining_from_log()
    if remaining is not None and remaining < cfg()["cfbd"]["min_remaining"]:
        raise BudgetExhausted(f"Only {remaining} CFBD calls left this month; refusing to call {endpoint}.")

    api_name, method = ENDPOINTS[endpoint]
    conf = cfbd.Configuration(access_token=secret("CFBD_API_KEY"))
    with cfbd.ApiClient(conf) as client:
        api = getattr(cfbd, api_name)(client)
        resp = getattr(api, f"{method}_with_http_info")(**params, _preload_content=False)
        body = resp.raw_data
        remaining_hdr = (resp.headers or {}).get("X-CallLimit-Remaining", "")
        status = resp.status_code

    _log_call(endpoint, params, status, remaining_hdr)
    if status != 200:
        raise RuntimeError(f"CFBD {endpoint} {params} returned HTTP {status}")

    payload = json.loads(body)
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(payload))
    log.info("CFBD %s %s -> %d rows (remaining: %s)", endpoint, params, len(payload), remaining_hdr)
    return payload


def _log_call(endpoint: str, params: dict, status: int, remaining: str) -> None:
    log_file = path("cfbd_call_log")
    new = not log_file.exists()
    with open(log_file, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "endpoint", "params", "status", "remaining"])
        w.writerow([datetime.now(UTC).isoformat(), endpoint, json.dumps(params), status, remaining])
