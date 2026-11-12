"""Forecaster for make.fun roblox-up-and-coming-forecast-01.

Reads /input/history.ndjson (hourly-ish snapshots of Roblox's Up-and-Coming
chart) + /input/task.json, writes /output/forecast.json:
  {"probabilities": [{"universe_id", "place", "p"}], "top10": [10 ids]}

Model (stdlib only, deterministic, no network, no model calls):
- Every game on the last chart gets a rank process: its rank at horizon h
  hours is modelled as Normal(mean = r0 + drift*h, sd = vol(h)), with drift
  and volatility estimated from the history's consecutive-read transitions
  (pooled by starting-rank bucket, volatility growing like sqrt(h)).
- A game's probability of holding place k is the time-average over the
  scored window's hourly slots of P(rank in [k-0.5, k+0.5]) -- the optimal
  single forecast under log score, since the score averages over the reads.
- Games age out of the chart 28 days after publication: when a created date
  is known (house reads carry details.created) the mass is zeroed after
  created+28d; otherwise first-seen on the chart is used as a proxy.
- A share of every place is left unnamed so new climbers (games not named
  at that place, including ones that join the chart during the window) get
  the unnamed share instead of the floor.
"""
import json
import math
import os
from datetime import datetime, timedelta, timezone

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
PLACES = 10
CHART_AGE_DAYS = 28

# Tunables (calibrated on the practice backtest; overridable via env for sweeps).
UNNAMED = float(os.environ.get("MF_UNNAMED", "0.10"))
VOL0 = float(os.environ.get("MF_VOL0", "1.6"))
VOL1 = float(os.environ.get("MF_VOL1", "0.85"))
DRIFT_SHRINK = float(os.environ.get("MF_DRIFT_SHRINK", "0"))
MIN_P = float(os.environ.get("MF_MIN_P", "0.03"))
MAX_ROWS_PER_PLACE = 100


def parse_time(s):
dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    # Robustness: a timestamp without timezone info would crash the run
    # later (aware - naive subtraction raises TypeError), zeroing the whole
    # entry. House reads always carry Z, but assume UTC if one ever lacks it.
    if dt.tzinfo is None:
              dt = dt.replace(tzinfo=timezone.utc)
          return dt


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def load_history():
    task = json.load(open(os.path.join(INPUT_DIR, "task.json"), encoding="utf-8"))
    snaps = []
    with open(os.path.join(INPUT_DIR, "history.ndjson"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                snaps.append(json.loads(line))
    snaps.sort(key=lambda s: s["at"])
    return task, snaps


def build_series(snaps):
    """Per-game time series: uid -> list of (t, rank, playing), plus meta."""
    series = {}
    first_seen, last_seen, created = {}, {}, {}
    names = {}
    for s in snaps:
        t = parse_time(s["at"])
        details = s.get("details") or {}
        for g in s.get("up_and_coming") or []:
            uid = g["universe_id"]
            series.setdefault(uid, []).append(
                (t, g["rank"], g.get("playing")))
            names[uid] = g.get("name")
            first_seen.setdefault(uid, t)
            last_seen[uid] = t
            d = details.get(str(uid)) or details.get(uid)
            if d and d.get("created"):
                created[uid] = parse_time(d["created"])
    return series, first_seen, last_seen, created, names


def pooled_stats(snaps):
    """Estimate per-hour rank drift and volatility from consecutive reads."""
    # bucket start rank -> list of (gap_h, drank)
    trans = []
    for a, b in zip(snaps, snaps[1:]):
        ta, tb = parse_time(a["at"]), parse_time(b["at"])
        gap = (tb - ta).total_seconds() / 3600.0
        if gap <= 0 or gap > 48:
            continue
        ra = {g["universe_id"]: g["rank"] for g in a.get("up_and_coming") or []}
        rb = {g["universe_id"]: g["rank"] for g in b.get("up_and_coming") or []}
        for uid, r0 in ra.items():
            if uid in rb:
                trans.append((r0, rb[uid] - r0, gap))
    # pooled per-hour drift by start-rank bucket
    buckets = [(1, 1), (2, 3), (4, 5), (6, 10), (11, 20), (21, 40), (41, 200)]
    drift = {}
    for lo, hi in buckets:
        xs = [(dr / g) for (r0, dr, g) in trans if lo <= r0 <= hi]
        drift[(lo, hi)] = sum(xs) / len(xs) if xs else 0.0
    # volatility: sd of drank vs gap -> fit sd = VOL0 + VOL1*sqrt(gap)
    # use grouped sd at several gaps, least squares on (sqrt(gap), sd)
    by_gap = {}
    for (r0, dr, g) in trans:
        by_gap.setdefault(round(g), []).append(dr)
    pts = []
    for g, drs in sorted(by_gap.items()):
        if len(drs) >= 8:
            m = sum(drs) / len(drs)
            var = sum((x - m) ** 2 for x in drs) / len(drs)
            pts.append((math.sqrt(g), math.sqrt(var)))
    if len(pts) >= 2:
        n = len(pts)
        sx = sum(p[0] for p in pts)
        sy = sum(p[1] for p in pts)
        sxx = sum(p[0] ** 2 for p in pts)
        sxy = sum(p[0] * p[1] for p in pts)
        den = n * sxx - sx * sx
        if den:
            b = (n * sxy - sx * sy) / den
            a = (sy - b * sx) / n
            return drift, max(0.5, a), min(2.5, max(0.2, b))
    return drift, VOL0, VOL1


def bucket_drift(drift, r0):
    for (lo, hi), v in drift.items():
        if lo <= r0 <= hi:
            return v
    return 0.0


def main():
    task, snaps = load_history()
    places = int(task.get("places", PLACES))
    window = task["window"]
    w_from = parse_time(window["from"])
    slot_min = int(window.get("slot_minutes", 60))
    n_slots = int(window.get("slots", 72))
    slot_times = [w_from + timedelta(minutes=slot_min * i) for i in range(n_slots)]

    series, first_seen, last_seen, created, names = build_series(snaps)
    drift_map, _fit0, _fit1 = pooled_stats(snaps)
    # Tuned on the practice backtest (the pooled fit is dominated by deep-rank
    # games and over-disperses): fixed defaults, env overrides only for sweeps.
    vol0 = float(os.environ["MF_VOL0"]) if "MF_VOL0" in os.environ else 0.5
    vol1 = float(os.environ["MF_VOL1"]) if "MF_VOL1" in os.environ else 1.0

    last = snaps[-1]
    t_last = parse_time(last["at"])
    last_chart = {g["universe_id"]: g for g in last.get("up_and_coming") or []}
    chart_size = max(len(last_chart), 90)

    # Per-game recent drift from its own trajectory (shrunk toward pooled).
    game_params = {}
    for uid, pts in series.items():
        r0 = pts[-1][1]
        own = 0.0
        if len(pts) >= 3:
            # rank change per hour over the last few reads
            (t1, r1, _), (t2, r2, _) = pts[-2], pts[-1]
            g = max((t2 - t1).total_seconds() / 3600.0, 1.0)
            own = (r2 - r1) / g
            own = max(-3.0, min(3.0, own))
        pooled = bucket_drift(drift_map, r0)
        drift_h = DRIFT_SHRINK * own + (1.0 - DRIFT_SHRINK) * pooled
        # games not on the last chart: they must climb back; start from last rank
        absent_h = (t_last - last_seen[uid]).total_seconds() / 3600.0
        game_params[uid] = {
            "r0": r0,
            "drift": drift_h,
            "absent_h": absent_h,
            "first_seen": first_seen[uid],
            "created": created.get(uid),
        }

    # Candidate set: every game seen recently (last chart + games seen in the
    # last 3 reads, they can climb back in).
    recent_uids = set(last_chart)
    for s in snaps[-3:]:
        for g in s.get("up_and_coming") or []:
            recent_uids.add(g["universe_id"])

    # prob[uid][place] averaged over slots
    prob = {uid: [0.0] * (places + 1) for uid in recent_uids}
    for h_idx, st in enumerate(slot_times):
        h = (st - t_last).total_seconds() / 3600.0
        if h < 0:
            h = 0.0
        sd = vol0 + vol1 * math.sqrt(h)
        for uid in recent_uids:
            gp = game_params[uid]
            # age-out: 28 days after publication
            pub = gp["created"] or gp["first_seen"]
            age_out_at = pub + timedelta(days=CHART_AGE_DAYS)
            if st >= age_out_at:
                continue
            # extra decay for games already off the chart
            if uid not in last_chart:
                # probability of re-entering fades with absence
                if gp["absent_h"] + h > 72:
                    continue
            mean = gp["r0"] + gp["drift"] * h
            # P(rank in [k-0.5, k+0.5]) for k=1..places; rank>=1
            for k in range(1, places + 1):
                lo = (k - 0.5 - mean) / sd
                hi = (k + 0.5 - mean) / sd
                if k == 1:
                    lo = float("-inf")
                p = norm_cdf(hi) - (norm_cdf(lo) if lo != float("-inf") else 0.0)
                prob[uid][k] += p / n_slots

    # re-entry fade for games not on the last chart (single factor)
    for uid in recent_uids:
        if uid not in last_chart:
            gp = game_params[uid]
            fade = math.exp(-gp["absent_h"] / 36.0)
            for k in range(1, places + 1):
                prob[uid][k] *= fade

    # Scale each place so named mass = 1 - UNNAMED, then emit rows.
    probabilities = []
    for k in range(1, places + 1):
        tot = sum(prob[uid][k] for uid in recent_uids)
        if tot <= 0:
            continue
        scale = (1.0 - UNNAMED) / tot
        rows = sorted(
            ((prob[uid][k] * scale, uid) for uid in recent_uids),
            reverse=True,
        )[:MAX_ROWS_PER_PLACE]
        for p, uid in rows:
            if p >= MIN_P:
                probabilities.append(
                    {"universe_id": int(uid), "place": k, "p": round(p, 6)})

    # top10: order by expected rank (lower is better), tie-break by P(top10)
    def expected_rank(uid):
        ex = 0.0
        p_in = 0.0
        for k in range(1, places + 1):
            ex += prob[uid][k] * k
            p_in += prob[uid][k]
        # mass outside top 10 counts as rank ~ chart_size/2 + 10
        ex += (1.0 - p_in) * (chart_size / 2.0 + 10)
        return ex

    top10 = sorted(recent_uids, key=expected_rank)[:places]
    top10 = [int(u) for u in top10]

    out = {"probabilities": probabilities, "top10": top10}
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, task.get("output", "forecast.json")), "w", encoding="utf-8") as f:
        json.dump(out, f)
        f.write("\n")


if __name__ == "__main__":
    main()
