"""gta6-twitch-ccu-forecast-01 entry (Python 3.13, standard library only).

Forecasts hours 2..24 of a big game launch's release day on Twitch, as five
quantiles per hour, from the case's hour-1 print and its comparables (past big
launches' hourly curves), fitting everything at run time from the case itself:

  1. Donor ensemble: each comparable's per-hour log-ratio curve (hour h viewers
     over its own hour-1 viewers, UTC release day), weighted by resemblance to
     the case: a scale kernel on log hour-1, an unlock-structure match (rolling
     midnight-local launches resemble this one; single-moment global unlocks
     are cut hard), early-access and late-PC-wave status, a console-only match
     and an era decay.
  2. A ridge regression layer, one fit per forecast hour, trained on the donors
     themselves (each donor's features are its leave-one-out ensemble centre
     and dispersion, its scale, its pre-release ramp level and peak, its
     early-access and late-PC-wave status and its unlock position; the ridge
     penalty is picked per hour by the closed-form leave-one-out shortcut).
  3. A shrunk bias correction from the donors' leave-one-out residuals.
  4. A spread from the leave-one-out residual dispersion, with its multiplier
     calibrated on the donors so the 5%-95% band covers about 90%.

Forecast = hour-1 print x exp(centre + z_tau x spread). Any failure in the
model path falls back per case to the persistence baseline, so the output is
always valid: one key, 23 rows, five non-decreasing finite quantiles above 0.
Deterministic: pure functions of the input file, no randomness, no network.
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]
HOURS = list(range(2, 25))

# Hyperparameters (tuned on the leave-one-out loop over the launch comparables).
PARAMS = {
    "sigma_scale": 0.9,      # width of the log hour-1 scale kernel
    "tau_ramp": 100.0,       # ramp-slope weight factor: disabled by the LOO (the
                             # slope saturates for dormant categories, so the
                             # distance carried no signal); slope stays a feature
    "unlock_unknown": 0.4,   # weight factor, donor unlock class unknown vs case
    "unlock_global": 0.15,   # weight factor, donor global unlock vs rolling case
    "ea_mismatch": 0.55,     # weight factor, donor early-access status differs
    "pc_mismatch": 0.7,      # weight factor, donor late-PC-wave status differs
    "console_match": 1.6,    # console-only donor for a console-only case
    "console_mismatch": 0.6,  # PC donor for a console-only case
    "era_halflife": 2.0,     # years for the era decay to halve a weight
    "lambdas": [0.3, 1.0, 3.0, 10.0, 30.0, 100.0],  # ridge penalties, LOO-picked
    "bias_kappa": 8.0,       # shrinkage on the per-hour bias estimate
    "centre_smooth": 0.25,   # neighbour smoothing of predicted curves (LOO-tested)
    "feat_late": 0.0,        # late-ramp feature: LOO-neutral, dropped from the fit
    "feat_vch": 0.0,         # viewers-per-channel feature: LOO-harmful, dropped
    "spread_shrink": 6.0,    # shrinkage of per-hour residual SD toward the global SD
    "spread_floor": 0.10,    # never let an hour's spread collapse below this
    "m_grid": [0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4],  # spread multipliers tried
    "min_donors": 10,
}


def parse_ts(s):
    return datetime.fromisoformat(str(s).replace("Z", "+00:00"))


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def unlock_class(text):
    """Classify a launch's unlock structure from its notes text."""
    t = (text or "").lower()
    if "midnight local" in t:
        return "rolling"
    if "single global" in t or "simultaneous global" in t or "single simultaneous" in t or "global unlock" in t:
        return "global"
    if "rolling" in t:
        return "rolling"
    if "simultaneous" in t:
        return "global"
    return "unknown"


def unlock_hour(text, cls):
    """Hour index (1..24) holding the launch's main (US / global) unlock moment."""
    times = [int(m) for m in re.findall(r"(\d{1,2}):00Z", text or "")]
    if cls == "rolling":
        for hh in times:
            if hh in (4, 5):  # US Eastern midnight, EDT or EST
                return hh + 1
        return 6
    if cls == "global":
        if times:
            return clamp(times[0] + 1, 1, 24)
        return 1
    return 6


def pc_unlock(text, day_start):
    """(pc_late, pc_hour): a PC/Steam unlock landing well inside the release day
    is a second audience wave; pc_hour is its hour index (0 = no late PC wave)."""
    t = (text or "").lower()
    if "pc unlock later" in t:
        return 1, 12
    m = re.search(r"(pc|steam)[^.;]*?(\d{4})-(\d{2})-(\d{2})t(\d{2}):00z", t)
    if m:
        y, mo, dd, hh = (int(m.group(i)) for i in (2, 3, 4, 5))
        if (y, mo, dd) < (day_start.year, day_start.month, day_start.day):
            return 0, 0  # PC unlocked before the release day: in from hour 1
        if (y, mo, dd) == (day_start.year, day_start.month, day_start.day) and hh >= 2:
            return 1, hh + 1
        return 0, 0
    m = re.search(r"(pc|steam)[^.;]*?(\d{1,2}):00z", t)
    if m:
        hh = int(m.group(2))
        if hh >= 2:
            return 1, hh + 1
    return 0, 0


def hourly_series(points, key="from"):
    """{datetime hour-start: viewers} from a list of hourly points."""
    out = {}
    for pt in points or []:
        try:
            v = pt.get("viewers")
            if v is None:
                continue
            out[parse_ts(pt[key])] = float(v)
        except Exception:
            continue
    return out


def window_mean(series, start, end):
    vals = [v for t, v in series.items() if start <= t < end]
    if len(vals) < 6:
        return None
    return sum(vals) / len(vals)


def ramp_features(series, day_start):
    """(ramp_slope, pre24_mean) from the pre-release windows, Nones if unknown."""
    pre24 = window_mean(series, day_start - timedelta(hours=24), day_start)
    pre72 = window_mean(series, day_start - timedelta(hours=72), day_start - timedelta(hours=24))
    if pre24 is None:
        return None, None
    slope = None
    if pre72 is not None:
        slope = clamp(math.log(pre24 + 1.0) - math.log(pre72 + 1.0), -3.0, 3.0)
    return slope, pre24


def build_donors(case):
    """Launch comparables with a full UTC release day, as donor dicts."""
    donors = []
    for cp in case.get("comparables") or []:
        try:
            ds = parse_ts(cp["day_start"])
            series = hourly_series(cp.get("hourly"))
            day = {}
            for k in range(24):
                v = series.get(ds + timedelta(hours=k))
                if v is None:
                    break
                day[k + 1] = v
            if len(day) < 24 or day[1] <= 0:
                continue
            log_h1 = math.log(day[1])
            r = {h: math.log(day[h]) - log_h1 for h in HOURS}
            slope, pre24 = ramp_features(series, ds)
            ratio = None
            if pre24 is not None:
                ratio = clamp(math.log(pre24 + 1.0) - log_h1, -8.0, 1.0)
            pre72 = [v for t, v in series.items() if ds - timedelta(hours=72) <= t < ds]
            prepk = None
            if pre72:
                prepk = clamp(math.log(max(pre72) + 1.0) - log_h1, -3.0, 3.5)
            pre6 = window_mean(series, ds - timedelta(hours=6), ds)
            pre18 = window_mean(series, ds - timedelta(hours=24), ds - timedelta(hours=6))
            ramp_late = None
            if pre6 is not None and pre18 is not None:
                ramp_late = clamp(math.log(pre6 + 1.0) - math.log(pre18 + 1.0), -3.0, 3.0)
            ch1 = None
            for pt in cp.get("hourly") or []:
                if pt.get("from") == cp.get("day_start"):
                    ch1 = pt.get("channels")
            vch = None
            if ch1:
                vch = clamp(math.log(day[1] / ch1), 0.0, 8.0)
            cls = unlock_class(cp.get("unlock"))
            platforms = cp.get("platforms") or ""
            unlock_txt = (cp.get("unlock") or "").lower()
            pc_late, pc_hour = pc_unlock(cp.get("unlock"), ds)
            donors.append({
                "name": cp.get("name", "?"),
                "log_h1": log_h1,
                "r": r,
                "ramp_slope": slope,
                "ramp_ratio": ratio,
                "prepk": prepk,
                "ramp_late": ramp_late,
                "vch": vch,
                "ea": "early access" in unlock_txt and "no early access" not in unlock_txt,
                "pc_late": pc_late,
                "pc_hour": pc_hour,
                "unlock_class": cls,
                "u": unlock_hour(cp.get("unlock"), cls),
                "year": ds.year,
                "console_only": "PC" not in platforms,
            })
        except Exception:
            continue
    return donors


def case_features(case, p1, ch1=None):
    """The target's features, mirroring the donor feature definitions."""
    ds = parse_ts(case["day_start"])
    series = hourly_series(case.get("history_hourly"))
    if len(series) < 6:
        pts = [{"from": p["at"], "viewers": p.get("viewers")} for p in case.get("history") or []]
        series = hourly_series(pts)
    slope, pre24 = ramp_features(series, ds)
    ratio = None
    if pre24 is not None:
        ratio = clamp(math.log(pre24 + 1.0) - math.log(p1), -8.0, 1.0)
    pre72 = [v for t, v in series.items() if ds - timedelta(hours=72) <= t < ds]
    prepk = None
    if pre72:
        prepk = clamp(math.log(max(pre72) + 1.0) - math.log(p1), -3.0, 3.5)
    pre6 = window_mean(series, ds - timedelta(hours=6), ds)
    pre18 = window_mean(series, ds - timedelta(hours=24), ds - timedelta(hours=6))
    ramp_late = None
    if pre6 is not None and pre18 is not None:
        ramp_late = clamp(math.log(pre6 + 1.0) - math.log(pre18 + 1.0), -3.0, 3.0)
    vch = None
    if ch1:
        vch = clamp(math.log(p1 / ch1), 0.0, 8.0)
    notes = str(case.get("notes") or "")
    cls = unlock_class(notes)
    if cls == "unknown":
        cls = "rolling"  # every case in this competition rolls at midnight local
    pc_late, pc_hour = pc_unlock(notes, ds)
    low = notes.lower()
    console_case = any(w in low for w in ("ps5", "ps4", "playstation", "xbox"))
    return {
        "log_h1": math.log(p1),
        "ramp_slope": slope,
        "ramp_ratio": ratio,
        "prepk": prepk,
        "ramp_late": ramp_late,
        "vch": vch,
        "ea": "early access" in low and "no early access" not in low,
        "pc_late": pc_late,
        "pc_hour": pc_hour,
        "unlock_class": cls,
        "u": unlock_hour(notes, cls),
        "year": ds.year,
        "console_only": console_case,
    }


# ---------------------------------------------------------------------------
# Small dense linear algebra (stdlib only)
# ---------------------------------------------------------------------------

def solve_sym(A, b):
    """Solve A x = b by Gauss-Jordan elimination with partial pivoting."""
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda rr: abs(M[rr][col]))
        if abs(M[piv][col]) < 1e-12:
            M[piv][col] = 1e-12
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        for rr in range(n):
            if rr != col and M[rr][col] != 0.0:
                f = M[rr][col] / pv
                for cc in range(col, n + 1):
                    M[rr][cc] -= f * M[col][cc]
    return [M[i][n] / M[i][i] for i in range(n)]


def invert(A):
    n = len(A)
    cols = []
    for j in range(n):
        e = [0.0] * n
        e[j] = 1.0
        cols.append(solve_sym(A, e))
    return [[cols[j][i] for j in range(n)] for i in range(n)]


def ridge_fit(X, y, lam, wts=None):
    """Ridge with an unpenalised intercept and optional row weights.

    Returns (beta, hat diagonal of the weighted hat matrix)."""
    n = len(X)
    p = len(X[0])
    if wts is None:
        wts = [1.0] * n
    XtX = [[0.0] * p for _ in range(p)]
    Xty = [0.0] * p
    for i in range(n):
        xi = X[i]
        yi = y[i]
        wi = wts[i]
        for a in range(p):
            Xty[a] += wi * xi[a] * yi
            xa = wi * xi[a]
            row = XtX[a]
            for bcol in range(p):
                row[bcol] += xa * xi[bcol]
    A = [row[:] for row in XtX]
    for a in range(1, p):
        A[a][a] += lam
    Minv = invert(A)
    beta = [sum(Minv[a][b] * Xty[b] for b in range(p)) for a in range(p)]
    hat = []
    for i in range(n):
        xi = X[i]
        h = 0.0
        for a in range(p):
            s = 0.0
            for bcol in range(p):
                s += Minv[a][bcol] * xi[bcol]
            h += xi[a] * s
        hat.append(wts[i] * h)
    return beta, hat


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

def donor_weights(target, donors, P):
    ws = []
    for d in donors:
        w = math.exp(-((d["log_h1"] - target["log_h1"]) ** 2) / (2.0 * P["sigma_scale"] ** 2))
        if target["ramp_slope"] is not None and d["ramp_slope"] is not None:
            w *= math.exp(-abs(d["ramp_slope"] - target["ramp_slope"]) / P["tau_ramp"])
        dc, tc = d["unlock_class"], target["unlock_class"]
        if dc != tc:
            w *= P["unlock_global"] if dc == "global" else P["unlock_unknown"]
        if d.get("ea") != target.get("ea"):
            w *= P["ea_mismatch"]
        if d.get("pc_late") != target.get("pc_late"):
            w *= P.get("pc_mismatch", 1.0)
        if target["console_only"]:
            w *= P["console_match"] if d["console_only"] else P["console_mismatch"]
        w *= 0.5 ** (abs(d["year"] - target["year"]) / P["era_halflife"])
        ws.append(max(w, 1e-9))
    return ws


def fit_predict(target, donors, P=PARAMS):
    """Returns (centre, spread): dicts over hours 2..24 (log-ratio centre/spread)."""
    n = len(donors)
    if n < P["min_donors"]:
        raise ValueError("too few donors")
    # Impute missing ramp values with the donor pool's median.
    for key in ("ramp_slope", "ramp_ratio", "prepk", "ramp_late", "vch"):
        vals = sorted(d[key] for d in donors if d[key] is not None)
        med = vals[len(vals) // 2] if vals else 0.0
        for d in donors:
            if d[key] is None:
                d[key] = med
        if target[key] is None:
            target[key] = med

    ws = donor_weights(target, donors, P)
    mean_log_h1 = sum(d["log_h1"] for d in donors) / n
    mean_ratio = sum(d["ramp_ratio"] for d in donors) / n
    mean_prepk = sum(d["prepk"] for d in donors) / n
    mean_late = sum(d["ramp_late"] for d in donors) / n
    mean_vch = sum(d["vch"] for d in donors) / n
    sw_total = sum(ws)
    n_eff = sw_total * sw_total / sum(w * w for w in ws)

    # Ensemble centres/dispersions, full and leave-one-out, per hour.
    C, S, Cloo, Sloo = {}, {}, {}, {}
    for h in HOURS:
        sw = sum(ws)
        swr = sum(ws[i] * donors[i]["r"][h] for i in range(n))
        swr2 = sum(ws[i] * donors[i]["r"][h] ** 2 for i in range(n))
        c = swr / sw
        var = max(swr2 / sw - c * c, 0.0)
        C[h], S[h] = c, math.sqrt(var)
        clo, slo = [], []
        for i in range(n):
            swi = sw - ws[i]
            if swi <= 1e-12:
                clo.append(c)
                slo.append(math.sqrt(var))
                continue
            ci = (swr - ws[i] * donors[i]["r"][h]) / swi
            vi = max((swr2 - ws[i] * donors[i]["r"][h] ** 2) / swi - ci * ci, 0.0)
            clo.append(ci)
            slo.append(math.sqrt(vi))
        Cloo[h], Sloo[h] = clo, slo

    def feats(get_c, get_s, src, h):
        c_h = get_c(h)
        c_prev = get_c(h - 1) if h > 2 else 0.0
        c_next = get_c(h + 1) if h < 24 else get_c(24)
        x = [1.0, c_h, c_prev, c_next, get_s(h), src["log_h1"] - mean_log_h1,
             src["ramp_slope"], src["ramp_ratio"] - mean_ratio,
             src["prepk"] - mean_prepk,
             P.get("feat_late", 1.0) * (src["ramp_late"] - mean_late),
             P.get("feat_vch", 1.0) * (src["vch"] - mean_vch),
             1.0 if src.get("ea") else 0.0,
             1.0 if src.get("pc_late") else 0.0,
             1.0 if src.get("pc_hour") and h >= src["pc_hour"] else 0.0,
             1.0 if h >= src["u"] else 0.0]
        return x

    centre = {}
    loo_pred = {h: [0.0] * n for h in HOURS}
    loo_resid = {h: [0.0] * n for h in HOURS}
    if not P.get("use_regression", True):
        for h in HOURS:
            centre[h] = C[h]
            for i in range(n):
                loo_pred[h][i] = Cloo[h][i]
                loo_resid[h][i] = donors[i]["r"][h] - Cloo[h][i]
    for h in HOURS if P.get("use_regression", True) else []:
        X = []
        for i in range(n):
            X.append(feats(lambda hh, i=i: Cloo[hh][i], lambda hh, i=i: Sloo[hh][i],
                           donors[i], h))
        y = [donors[i]["r"][h] for i in range(n)]
        best = None
        for lam in P["lambdas"]:
            beta, hat = ridge_fit(X, y, lam, ws)
            errs = []
            for i in range(n):
                fit_i = sum(beta[a] * X[i][a] for a in range(len(beta)))
                denom = max(1.0 - hat[i], 0.05)
                errs.append((y[i] - fit_i) / denom)
            mse = sum(e * e for e in errs) / n
            if best is None or mse < best[0]:
                best = (mse, beta, errs)
        _, beta, errs = best
        for i in range(n):
            loo_resid[h][i] = errs[i]
            loo_pred[h][i] = y[i] - errs[i]
        x_star = feats(lambda hh: C[hh], lambda hh: S[hh], target, h)
        centre[h] = sum(beta[a] * x_star[a] for a in range(len(beta)))

    # Shrunk bias correction, from the resemblance-weighted LOO residuals
    # (predicted - actual = -resid), so the correction reflects this target's regime.
    shrink = n_eff / (n_eff + P["bias_kappa"])
    for h in HOURS:
        bias = -sum(ws[i] * loo_resid[h][i] for i in range(n)) / sw_total
        centre[h] -= bias * shrink
        for i in range(n):
            loo_pred[h][i] -= bias * shrink

    # Optional smoothing of the predicted curves over neighbouring hours
    # (true launch curves are smooth; per-hour fits jitter).
    cw = P.get("centre_smooth", 0.0)
    if cw > 0:
        def smooth(vals):
            out = {}
            for h in HOURS:
                lo = vals[h - 1] if h > 2 else vals[2]
                hi = vals[h + 1] if h < 24 else vals[24]
                out[h] = cw * lo + (1 - 2 * cw) * vals[h] + cw * hi
            return out

        centre = smooth(centre)
        for i in range(n):
            col = smooth({h: loo_pred[h][i] for h in HOURS})
            for h in HOURS:
                loo_pred[h][i] = col[h]

    # Spread: per-hour resemblance-weighted LOO residual SD, shrunk to the
    # weighted global SD, smoothed over neighbouring hours, then a multiplier
    # calibrated so the weighted LOO coverage of the 5%-95% band is ~0.90.
    resid_all = [(ws[i], donors[i]["r"][h] - loo_pred[h][i]) for h in HOURS for i in range(n)]
    gmean = sum(w * e for w, e in resid_all) / sum(w for w, _ in resid_all)
    gsd = math.sqrt(sum(w * (e - gmean) ** 2 for w, e in resid_all) / sum(w for w, _ in resid_all))
    sd_h = {}
    for h in HOURS:
        es = [(ws[i], donors[i]["r"][h] - loo_pred[h][i]) for i in range(n)]
        m_h = sum(w * e for w, e in es) / sw_total
        sd = math.sqrt(sum(w * (e - m_h) ** 2 for w, e in es) / sw_total)
        sd_h[h] = math.sqrt((n_eff * sd * sd + P["spread_shrink"] * gsd * gsd) / (n_eff + P["spread_shrink"]))
    sd_s = {}
    for h in HOURS:
        vs = [sd_h[hh] ** 2 for hh in (h - 1, h, h + 1) if hh in sd_h]
        sd_s[h] = math.sqrt(sum(vs) / len(vs))
    best_m, best_gap = 1.0, 9.9
    for m in P["m_grid"]:
        cov = 0.0
        for h in HOURS:
            half = Z[4] * m * sd_s[h]
            for i in range(n):
                if abs(donors[i]["r"][h] - loo_pred[h][i]) <= half:
                    cov += ws[i]
        gap = abs(cov / (sw_total * len(HOURS)) - 0.90)
        if gap < best_gap:
            best_gap, best_m = gap, m
    spread = {h: max(best_m * sd_s[h], P["spread_floor"]) for h in HOURS}
    return centre, spread


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def round2(x):
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if float(value).is_integer() else value


def baseline_rows(p1):
    rows = []
    for h in HOURS:
        q = [round2(p1 * math.exp(z * 0.15 * math.sqrt(h - 1))) for z in Z]
        rows.append({"hour": h, "q": q})
    return rows


def model_rows(case, p1, ch1=None):
    donors = build_donors(case)
    target = case_features(case, p1, ch1)
    centre, spread = fit_predict(target, donors)
    rows = []
    for h in HOURS:
        q = []
        for z in Z:
            v = p1 * math.exp(centre[h] + z * spread[h])
            q.append(round2(clamp(v, 0.05, 1e9)))
        rows.append({"hour": h, "q": sorted(q)})
    return rows


def forecast_case(case):
    first = None
    known = case.get("known")
    if isinstance(known, list):
        first = next((k for k in known if k.get("hour") == 1), None)
    elif isinstance(known, dict):
        first = known
    if first is None:
        raise ValueError("no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))
    try:
        return {"hours": model_rows(case, p1, first.get("channels"))}
    except Exception:
        return {"hours": baseline_rows(p1)}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
