# GTA VI Twitch CCU forecast — gta6-twitch-ccu-forecast-01 (v2)

[![Competition](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/c/gta6-twitch-ccu-forecast-01)
[![Rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/hdr2804/makefun-gta6-twitch-ccu-forecast-01-v2/rank.svg)](https://make.fun/c/gta6-twitch-ccu-forecast-01)
[![Reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/hdr2804/makefun-gta6-twitch-ccu-forecast-01-v2/reward.svg)](https://make.fun/c/gta6-twitch-ccu-forecast-01)

Entry for the make.fun bounty [gta6-twitch-ccu-forecast-01](https://make.fun/c/gta6-twitch-ccu-forecast-01):
forecast, hour by hour, how many people watch Grand Theft Auto VI live on
Twitch over the 24 UTC hours of its release day (2026-11-19), from everything
recorded up to the end of hour 1 — five quantiles (5/25/50/75/95%) for each of
hours 2–24, scored by mean pinball loss on the log scale.

## The model

Everything is fitted at run time from the case's own comparables (past big
launches' hourly Twitch curves); there are no baked-in coefficients beyond
hyperparameters tuned on a leave-one-out loop over those comparables.

1. **Donor ensemble** — each comparable's per-hour log-ratio curve (hour *h*
   viewers over its own hour-1 viewers, UTC release day), weighted by
   resemblance to the case: a scale kernel on log hour-1, unlock-structure
   match (rolling midnight-local launches resemble this one; single-moment
   global unlocks are cut hard), early-access and late-PC-wave status,
   console-only match, and an era decay.
2. **Ridge regression layer** — one fit per forecast hour, trained on the
   donors themselves: each donor's features are its leave-one-out ensemble
   centre and dispersion at that hour, its scale, its pre-release ramp, its
   early-access and late-PC-wave status and its unlock position. The ridge
   penalty is picked per hour by the closed-form leave-one-out shortcut.
3. **Bias correction** — the donors' leave-one-out residuals, resemblance-
   weighted and shrunk, estimate the centre's bias per hour; it is subtracted.
4. **Spread calibration** — the spread comes from the leave-one-out residual
   dispersion per hour, with its multiplier calibrated on the donors so the
   5%–95% band covers about 90%.

Forecast = hour-1 print × exp(centre + z·spread), with the standard normal
quantiles. Any failure in the model path falls back per case to a persistence
forecast, so the output is always valid. Deterministic, standard library
only, no network.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` case file in the input directory (name order) and writes
one `<id>.out` per case to the output directory. With the environment
variables unset it uses `/input` and `/output`, as the grader does.

## Validation

The scorer was validated exactly against the rules' worked example (loss
0.112545, typical miss 19.52%, coverage 1.000). On leave-one-out over the 37
launch comparables in the kit (each launch predicted from the other 36):
overall loss 0.3246; on the big-launch subset (hour-1 ≥ 20,000 viewers — the
regime of all four practice cases and of GTA VI) loss **0.2038**, typical
miss 36.4%, 90%-band coverage 0.894. For reference, the previous entry's
design (a kernel-weighted donor ensemble alone) measured 0.2452 on the same
big-subset leave-one-out, with a server practice coverage of 0.957 — this
model is both sharper and more honestly calibrated.

MIT licensed — see [LICENSE](LICENSE).
