#!/usr/bin/env python3
"""
Zone scanner for the Zone Ticket page.

Screens a watchlist, finds fresh supply/demand zones on the 15-minute chart,
and ranks the plays. Writes plays.json, which you paste into the page.

Setup:   pip install yfinance pandas numpy
Run:     python scanner.py NVDA TSLA AMD
         python scanner.py --file watchlist.txt
Options: --all        show every zone, not only the best two per stock
         --max-dist N only plays within N x 15m ATR of price (default 8)
         --json PATH  where to write the output (default plays.json)

Data is delayed and free. Always check the chart before you place an order.
"""
import argparse
import datetime as dt
import json
import sys

import numpy as np
import pandas as pd


# ---------- indicators ----------
def atr_series(df, n=14):
    pc = df.Close.shift()
    tr = pd.concat([df.High - df.Low, (df.High - pc).abs(), (df.Low - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


# ---------- zone detection ----------
def find_zones(df, a, impulse=2.0):
    """A zone is 1-3 small candles (the base) followed by a big candle that
    leaves the base and travels at least `impulse` x ATR within 3 candles."""
    o, h, l, c, v = (df[k].values for k in ("Open", "High", "Low", "Close", "Volume"))
    n = len(df)
    body = np.abs(c - o)
    rng = h - l
    small = (body <= 0.5 * a) & (rng <= 1.2 * a)
    avgv = float(np.mean(v)) if n else 0.0
    zones = []
    for j in range(1, n - 1):
        k = 0
        while k < 3 and j - 1 - k >= 0 and small[j - 1 - k]:
            k += 1
        if k == 0 or body[j] < 0.8 * a:
            continue
        base = slice(j - k, j)
        end = min(j + 3, n)
        if c[j] > o[j]:
            low = l[base].min()
            move = h[j:end].max() - low
            if move < impulse * a:
                continue
            typ, prox, dist = "demand", max(o[base].max(), c[base].max()), low
        elif c[j] < o[j]:
            high = h[base].max()
            move = high - l[j:end].min()
            if move < impulse * a:
                continue
            typ, prox, dist = "supply", min(o[base].min(), c[base].min()), high
        else:
            continue
        height = abs(prox - dist)
        if height < 0.05 * a:
            continue
        after = slice(min(j + 3, n), n)
        if typ == "demand":
            broken = bool((c[after] < dist).any())
            touched = bool((l[after] <= prox).any())
        else:
            broken = bool((c[after] > dist).any())
            touched = bool((h[after] >= prox).any())
        zones.append(dict(type=typ, prox=float(prox), dist=float(dist), height=float(height),
                          idx=j, depart=float(move / a), volx=float(v[j] / avgv) if avgv else 0.0,
                          fresh=not touched, broken=broken))
    return zones


# ---------- play building ----------
ATR_RANGE = {"stock": (2, 8), "forex": (0.25, 2.5), "commodity": (0.8, 5),
             "index": (0.6, 4), "crypto": (2, 10)}


def decimals(price):
    return 2 if price >= 100 else 3 if price >= 10 else 5 if price >= 1 else 6


def build_plays(tk, d15, d1h, dd, spread=None, max_dist=8.0, buf_pct=10.0, kind="stock"):
    a = float(atr_series(d15).iloc[-1])
    price = float(d15.Close.iloc[-1])
    nd = decimals(price)
    adv = float(dd.Volume.iloc[-21:-1].mean())
    rvol = float(dd.Volume.iloc[-1] / adv) if adv else 0.0
    atr_pct = float(atr_series(dd).iloc[-1] / dd.Close.iloc[-1] * 100)

    ema1h = d1h.Close.ewm(span=20, adjust=False).mean()
    up1h = d1h.Close.iloc[-1] > ema1h.iloc[-1] and ema1h.iloc[-1] > ema1h.iloc[-5]
    dn1h = d1h.Close.iloc[-1] < ema1h.iloc[-1] and ema1h.iloc[-1] < ema1h.iloc[-5]
    emad = dd.Close.ewm(span=20, adjust=False).mean()
    upd = dd.Close.iloc[-1] > emad.iloc[-1]
    dnd = dd.Close.iloc[-1] < emad.iloc[-1]

    lo, hi = ATR_RANGE.get(kind, ATR_RANGE["stock"])
    if kind == "stock":
        screen = [
            price > 5,
            adv >= 1e6,
            rvol >= 1.5,
            lo <= atr_pct <= hi,
            True if spread is None else spread <= 0.1,
        ]
    else:
        # forex, commodities, indices and crypto have no reliable volume feed here,
        # so the screen only asks for enough daily movement to be worth trading.
        screen = [lo <= atr_pct <= hi]
    screen_ok = all(screen)

    zones = [z for z in find_zones(d15, a) if not z["broken"]]
    session = d15.tail(52)  # roughly the last two sessions
    plays = []
    for z in zones:
        long = z["type"] == "demand"
        entry = z["prox"]
        if long and not price > entry:
            continue
        if (not long) and not price < entry:
            continue
        if abs(price - entry) > max_dist * a:
            continue
        buf = z["height"] * buf_pct / 100
        stop = z["dist"] - buf if long else z["dist"] + buf
        # target: nearest opposing zone ahead of price, else the recent swing
        opp = [x for x in zones if x["type"] == ("supply" if long else "demand")]
        if long:
            ahead = [x["prox"] for x in opp if x["prox"] > price]
            tgt = min(ahead) if ahead else float(session.High.max())
            if tgt <= entry:
                continue
        else:
            ahead = [x["prox"] for x in opp if x["prox"] < price]
            tgt = max(ahead) if ahead else float(session.Low.min())
            if tgt >= entry:
                continue
        risk = abs(entry - stop)
        if risk <= 0:
            continue
        rr = abs(tgt - entry) / risk
        checks = [
            z["depart"] >= 3.0,                                  # strong departure
            z["fresh"],                                          # untouched since forming
            z["height"] <= 1.0 * a,                              # tight base
            (up1h and upd) if long else (dn1h and dnd),          # higher timeframe agrees
            (z["volx"] >= 1.5 or rvol >= 2.0) if kind == "stock" else z["depart"] >= 4.0,  # volume or momentum
        ]
        score = int(sum(checks))
        take = screen_ok and score >= 4 and rr >= 3
        plays.append(dict(
            tkr=tk, side="long" if long else "short", kind=kind, nd=nd,
            px=round(price, nd), adv=int(round(adv)), rvol=round(rvol, 2), atr=round(atr_pct, 2),
            spr=None if spread is None else round(spread, 3),
            prox=round(entry, nd), dist=round(z["dist"], nd), tgt=round(tgt, nd),
            rr=round(rr, 1), score=score, sc=[bool(x) for x in checks],
            screen=[bool(x) for x in screen], take=bool(take), tf="15m",
        ))
    plays.sort(key=lambda p: (p["take"], p["score"], p["rr"]), reverse=True)
    return plays


# ---------- data ----------
def fetch(tk, with_info=True):
    import yfinance as yf
    t = yf.Ticker(tk)
    d15 = t.history(period="10d", interval="15m")
    d1h = t.history(period="1mo", interval="1h")
    dd = t.history(period="3mo", interval="1d")
    spread = None
    if with_info:
        try:
            info = t.info
            bid, ask = info.get("bid") or 0, info.get("ask") or 0
            if bid > 0 and ask > 0:
                spread = (ask - bid) / ((ask + bid) / 2) * 100
        except Exception:
            pass
    return d15.dropna(), d1h.dropna(), dd.dropna(), spread


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tickers", nargs="*")
    ap.add_argument("--file")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--max-dist", type=float, default=8.0)
    ap.add_argument("--json", default="plays.json")
    args = ap.parse_args()

    tks = [t.upper() for t in args.tickers]
    if args.file:
        with open(args.file) as f:
            tks += [x.strip().upper() for x in f.read().replace(",", "\n").split() if x.strip()]
    if not tks:
        sys.exit("Give tickers or --file watchlist.txt")

    out = []
    for tk in dict.fromkeys(tks):
        try:
            d15, d1h, dd, spread = fetch(tk)
            if len(d15) < 60 or len(d1h) < 10 or len(dd) < 25:
                print(f"{tk}: not enough data, skipped")
                continue
            plays = build_plays(tk, d15, d1h, dd, spread, args.max_dist)
            out += plays if args.all else plays[:2]
            if not plays:
                print(f"{tk}: no zone near price")
        except Exception as e:
            print(f"{tk}: error ({e})")

    out.sort(key=lambda p: (p["take"], p["score"], p["rr"]), reverse=True)
    print(f"\n{'TICKER':8}{'SIDE':7}{'ENTRY':>9}{'STOP*':>9}{'TARGET':>9}{'R:R':>6}{'SCORE':>7}  RESULT")
    for p in out:
        scr = "" if all(p["screen"]) else " (fails screen)"
        stop_hint = p["dist"]
        print(f"{p['tkr']:8}{p['side']:7}{p['prox']:>9.2f}{stop_hint:>9.2f}{p['tgt']:>9.2f}"
              f"{p['rr']:>6.1f}{p['score']:>5}/5  {'TAKE' if p['take'] else 'skip'}{scr}")
    print("* far edge of the zone; the page adds the stop buffer.")

    payload = dict(generated=dt.datetime.now().isoformat(timespec="minutes"), plays=out)
    with open(args.json, "w") as f:
        json.dump(payload, f)
    print(f"\nWrote {args.json}. Paste its contents into the Zone Ticket page.")


if __name__ == "__main__":
    main()
