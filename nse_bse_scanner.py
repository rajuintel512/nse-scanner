#!/usr/bin/env python3
"""
NSE / BSE paper-trading signal scanner with market-hours mode and email alerts.
EDUCATIONAL ONLY - not financial advice.

Setup (once):
    pip install yfinance pandas requests

Usage:
    python nse_bse_scanner.py --limit 100          # end-of-day scan, quick test on 100 stocks
    python nse_bse_scanner.py --paper              # end-of-day scan + paper portfolio update
    python nse_bse_scanner.py --live --paper       # scan DURING market hours, email alerts
    python nse_bse_scanner.py --live --once        # one live scan right now (good for testing)
    python nse_bse_scanner.py --test-alerts        # check your email/Telegram/WhatsApp settings
    add --bse to include tickers from bse_symbols.txt, --alert to send an end-of-day scan

Alerts can go to email, Telegram and/or WhatsApp - whichever you configure (see below).
No laptop? Run it in the cloud for free with the included GitHub Actions file (scan.yml):
it runs `--live --once` every 30 minutes during market hours and stores its state in the repo.

Live mode (--live):
  * Start it any trading day before 3:25 PM IST (ideally ~9:10 AM). It downloads daily
    history first, waits for 9:30 AM, then rescans every --interval minutes (default 30)
    using today's 5-minute candles as the "current" daily candle. Today's volume is
    projected to a full-day figure so volume-breakout checks stay meaningful.
  * The last scan runs at 3:25 PM IST and the script then exits. On exchange holidays
    (no data for today) it exits by itself.
  * You get an email only for NEW signals (a stock+signal is emailed once per day).
  * With --paper, the virtual portfolio is updated once, at --paper-time (default 15:20),
    and a summary is emailed.
  * Yahoo Finance data can lag by several minutes. Do not treat it as tick-accurate.
  * Run it daily by hand, or schedule it (Windows Task Scheduler / cron) at 9:10 AM on
    weekdays:   10 9 * * 1-5  cd /path/to/folder && python nse_bse_scanner.py --live --paper
    (cron uses the machine's clock; the script itself always uses IST.)

Email setup (Gmail example):
  1. Turn on 2-Step Verification in your Google account.
  2. Create an App Password at https://myaccount.google.com/apppasswords
  3. Set environment variables (never paste the password into this file):
       Mac/Linux:  export SMTP_USER="you@gmail.com"
                   export SMTP_PASS="your-16-char-app-password"
       Windows:    setx SMTP_USER "you@gmail.com"
                   setx SMTP_PASS "your-16-char-app-password"     (then reopen the terminal)
     Optional: ALERT_TO (send to a different address), SMTP_HOST (default smtp.gmail.com),
     SMTP_PORT (default 465, SSL).
  4. Test with:  python nse_bse_scanner.py --test-alerts

Telegram setup (recommended - free, official, reliable):
  1. In Telegram, open @BotFather, send /newbot, follow the steps, copy the bot token.
  2. Send any message ("hi") to your new bot.
  3. Open  https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates  in a browser and copy the
     number after "chat":{"id":  - that is your chat id.
  4. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID (environment variables, or GitHub Secrets).

WhatsApp setup (free, via the third-party CallMeBot service - unofficial, can be unreliable,
and your alert text passes through their server; fine for paper-trading alerts):
  1. Follow the registration steps on callmebot.com (save their WhatsApp number, send
     "I allow callmebot to send me messages", and you get an API key back in WhatsApp).
     Their number changes from time to time, so always take it from their website.
  2. Set CALLMEBOT_PHONE (your number with country code, e.g. +919876543210) and
     CALLMEBOT_APIKEY.
  WhatsApp messages are shortened to ~1500 characters.

Other notes:
  * NSE list is downloaded from NSE automatically. If that fails, put one symbol
    per line in nse_symbols.txt (e.g. RELIANCE, TCS, INFY).
  * BSE has no easy free list, so put BSE tickers/scrip codes (one per line) in
    bse_symbols.txt (e.g. 500325 or RELIANCE). Stocks on both exchanges are
    scanned once (NSE copy wins).
  * "SELL" means: exit long positions / avoid. It does not mean short-sell.
"""
import argparse
import io
import json
import os
import smtplib
import sys
import time
from datetime import datetime, timedelta, timezone
from datetime import time as dtime
from email.message import EmailMessage
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import requests
    import yfinance as yf
except ImportError:
    sys.exit("Missing packages. Run: pip install yfinance pandas requests")

NSE_LIST_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
PORTFOLIO_FILE = Path("paper_portfolio.json")
STATE_FILE = Path("alerts_state.json")

IST = timezone(timedelta(hours=5, minutes=30))
OPEN_T = dtime(9, 15)
FIRST_SCAN = dtime(9, 30)
LAST_SCAN = dtime(15, 25)
SESSION_MINUTES = 375  # 9:15 to 15:30

# ----------------------------------------------------------------------------
# Symbol lists
# ----------------------------------------------------------------------------


def read_list(path, suffix):
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        s = line.strip().upper()
        if not s or s.startswith("#"):
            continue
        if not (s.endswith(".NS") or s.endswith(".BO")):
            s += suffix
        out.append(s)
    return out


def get_nse_symbols():
    try:
        r = requests.get(NSE_LIST_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = [c.strip() for c in df.columns]
        df = df[df["SERIES"].str.strip() == "EQ"]
        syms = [s.strip() + ".NS" for s in df["SYMBOL"]]
        print(f"NSE: {len(syms)} symbols downloaded")
        return syms
    except Exception as e:  # noqa: BLE001
        print(f"Could not download NSE list ({e}). Falling back to nse_symbols.txt")
        return read_list("nse_symbols.txt", ".NS")


def build_universe(include_bse):
    syms = get_nse_symbols()
    if include_bse:
        nse_names = {s.rsplit(".", 1)[0] for s in syms}
        bse = [s for s in read_list("bse_symbols.txt", ".BO") if s.rsplit(".", 1)[0] not in nse_names]
        print(f"BSE: {len(bse)} extra symbols from bse_symbols.txt")
        syms += bse
    return list(dict.fromkeys(syms))


# ----------------------------------------------------------------------------
# Data download
# ----------------------------------------------------------------------------


def get_frame(data, t):
    """yfinance returns MultiIndex columns for several tickers (and sometimes for one)."""
    return data[t] if isinstance(data.columns, pd.MultiIndex) else data


def download_all(symbols, chunk=100):
    frames = {}
    for i in range(0, len(symbols), chunk):
        batch = symbols[i:i + chunk]
        print(f"Downloading {i + 1}-{i + len(batch)} of {len(symbols)} ...", flush=True)
        try:
            data = yf.download(batch, period="1y", interval="1d", group_by="ticker",
                               auto_adjust=True, threads=True, progress=False)
        except Exception as e:  # noqa: BLE001
            print(f"  batch failed: {e}")
            continue
        for t in batch:
            try:
                d = get_frame(data, t).dropna(subset=["Close"])
                if len(d):
                    frames[t] = d
            except Exception:  # noqa: BLE001
                continue
    return frames


def download_intraday(symbols, today, chunk=100):
    """Return {symbol: today's partial daily bar built from 5-minute candles}."""
    bars = {}
    for i in range(0, len(symbols), chunk):
        batch = symbols[i:i + chunk]
        try:
            data = yf.download(batch, period="1d", interval="5m", group_by="ticker",
                               auto_adjust=True, threads=True, progress=False)
        except Exception as e:  # noqa: BLE001
            print(f"  intraday batch failed: {e}")
            continue
        for t in batch:
            try:
                d = get_frame(data, t).dropna(subset=["Close"])
                idx = d.index.tz_convert(IST) if d.index.tz is not None else d.index
                d = d[idx.date == today]
                if len(d):
                    bars[t] = {"Open": float(d["Open"].iloc[0]), "High": float(d["High"].max()),
                               "Low": float(d["Low"].min()), "Close": float(d["Close"].iloc[-1]),
                               "Volume": float(d["Volume"].sum())}
            except Exception:  # noqa: BLE001
                continue
    return bars


def session_fraction(now):
    mins = now.hour * 60 + now.minute - (OPEN_T.hour * 60 + OPEN_T.minute)
    return min(1.0, max(0.15, mins / SESSION_MINUTES))


def merge_live(daily_df, bar, today, frac):
    """Daily history + today's partial candle (volume projected to a full day)."""
    d = daily_df[daily_df.index.date < today]
    row = pd.DataFrame({"Open": [bar["Open"]], "High": [bar["High"]], "Low": [bar["Low"]],
                        "Close": [bar["Close"]], "Volume": [bar["Volume"] / frac]},
                       index=[pd.Timestamp(today)])
    return pd.concat([d[["Open", "High", "Low", "Close", "Volume"]], row])


# ----------------------------------------------------------------------------
# Indicators
# ----------------------------------------------------------------------------


def rsi(close, n=14):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def atr(high, low, close, n=14):
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def crossed_up(a, b, n=3):
    d = a - b
    return bool(((d.shift(1) <= 0) & (d > 0)).iloc[-n:].any())


def crossed_down(a, b, n=3):
    d = a - b
    return bool(((d.shift(1) >= 0) & (d < 0)).iloc[-n:].any())


# ----------------------------------------------------------------------------
# Signal logic
# ----------------------------------------------------------------------------


def analyze(sym, df, min_turnover=1e7, min_price=20):
    df = df.dropna(subset=["Close"])
    if len(df) < 210:
        return None
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    price = float(c.iloc[-1])
    vol_avg = v.rolling(20).mean()
    turnover = float(vol_avg.iloc[-1] * price)
    if price < min_price or not turnover >= min_turnover:
        return None  # skip illiquid / penny stocks

    sma20, sma50, sma200 = c.rolling(20).mean(), c.rolling(50).mean(), c.rolling(200).mean()
    r = rsi(c)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    macd_sig = macd.ewm(span=9, adjust=False).mean()
    a = float(atr(h, l, c).iloc[-1])
    prior_high = float(h.iloc[-21:-1].max())
    prior_low = float(l.iloc[-21:-1].min())
    vol_spike = float(v.iloc[-1]) > 1.5 * float(vol_avg.iloc[-1])

    buy, sell, why_b, why_s = 0, 0, [], []

    if price > sma200.iloc[-1] and sma50.iloc[-1] > sma200.iloc[-1]:
        buy += 2; why_b.append("uptrend (price>200DMA, 50DMA>200DMA)")
    if price < sma200.iloc[-1] and sma50.iloc[-1] < sma200.iloc[-1]:
        sell += 2; why_s.append("downtrend (price<200DMA, 50DMA<200DMA)")

    if crossed_up(macd, macd_sig):
        buy += 2; why_b.append("MACD bullish cross")
    if crossed_down(macd, macd_sig):
        sell += 2; why_s.append("MACD bearish cross")

    if crossed_up(sma20, sma50):
        buy += 1; why_b.append("20DMA crossed above 50DMA")
    if crossed_down(sma20, sma50):
        sell += 1; why_s.append("20DMA crossed below 50DMA")

    if price > prior_high and vol_spike:
        buy += 2; why_b.append("20-day breakout on high volume")
    if price < prior_low and vol_spike:
        sell += 2; why_s.append("20-day breakdown on high volume")

    rv = float(r.iloc[-1])
    if 50 <= rv <= 70:
        buy += 1; why_b.append(f"RSI {rv:.0f} (healthy momentum)")
    if rv > 75:
        sell += 1; why_s.append(f"RSI {rv:.0f} (overbought)")
    if rv < 40 and price < sma50.iloc[-1]:
        sell += 1; why_s.append(f"RSI {rv:.0f} (weak)")

    signal = "HOLD"
    if buy >= 4 and buy - sell >= 3:
        signal = "BUY"
    elif sell >= 4 and sell - buy >= 3:
        signal = "SELL"

    return {
        "symbol": sym, "signal": signal, "buy_score": buy, "sell_score": sell,
        "close": round(price, 2), "rsi": round(rv, 1), "atr": round(a, 2),
        "turnover_cr": round(turnover / 1e7, 2),
        "reasons": "; ".join(why_b if signal == "BUY" else why_s if signal == "SELL" else why_b + why_s),
        "bar_date": str(df.index[-1].date()),
        "high": round(float(h.iloc[-1]), 2), "low": round(float(l.iloc[-1]), 2),
    }


# ----------------------------------------------------------------------------
# Email
# ----------------------------------------------------------------------------


def _send_email(subject, body):
    """True = sent, False = failed, None = email not configured."""
    user, pw = os.environ.get("SMTP_USER"), os.environ.get("SMTP_PASS")
    if not user or not pw:
        return None
    msg = EmailMessage()
    msg["Subject"], msg["From"], msg["To"] = subject, user, os.environ.get("ALERT_TO", user)
    msg.set_content(body)
    try:
        with smtplib.SMTP_SSL(os.environ.get("SMTP_HOST", "smtp.gmail.com"),
                              int(os.environ.get("SMTP_PORT", "465")), timeout=30) as s:
            s.login(user, pw)
            s.send_message(msg)
        print(f"[email sent] {subject}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[email failed] {type(e).__name__}")
        return False


def _send_telegram(subject, body):
    """True = sent, False = failed, None = Telegram not configured."""
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        return None
    text = f"{subject}\n\n{body}"
    try:
        for i in range(0, len(text), 4000):  # Telegram limit is 4096 characters
            r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                              data={"chat_id": chat, "text": text[i:i + 4000]}, timeout=30)
            r.raise_for_status()
        print("[telegram sent]")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[telegram failed] {type(e).__name__}")  # never print the URL: it contains the token
        return False


def _send_whatsapp(subject, body):
    """True = sent, False = failed, None = WhatsApp not configured (uses CallMeBot)."""
    phone, key = os.environ.get("CALLMEBOT_PHONE"), os.environ.get("CALLMEBOT_APIKEY")
    if not phone or not key:
        return None
    text = f"{subject}\n{body}"
    if len(text) > 1500:
        text = text[:1500].rsplit("\n", 1)[0] + "\n... (shortened)"
    try:
        r = requests.get("https://api.callmebot.com/whatsapp.php",
                         params={"phone": phone, "text": text, "apikey": key}, timeout=30)
        r.raise_for_status()
        print("[whatsapp sent]")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[whatsapp failed] {type(e).__name__}")
        return False


def notify(subject, body):
    """Send through every configured channel. True = at least one worked,
    False = all configured channels failed, None = no channel configured."""
    outcomes = [f(subject, body) for f in (_send_email, _send_telegram, _send_whatsapp)]
    configured = [o for o in outcomes if o is not None]
    if not configured:
        print("[alerts skipped] no email / Telegram / WhatsApp settings found (see the top of this file)")
        return None
    return any(configured)


def format_alert(results, now):
    lines = [f"NSE/BSE scanner - {now:%d %b %Y, %H:%M} IST", ""]
    for sig in ("BUY", "SELL"):
        key = "buy_score" if sig == "BUY" else "sell_score"
        sub = sorted([r for r in results if r["signal"] == sig], key=lambda r: r[key], reverse=True)
        if not sub:
            continue
        lines.append(f"{sig} signals ({len(sub)}){'  [exit / avoid]' if sig == 'SELL' else ''}:")
        for r in sub[:25]:
            lines.append(f"  {r['symbol'].rsplit('.', 1)[0]:<12} Rs {r['close']:<9} RSI {r['rsi']:<5} {r['reasons']}")
        if len(sub) > 25:
            lines.append(f"  ... and {len(sub) - 25} more")
        lines.append("")
    lines.append("Paper-trading / educational use only. Not financial advice.")
    return "\n".join(lines)


def load_state(today):
    if STATE_FILE.exists():
        try:
            s = json.loads(STATE_FILE.read_text())
            if s.get("date") == str(today):
                return s
        except Exception:  # noqa: BLE001
            pass
    return {"date": str(today), "sent": []}


def alert_new(results, state, now):
    """Alert only on signals not already sent today."""
    new = [r for r in results if r["signal"] in ("BUY", "SELL")
           and f"{r['symbol']}|{r['signal']}" not in state["sent"]]
    if not new:
        print("No new signals to send.")
        return
    nb = sum(r["signal"] == "BUY" for r in new)
    ok = notify(f"[Scanner] {nb} new BUY / {len(new) - nb} new SELL - {now:%H:%M}", format_alert(new, now))
    if ok is not False:  # sent, or no channel configured (don't retry forever)
        state["sent"] += [f"{r['symbol']}|{r['signal']}" for r in new]
        STATE_FILE.write_text(json.dumps(state))


# ----------------------------------------------------------------------------
# Paper portfolio (virtual money, no broker connection)
# ----------------------------------------------------------------------------


def load_portfolio(capital):
    if PORTFOLIO_FILE.exists():
        return json.loads(PORTFOLIO_FILE.read_text())
    return {"cash": capital, "start_capital": capital, "positions": {}, "history": []}


def equity(pf, last_close):
    return pf["cash"] + sum(p["qty"] * last_close.get(s, p["entry"]) for s, p in pf["positions"].items())


def run_paper(results, max_positions=10, risk_pct=0.01, max_weight=0.10, cost=0.001,
              capital=1_000_000, stop_atr=2.0, target_atr=4.0):
    """Update the virtual portfolio. Returns the printed lines (used for the email)."""
    log = []

    def say(s):
        print(s)
        log.append(s)

    pf = load_portfolio(capital)
    by_sym = {r["symbol"]: r for r in results}
    last_close = {s: r["close"] for s, r in by_sym.items()}

    # 1) manage open positions (only candles after the entry day are used)
    for sym in list(pf["positions"]):
        p, r = pf["positions"][sym], by_sym.get(sym)
        if not r or r["bar_date"] <= p["date"]:
            continue
        exit_price, reason = None, None
        if r["low"] <= p["stop"]:
            exit_price, reason = p["stop"], "stop-loss"
        elif r["high"] >= p["target"]:
            exit_price, reason = p["target"], "target"
        elif r["signal"] == "SELL":
            exit_price, reason = r["close"], "sell signal"
        if exit_price:
            proceeds = p["qty"] * exit_price * (1 - cost)
            pnl = proceeds - p["qty"] * p["entry"] * (1 + cost)
            pf["cash"] += proceeds
            pf["history"].append({"symbol": sym, "entry": p["entry"], "exit": round(exit_price, 2),
                                  "qty": p["qty"], "pnl": round(pnl, 2), "reason": reason,
                                  "entry_date": p["date"], "exit_date": r["bar_date"]})
            del pf["positions"][sym]
            say(f"  EXIT  {sym}: {reason} @ {exit_price:.2f}  P&L {pnl:,.0f}")

    # 2) open new positions from best BUY signals
    eq = equity(pf, last_close)
    buys = sorted([r for r in results if r["signal"] == "BUY" and r["symbol"] not in pf["positions"]],
                  key=lambda r: (r["buy_score"], r["turnover_cr"]), reverse=True)
    for r in buys:
        if len(pf["positions"]) >= max_positions:
            break
        risk_per_share = stop_atr * r["atr"]
        if risk_per_share <= 0:
            continue
        qty = int(min(eq * risk_pct / risk_per_share, eq * max_weight / r["close"]))
        cost_total = qty * r["close"] * (1 + cost)
        if qty < 1 or cost_total > pf["cash"]:
            continue
        pf["cash"] -= cost_total
        pf["positions"][r["symbol"]] = {
            "qty": qty, "entry": r["close"], "date": r["bar_date"],
            "stop": round(r["close"] - risk_per_share, 2),
            "target": round(r["close"] + target_atr * r["atr"], 2)}
        pos = pf["positions"][r["symbol"]]
        say(f"  ENTER {r['symbol']}: {qty} @ {r['close']}  stop {pos['stop']}  target {pos['target']}")

    PORTFOLIO_FILE.write_text(json.dumps(pf, indent=2))
    eq = equity(pf, last_close)
    wins = [t for t in pf["history"] if t["pnl"] > 0]
    say(f"\nPaper portfolio: equity Rs {eq:,.0f} ({(eq / pf['start_capital'] - 1) * 100:+.2f}%), "
        f"cash Rs {pf['cash']:,.0f}, open {len(pf['positions'])}, closed {len(pf['history'])}"
        + (f", win rate {len(wins) / len(pf['history']) * 100:.0f}%" if pf["history"] else ""))
    if pf["positions"]:
        say("Open positions: " + ", ".join(f"{s.rsplit('.', 1)[0]} ({p['qty']} @ {p['entry']})"
                                          for s, p in pf["positions"].items()))
    return log


# ----------------------------------------------------------------------------
# Live (market-hours) mode
# ----------------------------------------------------------------------------


def scan_cycle(daily, today, frac, min_turnover):
    bars = download_intraday(list(daily), today)
    results = []
    for s, b in bars.items():
        r = analyze(s, merge_live(daily[s], b, today, frac), min_turnover)
        if r:
            results.append(r)
    return bars, results


def print_summary(results, now):
    print(f"\n[{now:%H:%M} IST] {len(results)} stocks analysed: "
          f"{sum(r['signal'] == 'BUY' for r in results)} BUY, {sum(r['signal'] == 'SELL' for r in results)} SELL")
    for sig, key in (("BUY", "buy_score"), ("SELL", "sell_score")):
        sub = sorted([r for r in results if r["signal"] == sig], key=lambda r: r[key], reverse=True)[:10]
        for r in sub:
            print(f"  {sig:<4} {r['symbol'].rsplit('.', 1)[0]:<12} Rs {r['close']:<9} {r['reasons']}")


def live(args):
    now = datetime.now(IST)
    today = now.date()
    if not args.once:
        if now.weekday() >= 5:
            sys.exit("Market is closed on weekends.")
        if now.time() >= LAST_SCAN:
            sys.exit("Market is closed for today. Start this before 3:25 PM IST on a trading day.")

    symbols = build_universe(args.bse)
    if args.limit:
        symbols = symbols[:args.limit]
    if not symbols:
        sys.exit("No symbols to scan.")

    print("Loading daily history (this takes a few minutes for all NSE stocks) ...")
    daily = {}
    for s, d in download_all(symbols).items():
        if analyze(s, d, args.min_turnover):
            d = d[["Open", "High", "Low", "Close", "Volume"]].copy()
            d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
            daily[s] = d
    if not daily:
        sys.exit("No liquid stocks with enough history.")
    print(f"{len(daily)} liquid stocks will be scanned during market hours.")

    h, m = (int(x) for x in args.paper_time.split(":"))
    paper_dt = datetime.combine(today, dtime(h, m), IST)
    first_dt = datetime.combine(today, FIRST_SCAN, IST)
    last_dt = datetime.combine(today, LAST_SCAN, IST)
    paper_done = not args.paper
    state = load_state(today)

    while not args.once and datetime.now(IST) < first_dt:
        time.sleep(20)

    while True:
        now = datetime.now(IST)
        bars, results = scan_cycle(daily, today, session_fraction(now), args.min_turnover)
        if not bars:
            if args.once:
                print("No intraday data for today - probably an exchange holiday (or Yahoo returned nothing). Exiting.")
                return
            if now.time() > dtime(10, 30):
                sys.exit("No intraday data for today - probably an exchange holiday (or no market data). Exiting.")
            print("No intraday data yet, will retry.")
        else:
            print_summary(results, now)
            alert_new(results, state, now)
            if not paper_done and now >= paper_dt:
                print("\n--- Paper trading ---")
                log = run_paper(results, capital=args.capital)
                notify(f"[Scanner] Paper portfolio update - {now:%d %b}", "\n".join(log))
                paper_done = True

        now = datetime.now(IST)
        if args.once or now >= last_dt:
            break
        must = [t for t in ([] if paper_done else [paper_dt]) + [last_dt] if t > now]
        nxt = min([now + timedelta(minutes=args.interval)] + must)
        print(f"Next scan at {nxt:%H:%M} IST")
        time.sleep(max(1, (nxt - datetime.now(IST)).total_seconds()))
    print("Done for today.")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description="NSE/BSE signal scanner + paper trader")
    ap.add_argument("--live", action="store_true", help="scan during market hours (9:30-15:25 IST)")
    ap.add_argument("--once", action="store_true", help="with --live: do a single scan now and exit")
    ap.add_argument("--interval", type=int, default=30, help="minutes between live scans (default 30)")
    ap.add_argument("--paper-time", default="15:20", help="IST time for the daily paper update in live mode")
    ap.add_argument("--alert", "--email", dest="alert", action="store_true",
                    help="send the end-of-day scan results to your alert channels")
    ap.add_argument("--test-alerts", "--test-email", dest="test_alerts", action="store_true",
                    help="send a test alert to every configured channel and exit")
    ap.add_argument("--bse", action="store_true", help="also scan tickers in bse_symbols.txt")
    ap.add_argument("--paper", action="store_true", help="update the virtual portfolio")
    ap.add_argument("--limit", type=int, default=0, help="scan only the first N symbols (testing)")
    ap.add_argument("--min-turnover", type=float, default=1e7,
                    help="min avg daily traded value in Rs (default 1 crore)")
    ap.add_argument("--capital", type=float, default=1_000_000, help="starting paper capital in Rs")
    args = ap.parse_args()

    if args.test_alerts:
        ok = notify("[Scanner] Test alert", "Your NSE/BSE scanner alerts are working.")
        sys.exit(0 if ok else 1)
    if args.live:
        return live(args)

    symbols = build_universe(args.bse)
    if args.limit:
        symbols = symbols[:args.limit]
    if not symbols:
        sys.exit("No symbols to scan.")

    frames = download_all(symbols)
    results = [r for s, d in frames.items() if (r := analyze(s, d, args.min_turnover))]
    print(f"\nAnalyzed {len(results)} liquid stocks out of {len(frames)} downloaded.")
    if not results:
        sys.exit("Nothing to report (no data or all filtered out).")

    df = pd.DataFrame(results)
    df = df.sort_values(["signal", "buy_score", "sell_score"], ascending=[True, False, False])
    out = f"signals_{datetime.now(IST):%Y-%m-%d}.csv"
    df.drop(columns=["high", "low"]).to_csv(out, index=False)

    cols = ["symbol", "close", "buy_score", "sell_score", "rsi", "reasons"]
    for sig in ("BUY", "SELL"):
        sub = df[df.signal == sig].sort_values("buy_score" if sig == "BUY" else "sell_score", ascending=False)
        print(f"\n=== {sig} signals ({len(sub)}) - top 20 ===")
        print(sub[cols].head(20).to_string(index=False) if len(sub) else "none")
    print(f"\nFull list saved to {out}")

    log = []
    if args.paper:
        print("\n--- Paper trading ---")
        log = run_paper(results, capital=args.capital)
    if args.alert:
        body = format_alert(results, datetime.now(IST))
        if log:
            body += "\n\n--- Paper trading ---\n" + "\n".join(log)
        notify(f"[Scanner] End-of-day signals - {datetime.now(IST):%d %b}", body)


if __name__ == "__main__":
    main()
