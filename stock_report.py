#!/usr/bin/env python3
"""Market Pulse.

1. Pulls prices from Yahoo Finance and ranks the top / worst stocks (plain code, no LLM).
2. One Claude call with web search gathers recent market news.
3. N independent Claude calls (in parallel) estimate next-day moves and a BTC signal.
4. Results are aggregated (median / majority vote) and posted to Telegram.

Required env vars: ANTHROPIC_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
Optional env vars: N_RUNS, NEWS_MODEL, PREDICT_MODEL, TICKERS, REPORT_TZ,
                   REPORT_TZ_LABEL, DRY_RUN=1 (print instead of sending)
"""

import html
import json
import math
import os
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from zoneinfo import ZoneInfo

import anthropic
import pandas as pd
import requests
import yfinance as yf

# ----------------------------- Config ---------------------------------------

N_RUNS = int(os.getenv("N_RUNS", "10"))
NEWS_MODEL = os.getenv("NEWS_MODEL", "claude-haiku-5-5")
PREDICT_MODEL = os.getenv("PREDICT_MODEL", "claude-haiku-5-5")
TZ = ZoneInfo(os.getenv("REPORT_TZ", "Europe/Moscow"))
TZ_LABEL = os.getenv("REPORT_TZ_LABEL", "MSK")
DRY_RUN = os.getenv("DRY_RUN") == "1"

TOP_N = 5
BTC_TICKER = "BTC-USD"

DEFAULT_UNIVERSE = (
    "AAPL MSFT NVDA AMZN GOOGL META TSLA AVGO ORCL ADBE CRM AMD INTC NFLX CSCO "
    "QCOM TXN JPM BAC WFC GS V MA BRK-B UNH JNJ LLY PFE MRK ABBV XOM CVX COP "
    "WMT COST HD MCD NKE KO PEP PG DIS BA CAT GE UPS T VZ UBER PYPL"
).split()
UNIVERSE = [t.strip().upper() for t in os.getenv("TICKERS", "").split(",") if t.strip()] or DEFAULT_UNIVERSE

# Trading-day offsets (stocks trade ~5/21/252 days per week/month/year; BTC trades daily)
STOCK_OFFSETS = {"week": 5, "month": 21, "year": 252}
BTC_OFFSETS = {"week": 7, "month": 30, "year": 365}

SIGNALS = {"BUY", "HOLD", "SELL"}
SIGNAL_EMOJI = {"BUY": "🟢", "HOLD": "🟡", "SELL": "🔴"}

# Each run gets a different analytical lens so the 10 runs are not near-duplicates.
LENSES = [
    "Give extra weight to recent momentum and trend.",
    "Give extra weight to mean reversion and overextended moves.",
    "Give extra weight to the news sentiment.",
    "Be a skeptic: assume most public news is already priced in.",
    "Take a balanced view of momentum, news and valuation.",
]

esc = html.escape


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ----------------------------- Market data ----------------------------------


def pct(new: float, old: float) -> float:
    return (new / old - 1) * 100


def build_rows(close: pd.DataFrame) -> dict:
    """Turn a DataFrame of daily closes (columns = tickers) into per-ticker stats."""
    rows = {}
    for t in close.columns:
        s = close[t].dropna()
        if len(s) < 2:
            continue
        off = BTC_OFFSETS if t == BTC_TICKER else STOCK_OFFSETS
        last = float(s.iloc[-1])

        def back(n: int) -> float:
            return float(s.iloc[max(len(s) - 1 - n, 0)])

        rows[t] = {
            "ticker": t,
            "price": last,
            "day": pct(last, float(s.iloc[-2])),
            "week": pct(last, back(off["week"])),
            "month": pct(last, back(off["month"])),
            "year": pct(last, back(off["year"])),
        }
    return rows


def fetch_market():
    """Download prices (with retries: Yahoo sometimes throttles shared CI IPs)."""
    tickers = UNIVERSE + [BTC_TICKER]
    last_err = None
    for attempt in range(3):
        try:
            data = yf.download(
                tickers, period="15mo", interval="1d",
                auto_adjust=True, progress=False, threads=True,
            )
            close = data["Close"]
            if close.dropna(how="all").empty:
                raise RuntimeError("empty price table")
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            log(f"Price download failed (attempt {attempt + 1}/3): {e}")
            time.sleep(5 * (attempt + 1))
    else:
        raise RuntimeError(f"Market data download failed: {last_err}")

    rows = build_rows(close)
    if BTC_TICKER not in rows:
        raise RuntimeError("No BTC data")
    stock_cols = [t for t in UNIVERSE if t in close.columns]
    as_of = close[stock_cols].dropna(how="all").index[-1].date()
    return rows, as_of


# ----------------------------- Claude helpers -------------------------------


def parse_json(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in reply")
    return json.loads(text[start : end + 1])


def ask_claude(client, model, system, user, tools=None, max_tokens=3000) -> str:
    messages = [{"role": "user", "content": user}]
    kwargs = {"tools": tools} if tools else {}
    resp = None
    for _ in range(5):  # web search can pause long turns; continue them
        resp = client.messages.create(
            model=model, max_tokens=max_tokens, system=system, messages=messages, **kwargs
        )
        if resp.stop_reason == "pause_turn":
            messages.append({"role": "assistant", "content": resp.content})
            continue
        break
    return "".join(b.text for b in resp.content if b.type == "text")


def ask_json(client, *args, retries=2, **kwargs) -> dict:
    last_err = None
    for _ in range(retries + 1):
        try:
            return parse_json(ask_claude(client, *args, **kwargs))
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise last_err


# ----------------------------- News (one shared search) ---------------------


def gather_news(client, picks: list, now: datetime) -> list:
    names = ", ".join(r["ticker"] for r in picks)
    system = (
        "You are a financial news researcher. Use web search to find the most "
        "market-moving news from the last 24 hours about the listed stocks, the US "
        "market overall, and Bitcoin. Use only URLs that appeared in your search "
        "results. Reply with ONLY a JSON object, no other text:\n"
        '{"news": [{"headline": "...", "url": "https://...", '
        '"tickers": ["NVDA"], "impact": "positive|negative|neutral"}]}\n'
        "Return 5 to 8 items, most important first."
    )
    user = f"Now: {now:%Y-%m-%d %H:%M} {TZ_LABEL}. Stocks of interest: {names}, plus BTC."
    tools = [{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}]
    try:
        data = ask_json(client, NEWS_MODEL, system, user, tools=tools)
    except Exception as e:  # noqa: BLE001
        log(f"News step failed: {e}")
        return []
    items = []
    for n in data.get("news", []):
        if not isinstance(n, dict):
            continue
        url, headline = str(n.get("url", "")), str(n.get("headline", "")).strip()
        if url.startswith(("http://", "https://")) and headline:
            items.append({**n, "url": url, "headline": headline[:140]})
    return items


# ----------------------------- Predictions (N parallel runs) ----------------


def run_prediction(client, i: int, payload: dict):
    system = (
        "You are a cautious market analyst. You estimate, you do not claim certainty. "
        "Reply with ONLY a JSON object, no other text:\n"
        '{"stocks": {"TICKER": {"expected_day_pct": 0.0}}, '
        '"btc": {"expected_day_pct": 0.0, "signal": "BUY|HOLD|SELL", '
        '"reason": "max 20 words"}}\n'
        "expected_day_pct is your estimate of the percent price change over the next "
        "trading day. Include every stock given. signal is BUY, HOLD or SELL for Bitcoin."
    )
    user = f"{LENSES[i % len(LENSES)]}\n\nData:\n{json.dumps(payload)}"
    try:
        return ask_json(client, PREDICT_MODEL, system, user, max_tokens=1500)
    except Exception as e:  # noqa: BLE001
        log(f"Run {i + 1} failed: {e}")
        return None


# ----------------------------- Aggregation ----------------------------------


def num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def aggregate(runs: list, tickers: list) -> dict:
    expected = {}
    for t in tickers:
        vals = []
        for r in runs:
            stocks = r.get("stocks") if isinstance(r.get("stocks"), dict) else {}
            entry = stocks.get(t) if isinstance(stocks.get(t), dict) else {}
            v = num(entry.get("expected_day_pct"))
            if v is not None:
                vals.append(v)
        expected[t] = statistics.median(vals) if vals else None

    btc_runs = []
    for r in runs:
        b = r.get("btc") if isinstance(r.get("btc"), dict) else {}
        sig = str(b.get("signal", "")).strip().upper()
        if sig in SIGNALS:
            btc_runs.append((sig, num(b.get("expected_day_pct")), str(b.get("reason", "")).strip()))

    result = {"expected": expected, "btc_expected": None,
              "signal": None, "confidence": None, "reason": ""}
    if not btc_runs:
        return result

    counts = Counter(sig for sig, _, _ in btc_runs)
    top = max(counts.values())
    leaders = [s for s, c in counts.items() if c == top]
    signal = "HOLD" if "HOLD" in leaders or len(leaders) > 1 else leaders[0]
    share = counts[signal] / len(btc_runs)

    btc_vals = [v for _, v, _ in btc_runs if v is not None]
    median = statistics.median(btc_vals) if btc_vals else None

    # Reason: from a run that agrees with the majority and sits closest to the median.
    agreeing = [(v, why) for sig, v, why in btc_runs if sig == signal and why]
    reason = ""
    if agreeing:
        if median is not None:
            agreeing.sort(key=lambda x: abs(x[0] - median) if x[0] is not None else 1e9)
        reason = agreeing[0][1]

    result.update(
        btc_expected=median,
        signal=signal,
        confidence="high" if share >= 0.8 else "medium" if share >= 0.6 else "low",
        reason=reason,
    )
    return result


# ----------------------------- Message --------------------------------------


def arrow(x: float) -> str:
    return "🔺" if x >= 0 else "🔻"


def fmt_pct(x: float) -> str:
    return f"{x:+.1f}%"


def stock_block(emoji: str, title: str, items: list, expected: dict) -> str:
    lines = [f"{emoji} <b>{title}</b>"]
    for i, r in enumerate(items, 1):
        exp = expected.get(r["ticker"])
        exp_txt = fmt_pct(exp) if exp is not None else "n/a"
        lines.append(f"{i}. <b>{esc(r['ticker'])}</b> · ${r['price']:,.2f}")
        lines.append(f"    {arrow(r['day'])} <b>{fmt_pct(r['day'])}</b> today · exp. {exp_txt}")
        lines.append(
            f"    <i>W {fmt_pct(r['week'])} · M {fmt_pct(r['month'])} · Y {fmt_pct(r['year'])}</i>"
        )
    return "\n".join(lines)


def build_message(now, as_of, top, worst, btc, agg, news, ok_runs, total_runs) -> str:
    parts = [f"📊 <b>Market Pulse</b> · {now:%a %d %b, %H:%M} {TZ_LABEL}"]
    if as_of < now.date():
        parts[0] += f"\n<i>Stocks: last close {as_of:%d %b}</i>"

    parts.append(stock_block("🟢", f"Top {TOP_N} Stocks", top, agg["expected"]))
    parts.append(stock_block("🔴", f"Worst {TOP_N} Stocks", worst, agg["expected"]))

    btc_exp = fmt_pct(agg["btc_expected"]) if agg["btc_expected"] is not None else "n/a"
    btc_lines = [
        f"₿ <b>Bitcoin</b> · ${btc['price']:,.0f}",
        f"{arrow(btc['day'])} <b>{fmt_pct(btc['day'])}</b> today · exp. {btc_exp}",
        f"<i>W {fmt_pct(btc['week'])} · M {fmt_pct(btc['month'])} · Y {fmt_pct(btc['year'])}</i>",
    ]
    if agg["signal"]:
        sig = agg["signal"]
        btc_lines.append(
            f"\nSignal: {SIGNAL_EMOJI[sig]} <b>{sig}</b> (confidence: {agg['confidence']})"
        )
        if agg["reason"]:
            btc_lines.append(f"Why: {esc(agg['reason'])}")
    parts.append("\n".join(btc_lines))

    if news:
        lines = ["📰 <b>News</b>"]
        for n in news[:3]:
            lines.append(f'• <a href="{esc(n["url"], quote=True)}">{esc(n["headline"])}</a>')
        parts.append("\n".join(lines))

    parts.append(
        f"<i>Not financial advice. Forecasts are model estimates ({ok_runs}/{total_runs} runs).</i>"
    )
    return "\n\n".join(parts)


def send_telegram(text: str) -> None:
    token, chat_id = os.environ["TELEGRAM_BOT_TOKEN"], os.environ["TELEGRAM_CHAT_ID"]
    r = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat_id, "text": text, "parse_mode": "HTML",
              "disable_web_page_preview": True},
        timeout=30,
    )
    if not r.ok:  # don't use raise_for_status(): its message contains the bot token URL
        sys.exit(f"Telegram error {r.status_code}: {r.text}")


# ----------------------------- Main -----------------------------------------


def main() -> None:
    now = datetime.now(TZ)
    rows, as_of = fetch_market()
    btc = rows[BTC_TICKER]
    stocks = sorted((rows[t] for t in UNIVERSE if t in rows), key=lambda r: r["day"], reverse=True)
    top, worst = stocks[:TOP_N], stocks[::-1][:TOP_N]
    picks = top + worst

    client = anthropic.Anthropic()
    news = gather_news(client, picks, now)

    payload = {
        "as_of": f"{now:%Y-%m-%d %H:%M} {TZ_LABEL}",
        "stocks": [{k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()} for r in picks],
        "btc": {k: (round(v, 2) if isinstance(v, float) else v) for k, v in btc.items()},
        "news": [{k: n.get(k) for k in ("headline", "tickers", "impact")} for n in news],
    }
    with ThreadPoolExecutor(max_workers=N_RUNS) as pool:
        results = list(pool.map(lambda i: run_prediction(client, i, payload), range(N_RUNS)))
    runs = [r for r in results if r]
    if not runs:
        sys.exit("All prediction runs failed; nothing to send.")

    agg = aggregate(runs, [r["ticker"] for r in picks])
    text = build_message(now, as_of, top, worst, btc, agg, news, len(runs), N_RUNS)

    if DRY_RUN:
        print(text)
    else:
        send_telegram(text)
        log("Sent.")


if __name__ == "__main__":
    main()
