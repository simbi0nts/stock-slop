#!/usr/bin/env python3
"""Слоп Отчёт.

1. Цены: Yahoo Finance (US-акции и BTC) и API Мосбиржи (ISS, российские акции).
   Топ и аутсайдеры считаются обычным кодом, без LLM.
2. Один вызов Claude с веб-поиском собирает свежие новости (на русском).
3. N независимых вызовов Claude (параллельно) оценивают движение на день и сигнал по BTC.
4. Результаты агрегируются (медиана / голосование) и уходят в Telegram.

Обязательные env: ANTHROPIC_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
Необязательные env: N_RUNS, NEWS_MODEL, PREDICT_MODEL, TICKERS (US), LOCAL_TICKERS (MOEX),
                    REPORT_TZ, REPORT_TZ_LABEL, DRY_RUN=1 (печатать вместо отправки)
"""

import html
import json
import math
import os
import random
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
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
TG_LIMIT = 4000  # у Telegram лимит 4096 символов, берём с запасом

DEFAULT_US = (
    "AAPL MSFT NVDA AMZN GOOGL META TSLA AVGO ORCL ADBE CRM AMD INTC NFLX CSCO "
    "QCOM TXN JPM BAC WFC GS V MA BRK-B UNH JNJ LLY PFE MRK ABBV XOM CVX COP "
    "WMT COST HD MCD NKE KO PEP PG DIS BA CAT GE UPS T VZ UBER PYPL"
).split()
DEFAULT_RU = (
    "SBER GAZP LKOH ROSN NVTK GMKN YDEX T MTSS MGNT TATN SNGS PLZL CHMF NLMK MAGN "
    "ALRS AFLT VTBR MOEX PHOR RTKM IRAO HYDR AFKS PIKK OZON RUAL FLOT CBOM SVCB"
).split()


def tickers_from_env(name: str, default: list) -> list:
    custom = [t.strip().upper() for t in os.getenv(name, "").split(",") if t.strip()]
    return custom or default


US_UNIVERSE = tickers_from_env("TICKERS", DEFAULT_US)
RU_UNIVERSE = tickers_from_env("LOCAL_TICKERS", DEFAULT_RU)

MOEX_CANDLES = "https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities/{t}/candles.json"

# Торговые дни назад для недели / месяца / года (акции ~5/21/252, BTC торгуется ежедневно)
STOCK_OFFSETS = {"week": 5, "month": 21, "year": 252}
BTC_OFFSETS = {"week": 7, "month": 30, "year": 365}

SIGNALS = {"BUY", "HOLD", "SELL"}
SIGNAL_RU = {"BUY": ("🟢", "СКУПАЕМ"), "HOLD": ("🟡", "ХОЛДИМ"), "SELL": ("🔴", "ПРОДАЁМ")}

RU_DAYS = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
RU_MONTHS = [
    "Января", "Февраля", "Марта", "Апреля", "Мая", "Июня",
    "Июля", "Августа", "Сентября", "Октября", "Ноября", "Декабря",
]

# Каждый прогон получает свой «взгляд», чтобы 10 ответов не были копиями друг друга.
LENSES = [
    "Give extra weight to recent momentum and trend.",
    "Give extra weight to mean reversion and overextended moves.",
    "Give extra weight to the news sentiment.",
    "Be a skeptic: assume most public news is already priced in.",
    "Take a balanced view of momentum, news and valuation.",
]

# Мемные цитаты в стиле «цитат Стетхема». Добавляйте свои сюда.
QUOTES = [
    "Не бывает плохих акций. Бывает плохой вход.",
    "Волк не смотрит на график. Волк смотрит вперёд.",
    "Хомяк паникует на дне. Волк уже там и ждёт.",
    "Пока ты ждал идеальную точку входа, рынок уже вышел.",
    "Настоящий мужик не ловит падающий нож. Он делает вид, что так и задумывал.",
    "Купил на хаях — не лох, а долгосрочный инвестор.",
    "Лучше сидеть в кэше и молчать, чем сидеть в минусе и называть это стратегией.",
    "Не кричи «ракета». Подожди, пока долетит.",
    "Тот, кто продал на дне, всегда знает, где было дно. Теперь.",
    "Диверсификация — это когда убытки в разных тикерах.",
    "Купил — держи. Не купил — не рассказывай, что хотел.",
    "Рынок как дорога: без ям не бывает, и в каждой кто-то уже стоит.",
]

esc = html.escape


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ----------------------------- Market data ----------------------------------


def pct(new: float, old: float) -> float:
    return (new / old - 1) * 100


def build_rows(close: pd.DataFrame) -> dict:
    """DataFrame дневных закрытий (колонки = тикеры) -> статистика по каждому тикеру."""
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


def fetch_us_and_btc() -> dict:
    """US-акции и BTC с Yahoo (с ретраями: Yahoo иногда душит общие IP раннеров)."""
    tickers = US_UNIVERSE + [BTC_TICKER]
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
    return rows


def moex_series(t: str, start: date, end: date):
    """Дневные закрытия одной бумаги с Мосбиржи (ISS). None, если данных нет."""
    r = requests.get(
        MOEX_CANDLES.format(t=t),
        params={"from": start.isoformat(), "till": end.isoformat(),
                "interval": 24, "iss.meta": "off"},
        timeout=20,
    )
    r.raise_for_status()
    c = r.json()["candles"]
    if not c["data"]:
        return None
    df = pd.DataFrame(c["data"], columns=c["columns"])
    idx = pd.DatetimeIndex(pd.to_datetime(df["begin"]).dt.normalize())
    s = pd.Series(df["close"].astype(float).values, index=idx)
    return s[~s.index.duplicated(keep="last")]


def fetch_ru() -> dict:
    """Российские акции. Если Мосбиржа недоступна, отчёт уходит без этих блоков."""
    end = date.today()
    start = end - timedelta(days=400)

    def one(t):
        try:
            return t, moex_series(t, start, end)
        except Exception as e:  # noqa: BLE001
            log(f"MOEX {t} failed: {e}")
            return t, None

    series = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for t, s in pool.map(one, RU_UNIVERSE):
            if s is not None and len(s) >= 2:
                series[t] = s
    if not series:
        log("MOEX: no data, local blocks will be skipped")
        return {}
    return build_rows(pd.DataFrame(series))


def split_top_worst(items: list):
    """Топ и аутсайдеры по дневному росту, без пересечений."""
    items = sorted(items, key=lambda r: r["day"], reverse=True)
    k = min(TOP_N, len(items) // 2)
    return items[:k], items[::-1][:k]


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
    for _ in range(5):  # веб-поиск может приостановить длинный ход; продолжаем
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


# ----------------------------- News (один общий поиск) ----------------------


def gather_news(client, picks: list, now: datetime) -> list:
    names = ", ".join(r["ticker"] for r in picks)
    system = (
        "You are a financial news researcher. Use web search to find the most "
        "market-moving news from the last 24 hours that explains the moves of the listed "
        "stocks (US stocks and Russian MOEX stocks), the US and Russian markets overall "
        "(including the Bank of Russia key rate and the ruble), and Bitcoin. "
        "Use only URLs that appeared in your search results. "
        "Write every headline in RUSSIAN (translate if the source is in another language), "
        "short and factual, at most 90 characters. "
        "Reply with ONLY a JSON object, no other text:\n"
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
            items.append({**n, "url": url, "headline": headline[:120]})
    return items


# ----------------------------- Predictions (N параллельных прогонов) --------


def run_prediction(client, i: int, payload: dict):
    system = (
        "You are a cautious market analyst. You estimate, you do not claim certainty. "
        "Reply with ONLY a JSON object, no other text:\n"
        '{"stocks": {"TICKER": {"expected_day_pct": 0.0}}, '
        '"btc": {"expected_day_pct": 0.0, "signal": "BUY|HOLD|SELL", '
        '"reason": "max 20 words, in RUSSIAN"}}\n'
        "The stocks are US stocks (market US, USD) and Russian MOEX stocks (market RU, RUB). "
        "expected_day_pct is your estimate of the percent price change over the next "
        "trading day. Include every stock given. signal is BUY, HOLD or SELL for Bitcoin. "
        "The JSON keys and the signal stay in English; only the reason text is Russian."
    )
    user = f"{LENSES[i % len(LENSES)]}\n\nData:\n{json.dumps(payload, ensure_ascii=False)}"
    try:
        return ask_json(client, PREDICT_MODEL, system, user, max_tokens=2500)
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

    # Причина: из прогона, согласного с большинством и ближайшего к медиане.
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
    return "⬆️" if x >= 0 else "⬇️"


def fmt_pct(x: float) -> str:
    s = f"{x:+.1f}%"
    return "+0.0%" if s == "-0.0%" else s


def fmt_price(p: float, rub: bool) -> str:
    if rub:
        return f"{p:,.2f}".replace(",", " ") + "р"
    return f"${p:,.2f}"


def ru_date(now: datetime) -> str:
    return f"{RU_DAYS[now.weekday()]} {now.day} {RU_MONTHS[now.month - 1]}, {now:%H:%M}"


def stock_block(emoji: str, title: str, items: list, expected: dict, rub: bool) -> str:
    lines = [f"{emoji} <b>{title}</b>"]
    for i, r in enumerate(items, 1):
        exp = expected.get(r["ticker"])
        exp_txt = fmt_pct(exp) if exp is not None else "н/д"
        lines.append(f"{i}. <b>{esc(r['ticker'])}</b> · {fmt_price(r['price'], rub)}")
        lines.append(f"    {arrow(r['day'])} <b>{fmt_pct(r['day'])}</b> за сегодня · ожидаем ещё {exp_txt}")
        lines.append(f"    <i>за месяц было {fmt_pct(r['month'])} · за год было {fmt_pct(r['year'])}</i>")
    return "\n".join(lines)


def build_message(now, us_top, us_worst, ru_top, ru_worst, btc, agg, news, quote=None) -> str:
    quote = quote or random.choice(QUOTES)
    exp = agg["expected"]

    def assemble(news_count: int) -> str:
        parts = [f"📊 <b>Слоп Отчёт</b> · {ru_date(now)} {TZ_LABEL}"]
        if us_top:
            parts.append(stock_block("🟢", "Покупаем забугорные акции", us_top, exp, rub=False))
            parts.append(stock_block("🔴", "Не покупаем забугорные акции", us_worst, exp, rub=False))
        if ru_top:
            parts.append(stock_block("🟢", "Покупаем местные акции", ru_top, exp, rub=True))
            parts.append(stock_block("🔴", "Не покупаем местные акции", ru_worst, exp, rub=True))

        btc_exp = fmt_pct(agg["btc_expected"]) if agg["btc_expected"] is not None else "н/д"
        btc_lines = [
            f"₿ <b>Биток</b> · ${btc['price']:,.0f}",
            f"{arrow(btc['day'])} <b>{fmt_pct(btc['day'])}</b> за сегодня · ожидаем ещё {btc_exp}",
            f"<i>за месяц было {fmt_pct(btc['month'])} · за год было {fmt_pct(btc['year'])}</i>",
        ]
        if agg["signal"]:
            emoji, word = SIGNAL_RU[agg["signal"]]
            btc_lines.append(f"Прогноз: {emoji} <b>{word}</b>")
            if agg["reason"]:
                btc_lines.append(f"Почему: {esc(agg['reason'])}")
        parts.append("\n".join(btc_lines))

        if news_count:
            lines = ["📰 <b>Новости</b>"]
            for n in news[:news_count]:
                lines.append(f'• <a href="{esc(n["url"], quote=True)}">{esc(n["headline"])}</a>')
            parts.append("\n".join(lines))

        parts.append(f"<i>«{esc(quote)}» — Стетхем</i>")
        return "\n\n".join(parts)

    # Не больше 5 новостей, и сокращаем их, если сообщение не влезает в лимит Telegram.
    text = assemble(0)
    for count in range(min(5, len(news)), -1, -1):
        text = assemble(count)
        if len(text) <= TG_LIMIT:
            break
    return text


def send_telegram(text: str) -> None:
    token, chat_id = os.environ["TELEGRAM_BOT_TOKEN"], os.environ["TELEGRAM_CHAT_ID"]
    r = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat_id, "text": text, "parse_mode": "HTML",
              "disable_web_page_preview": True},
        timeout=30,
    )
    if not r.ok:  # не используем raise_for_status(): в его тексте есть токен бота
        sys.exit(f"Telegram error {r.status_code}: {r.text}")


# ----------------------------- Main -----------------------------------------


def rnd(r: dict, **extra) -> dict:
    return {**{k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}, **extra}


def main() -> None:
    now = datetime.now(TZ)
    rows = fetch_us_and_btc()
    ru_rows = fetch_ru()
    btc = rows[BTC_TICKER]

    us_top, us_worst = split_top_worst([rows[t] for t in US_UNIVERSE if t in rows])
    ru_top, ru_worst = split_top_worst(list(ru_rows.values()))
    us_picks, ru_picks = us_top + us_worst, ru_top + ru_worst
    picks = us_picks + ru_picks

    client = anthropic.Anthropic()
    news = gather_news(client, picks, now)

    payload = {
        "as_of": f"{now:%Y-%m-%d %H:%M} {TZ_LABEL}",
        "stocks": [rnd(r, market="US") for r in us_picks] + [rnd(r, market="RU") for r in ru_picks],
        "btc": rnd(btc),
        "news": [{k: n.get(k) for k in ("headline", "tickers", "impact")} for n in news],
    }
    with ThreadPoolExecutor(max_workers=N_RUNS) as pool:
        results = list(pool.map(lambda i: run_prediction(client, i, payload), range(N_RUNS)))
    runs = [r for r in results if r]
    if not runs:
        sys.exit("All prediction runs failed; nothing to send.")

    agg = aggregate(runs, [r["ticker"] for r in picks])
    text = build_message(now, us_top, us_worst, ru_top, ru_worst, btc, agg, news)

    if DRY_RUN:
        print(text)
    else:
        send_telegram(text)
        log("Sent.")


if __name__ == "__main__":
    main()
