#!/usr/bin/env python3
"""Слоп Отчёт.

1. Цены: Yahoo Finance (US-акции и BTC) и API Мосбиржи (ISS, российские акции).
   Топ и аутсайдеры считаются обычным кодом, без LLM.
2. Один вызов Claude с веб-поиском собирает свежие новости (на русском).
3. N независимых вызовов Claude (параллельно) оценивают движение на день и сигнал по BTC.
4. Результаты агрегируются (медиана / голосование) и уходят в Telegram.
   Блоки «Продаём» — бумаги, где падение ждёт большинство прогонов.

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
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

import anthropic
import pandas as pd
import requests
import yfinance as yf

# ----------------------------- Config ---------------------------------------

N_RUNS = max(1, int(os.getenv("N_RUNS", "10")))
NEWS_MODEL = os.getenv("NEWS_MODEL", "claude-haiku-5-5")
PREDICT_MODEL = os.getenv("PREDICT_MODEL", "claude-haiku-5-5")
TZ = ZoneInfo(os.getenv("REPORT_TZ", "Europe/Moscow"))
TZ_LABEL = os.getenv("REPORT_TZ_LABEL", "MSK")
DRY_RUN = os.getenv("DRY_RUN") == "1"

TOP_N = 3
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

# Мосбиржа использует сертификаты НУЦ Минцифры (Russian Trusted Root CA). Их нет в стандартном
# наборе Python, поэтому корневой сертификат лежит в репозитории: certs/russian_trusted_root_ca.pem
RU_CA_FILE = os.getenv(
    "RU_ROOT_CA",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "certs", "russian_trusted_root_ca_pem.crt"),
)

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
    "Если график пошёл вниз, значит, ему так надо.",
    "Не важно, сколько ты заработал. Важно, сколько ты не потерял.",
    "Биржа не казино. В казино хотя бы иногда дают выиграть.",
    "Когда все покупают, волк уже продаёт.",
    "Когда все продают, волк проверяет баланс и плачет.",
    "Не бывает позднего входа. Бывает ранний выход.",
    "Трейдер без стоп-лосса — как боксёр без подбородка.",
    "Не спорь с рынком. У него больше денег и меньше совести.",
    "Покупай страх. Продавай жадность. А если денег нет — наблюдай.",
    "Ракета без топлива — это просто дорогая палка.",
    "Волк не усредняет убытки. Волк усредняет оправдания.",
    "Если портфель красный — значит, ты патриот.",
    "Трейдинг любит терпеливых. Брокер любит всех.",
    "Не каждый памп ведёт к счастью. Некоторые ведут к ликвидации.",
    "Сильный не тот, кто купил на дне, а тот, кто признал, что купил на хаях.",
    "Не бойся коррекции. Бойся уведомления от брокера.",
    "Рынок забирает деньги у нетерпеливых и отдаёт их ещё более нетерпеливым.",
    "Когда биткоин падает, слабые продают, сильные усредняют, а умные выключают интернет.",
    "Волк не ищет иксы. Иксы сами ищут волка.",
    "Пока ты ищешь альтсезон, альты ищут твою ликвидность.",
    "Не бывает мёртвых монет. Бывают очень долгосрочные инвестиции.",
    "Если монета упала на 90%, ей осталось упасть всего на 10%.",
    "Хороший трейдер знает цену. Великий трейдер знает, когда закрыть терминал.",
    "Плечо — это когда ты хотел заработать быстрее, а ушёл раньше.",
    "Не бери кредит на крипту. Крипта сама возьмёт тебя в кредит.",
    "Стоп-лосс придуман не для слабых. Он придуман для тех, кто хочет жить.",
    "Волк не боится медведя. Волк боится маржин-колла.",
    "Если ты не знаешь, почему растёт монета, значит, ты не один такой.",
    "Настоящий инвестор не проверяет портфель каждые пять минут. Он проверяет каждые три.",
    "Не пытайся поймать дно. Даже батискаф не всегда возвращается.",
    "Когда рынок говорит «покупай», проверь, не говорит ли он это твоими деньгами.",
    "Убыток — это прибыль, которую ты ещё не придумал.",
    "Не надо паниковать. Надо было думать до покупки.",
    "Волк не читает прогнозы аналитиков. Он сам становится причиной их ошибок.",
    "Лучше пропустить памп, чем стать его топливом.",
    "Крипта не про деньги. Крипта про то, как быстро меняется настроение.",
    "Если ты купил по совету блогера, поздравляю — ты инвестировал в его отпуск.",
    "Настоящий хомяк покупает на новостях. Настоящий волк продаёт ему эти новости.",
    "Рынок не обязан делать тебя богатым. Он вообще тебя не знает.",
    "Не путай надежду с инвестиционной стратегией.",
    "Если ты держишь монету год, это уже не инвестиция, это отношения.",
    "Волк не усредняет падение. Он усредняет дистанцию до выхода.",
    "Пока ты ждёшь зелёную свечу, красная уже пишет завещание.",
    "Покупай дешево. Продавай дорого. Если не получилось — называй это опытом.",
    "Не бывает слишком большого плеча. Бывает слишком короткая жизнь депозита.",
    "Тот, кто купил биткоин на тысячу долларов, сегодня философ. Тот, кто продал, тоже философ.",
    "Секрет успеха прост: не теряй деньги быстрее, чем зарабатываешь.",
    "Волк не боится волатильности. Он просто не спит.",
    "Когда все говорят о лёгких деньгах, проверь, не твои ли это деньги.",
    "Если ты не можешь объяснить сделку за минуту, возможно, это не сделка.",
    "Не ищи сигнал там, где нет интернета.",
    "Рынок всегда прав. Даже когда он неправ, деньги всё равно его.",
    "Умный трейдер покупает по стратегии. Гений покупает по интуиции. Легенда — по ошибке.",
    "Пока ты изучал фундаментал, мемкоин сделал три икса и умер.",
    "Не надо бояться красных свечей. Они тоже имеют право на существование.",
    "Если твой портфель упал, не смотри на него сверху. Он и так внизу.",
    "Волк не покупает вершину. Он просто случайно там оказывается.",
    "Не бывает плохой монеты. Бывает плохой момент, чтобы о ней узнать.",
    "Сначала ты покупаешь крипту. Потом крипта покупает твоё спокойствие.",
    "Деньги любят тишину. Особенно когда ты сидишь в убытке.",
    "Не пытайся обогнать рынок. Он уже знает короткую дорогу.",
    "Пока ты ждёшь подтверждения тренда, тренд уже сменил прописку.",
    "Хомяк верит в проект. Волк верит в ликвидность.",
    "Никогда не инвестируй последние деньги. Оставь хотя бы на такси до работы.",
    "Если монета выросла на 1000%, это не значит, что ты опоздал. Это значит, что ты опоздал.",
    "Волк не смотрит на капитализацию. Он смотрит, кто последний держит пакет.",
    "Не продавай на панике. Паника должна сначала предложить хорошую цену.",
    "Лучше упустить прибыль, чем поймать убыток всей жизни.",
    "Рынок — это место, где терпение проверяется деньгами.",
    "Когда ты говоришь «я в долгосрок», рынок слышит «я не знаю, что делать».",
    "Не каждый холдер — инвестор. Некоторые просто забыли пароль.",
    "Успешный трейдинг начинается там, где заканчивается вера в чудо.",
    "Не верь в гарантированные иксы. Гарантирован только налог на прибыль.",
    "Если ты купил на слухах, продавай до того, как появятся факты.",
    "Волк не гонится за зелёной свечой. Он знает, что она может оказаться последней.",
    "Трейдер без плана — это турист с кредитным плечом.",
    "Самая дорогая ошибка — та, которую ты называешь временной просадкой.",
    "Не путай бычий рынок с собственной гениальностью.",
    "Когда биткоин растёт, все становятся экспертами. Когда падает, все становятся философами.",
    "Если ты не знаешь, где выйти, не удивляйся, что рынок выйдет за тебя.",
    "Хороший вход — половина дела. Хороший выход — половина депозита.",
    "Депозит не резиновый. А вот жадность безгранична.",
    "Волк не боится обвала. Он просто заранее снял деньги.",
    "Покупай то, что понимаешь. Не понимаешь — хотя бы не покупай много.",
    "Не надо быть умнее рынка. Достаточно не быть глупее собственного стопа.",
    "Каждая просадка делает тебя сильнее. Если после неё осталось что усиливать.",
    "Когда рынок падает, не ищи виноватых. Ищи кнопку закрытия позиции.",
    "Не бывает плохого дня для трейдинга. Бывают дни, когда лучше не торговать.",
    "Волк не верит в сигналы. Он верит в собственные ошибки.",
    "Пока ты искал следующий биткоин, биткоин искал твою зарплату.",
    "Если ты не можешь пережить минус 10%, не смотри на крипту.",
    "Богатство любит расчёт. Бедность любит усреднение без лимита.",
    "Хомяк держит до нуля. Волк держит до прибыли. А потом тоже до нуля.",
    "Рынок не мстит. Он просто выставляет счёт.",
    "Не покупай вершину. Она уже занята твоими знакомыми.",
    "Если твоя стратегия работает только на бычьем рынке, это не стратегия.",
    "Волк не ищет идеальную монету. Он ищет идеальный момент выйти.",
    "Не бывает бесплатного пампа. Кто-то обязательно платит.",
    "Когда ты покупаешь дип, убедись, что это не бездна.",
    "Сильные руки держат активы. Слабые руки держат надежду.",
    "Не называй убыток инвестицией в будущее. Будущее может не согласиться.",
    "Если ты купил мемкоин, не удивляйся, что он ведёт себя как мем.",
    "У каждого трейдера есть стратегия. У каждого второго — новая.",
    "Лучший индикатор рынка — это количество людей, которые уверены в его направлении.",
    "Не торгуй на эмоциях. Эмоции не платят комиссию.",
    "Когда рынок даёт шанс, не спрашивай, почему. Просто проверь риск.",
    "Волк не ждёт идеального момента. Он ждёт момента, когда риск оправдан.",
    "Не бывает вечного роста. Даже твоя зарплата когда-нибудь закончится.",
    "Если ты заработал на первой сделке, не думай, что ты избранный.",
    "Трейдинг — это когда ты платишь за опыт, а рынок не выдаёт чек.",
    "Не доверяй монете, у которой больше фанатов, чем покупателей.",
    "Хороший портфель — это когда ты можешь спать, а не когда можешь хвастаться.",
    "Не надо ловить нож. Надо знать, кто его бросил.",
    "Волк не покупает на эмоциях. Он покупает, когда эмоции уже всех разорили.",
    "Если все вокруг миллионеры, возможно, ты просто в чате с мошенниками.",
    "Сначала изучи риск. Потом изучи актив. Потом реши, стоит ли оно твоих нервов.",
    "Пока ты ждал подтверждения, твой депозит подтвердил убыток.",
    "Падающий рынок — не повод терять голову. Голова пригодится для следующей сделки.",
    "Не бывает лёгких денег. Бывают лёгкие люди, которые их отдают.",
    "Не продавай актив только потому, что он упал. Но и не женись на нём.",
    "Тот, кто купил на дне, не всегда молодец. Иногда он просто не успел продать.",
    "Если ты не понимаешь токеномику, токеномика уже понимает тебя.",
    "Волк не боится FUD. Волк боится, что FUD окажется правдой.",
    "Не путай альтсезон с сезоном сбора денег у населения.",
    "Рынок — единственное место, где люди радуются скидке и боятся её купить.",
    "Пока ты искал точку входа, точка выхода нашла тебя.",
    "Не надо кричать «туземун», пока ты ещё на парковке.",
    "Хомяк покупает мечту. Волк продаёт билет.",
    "Если ты торгуешь без риска, значит, риск торгует тобой.",
    "Никогда не ставь всё на одну монету. Даже если она называется «Последний шанс».",
    "Волк не просит рынок о прибыли. Он просто не отдаёт ему всё.",
    "Криптозима закаляет характер и уничтожает кредитную историю.",
    "Если ты называешь падение коррекцией, убедись, что коррекция согласна.",
    "Не каждый рост — бычий рынок. Иногда это просто последний вздох.",
    "Портфель должен расти. Если растёт только твой уровень стресса — меняй подход.",
    "Не пытайся заработать на каждом движении. Даже лифт иногда стоит.",
    "Волк не боится упущенной прибыли. Он боится упущенной возможности выйти.",
    "Покупка на слухах — это искусство. Продажа на фактах — выживание.",
    "Если ты вложил деньги в проект из-за красивого логотипа, ты не инвестор, ты дизайнер.",
    "Не бывает плохого прогноза. Бывает слишком много уверенности.",
    "Рынок не знает твою цену входа. И ему всё равно.",
    "Пока ты усреднял позицию, позиция усреднила твою жизнь.",
    "Не называй себя холдером, если просто не умеешь нажимать кнопку Sell.",
    "Волк не ищет дно. Волк ищет выход из леса.",
    "Если твоя прибыль существует только на скриншоте, это ещё не прибыль.",
    "Сильный трейдер умеет заработать. Мудрый умеет остановиться.",
    "Не покупай монету только потому, что она уже упала. Она может продолжить традицию.",
    "У рынка нет сердца. Зато есть комиссии.",
    "Когда ты говоришь «ещё чуть-чуть подержу», рынок уже планирует вечеринку без тебя.",
    "Не пытайся победить рынок за один день. Он тренируется круглосуточно.",
    "Волк не верит в удачу. Но иногда проверяет лотерейный билет.",
    "Твоя прибыль — это чужая ошибка. Твой убыток — тоже.",
    "Если все ждут отскок, проверь, не готовится ли рынок прыгнуть ещё ниже.",
    "Не торгуй ради адреналина. Для этого существуют американские горки.",
    "Хомяк покупает на зелёном. Волк покупает, когда зелёный уже закончился.",
    "Волк не держит монету из принципа. Принципы не оплачивают коммуналку.",
    "Пока ты ждал иксы, инфляция сделала свои.",
    "Успешный инвестор знает, когда войти. Богатый инвестор знает, когда уйти.",
    "Не бойся пропустить сделку. Бойся пропустить момент, когда пора остановиться.",
    "Рынок — это шахматы, где фигуры сами меняют правила.",
    "Если ты не можешь объяснить, зачем купил, объясни хотя бы, зачем держишь.",
    "Не бывает вечного холда. Бывает вечное сожаление.",
    "Плечо увеличивает прибыль. Но сначала оно увеличивает характер твоих проблем.",
    "Волк не смотрит на баланс каждый час. Он смотрит, сколько осталось до зарплаты.",
    "Если актив падает третий день, это не значит, что завтра он обязан извиниться.",
    "Покупай с холодной головой. Продавай без горячего сердца.",
    "Не слушай тех, кто обещает иксы. Слушай тех, кто объясняет риски.",
    "Даже самая сильная ракета однажды становится падающей звездой.",
    "Если ты потерял деньги, не теряй ещё и здравый смысл.",
    "Трейдер не должен быть прав каждый раз. Он должен оставаться в игре.",
    "Хомяк считает прибыль в процентах. Волк считает её в выводах.",
    "Не ищи лёгких денег. Ищи способ не отдавать сложные.",
    "Волк не боится медвежьего рынка. Он просто не забывает, что медведь голодный.",
    "Тот, кто знает рынок, не хвастается. Тот, кто хвастается, обычно ищет ликвидность.",
    "Пока ты ждал идеального сигнала, идеальный сигнал ушёл к другому.",
    "Не надо инвестировать в надежду. Она не торгуется на бирже.",
    "Самая стабильная монета — это та, которую ты ещё не купил.",
    "Если ты купил на дне, не спеши радоваться. Возможно, это был первый этаж подвала.",
    "Волк не спрашивает, будет ли рост. Он спрашивает, сколько готов потерять.",
    "Не каждый минус — поражение. Иногда это дорогой урок с подпиской на продолжение.",
    "Рынок любит смелых. Но особенно любит тех, кто не поставил всё.",
    "Не путай финансовую свободу с отсутствием денег на комиссии.",
    "Когда твой друг говорит «точно полетит», проверь, не продаёт ли он.",
    "Если твой прогноз не сбылся, не меняй рынок. Измени прогноз.",
    "Волк не боится красного портфеля. Он боится красного уведомления от банка.",
    "Не покупай то, что не можешь объяснить даже самому себе после трёх чашек кофе.",
    "Пока ты искал финансовую независимость, финансовая зависимость нашла тебя.",
    "Не бывает плохой стратегии. Бывает стратегия, которую ты придумал после убытка.",
    "Если ты пережил криптозиму, весна уже не кажется такой страшной.",
    "Хомяк мечтает о ламбо. Волк мечтает о том, чтобы биржа не легла.",
    "Главное в трейдинге — не сколько ты заработал, а сколько раз смог не нажать Buy.",
    "Настоящий Стетхем не ловит иксы. Он ловит такси после закрытия позиции."
]

esc = html.escape


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ----------------------------- Market data ----------------------------------


def pct(new: float, old: float) -> float:
    """Изменение new к old в процентах. Нулевая база не роняет отчёт."""
    return (new / old - 1) * 100 if old else 0.0


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


@lru_cache(maxsize=1)
def moex_verify():
    """Набор доверенных сертификатов для Мосбиржи: стандартный (certifi) + корень Минцифры.

    Проверка TLS остаётся включённой. Набор применяется только к запросам к Мосбирже.
    """
    if not os.path.isfile(RU_CA_FILE):
        log(f"MOEX: нет файла {RU_CA_FILE}, проверка сертификата Мосбиржи, скорее всего, не пройдёт")
        return True
    import certifi

    bundle = os.path.join(tempfile.mkdtemp(), "ca-bundle.pem")
    with open(bundle, "wb") as out:
        for path in (certifi.where(), RU_CA_FILE):
            with open(path, "rb") as f:
                out.write(f.read().rstrip() + b"\n")
    return bundle


def moex_series(t: str, start: date, end: date, verify=True, attempts: int = 2) -> pd.Series | None:
    """Дневные закрытия одной бумаги с Мосбиржи (ISS). None, если данных нет."""
    last_err = None
    for attempt in range(attempts):
        try:
            r = requests.get(
                MOEX_CANDLES.format(t=t),
                params={"from": start.isoformat(), "till": end.isoformat(),
                        "interval": 24, "iss.meta": "off"},
                headers={"User-Agent": "Mozilla/5.0 (compatible; stock-slop/1.0)"},
                verify=verify,
                timeout=20,
            )
            r.raise_for_status()
            c = r.json()["candles"]
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt + 1 < attempts:
                time.sleep(2 * (attempt + 1))
    else:
        raise last_err

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
    verify = moex_verify()

    def one(t):
        try:
            return t, moex_series(t, start, end, verify)
        except Exception as e:  # noqa: BLE001
            log(f"MOEX {t} failed: {str(e)[:200]}")
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


def split_sell(items: list, expected: dict, down_share: dict) -> list:
    """Бумаги, у которых больше всего шансов продолжить падение.

    Сортируем по доле прогонов с отрицательным прогнозом, при равенстве — по
    величине ожидаемого минуса. Берём только те, где минус ждёт большинство.
    """
    candidates = [
        r for r in items
        if (expected.get(r["ticker"]) or 0) < 0 and (down_share.get(r["ticker"]) or 0) > 0.5
    ]
    candidates.sort(key=lambda r: (-down_share[r["ticker"]], expected[r["ticker"]]))
    return candidates[:TOP_N]


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
    text = "".join(b.text for b in resp.content if b.type == "text")
    if not text.strip():
        raise RuntimeError(f"empty reply from model (stop_reason={resp.stop_reason})")
    return text


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
    items, seen = [], set()
    for n in data.get("news", []):
        if not isinstance(n, dict):
            continue
        url, headline = str(n.get("url", "")), str(n.get("headline", "")).strip()
        if not url.startswith(("http://", "https://")) or not headline or url in seen:
            continue
        seen.add(url)
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
    expected, down_share = {}, {}
    for t in tickers:
        vals = []
        for r in runs:
            stocks = r.get("stocks") if isinstance(r.get("stocks"), dict) else {}
            entry = stocks.get(t) if isinstance(stocks.get(t), dict) else {}
            v = num(entry.get("expected_day_pct"))
            if v is not None:
                vals.append(v)
        expected[t] = statistics.median(vals) if vals else None
        # Доля прогонов, которые ждут падение: «шанс, что бумага пойдёт вниз».
        down_share[t] = sum(v < 0 for v in vals) / len(vals) if vals else None

    btc_runs = []
    for r in runs:
        b = r.get("btc") if isinstance(r.get("btc"), dict) else {}
        sig = str(b.get("signal", "")).strip().upper()
        if sig in SIGNALS:
            btc_runs.append((sig, num(b.get("expected_day_pct")), str(b.get("reason", "")).strip()))

    result = {"expected": expected, "down_share": down_share, "btc_expected": None,
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


def stock_block(emoji: str, title: str, items: list, rub: bool) -> str:
    lines = [f"{emoji} <b>{title}</b>"]
    for i, r in enumerate(items, 1):
        lines.append(f"{i}. <b>{esc(r['ticker'])}</b> · {fmt_price(r['price'], rub)}")
        lines.append(
            f"    {arrow(r['day'])} <b>{fmt_pct(r['day'])}</b> за день "
            f"(за месяц {fmt_pct(r['month'])} · за год {fmt_pct(r['year'])})"
        )
    return "\n".join(lines)


def build_message(now, us_top, us_worst, us_sell, ru_top, ru_worst, ru_sell,
                  btc, agg, news, quote=None, ru_failed=False) -> str:
    quote = quote or random.choice(QUOTES)

    def assemble(news_count: int, item_limit: int) -> str:
        parts = [f"📊 <b>Слоп Отчёт</b> · {ru_date(now)} {TZ_LABEL}"]
        blocks = [
            ("🟢", "Покупаем забугорные акции", us_top, False),
            ("🔴", "Не покупаем забугорные акции", us_worst, False),
            ("💰", "Продаём забугорные акции", us_sell, False),
            ("🟢", "Покупаем местные акции", ru_top, True),
            ("🔴", "Не покупаем местные акции", ru_worst, True),
            ("💰", "Продаём местные акции", ru_sell, True),
        ]
        for emoji, title, items, rub in blocks:
            picked = items[:item_limit]
            if picked:
                parts.append(stock_block(emoji, title, picked, rub=rub))
        if ru_failed:
            parts.append("⚠️ <i>Местные акции: Мосбиржа не отдала данные, смотри лог</i>")

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

    # Сначала убираем новости, потом сокращаем списки бумаг — чтобы влезть в лимит Telegram.
    text = ""
    for item_limit in range(TOP_N, 0, -1):
        for count in range(min(5, len(news)), -1, -1):
            text = assemble(count, item_limit)
            if len(text) <= TG_LIMIT:
                if item_limit < TOP_N:
                    log(f"Report trimmed to {item_limit} stocks per block to fit Telegram")
                return text
    log(f"Report is {len(text)} chars, over the {TG_LIMIT} limit even after trimming")
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
        "news": [{k: n.get(k) for k in ("headline", "tickers", "impact") if n.get(k) is not None}
                 for n in news],
    }
    with ThreadPoolExecutor(max_workers=N_RUNS) as pool:
        results = list(pool.map(lambda i: run_prediction(client, i, payload), range(N_RUNS)))
    runs = [r for r in results if r]
    if not runs:
        sys.exit("All prediction runs failed; nothing to send.")

    agg = aggregate(runs, [r["ticker"] for r in picks])
    us_sell = split_sell(us_picks, agg["expected"], agg["down_share"])
    ru_sell = split_sell(ru_picks, agg["expected"], agg["down_share"])
    text = build_message(now, us_top, us_worst, us_sell, ru_top, ru_worst, ru_sell,
                         btc, agg, news, ru_failed=not ru_rows)

    if DRY_RUN:
        print(text)
    else:
        send_telegram(text)
        log("Sent.")


if __name__ == "__main__":
    main()
