"""
Scanner Bot V17
===============
Key changes from V16:
  - Dropped the noisy vector/pattern-matching layer (insufficient data to be useful)
  - Dynamic weights removed — fixed, well-tested weights only
  - Signals now require CONFIRMED candle close (not live candle)
  - Primary timeframe upgraded to 15m (less noise than 5m)
  - Tighter RSI windows: 45-65 buys, 35-55 sells
  - BTC filter hardened: bear BTC = no longs; bull BTC = no shorts
  - Minimum 3 hard-gate conditions required (trend alignment, PA, VWAP, ADX)
  - Trend filter requires 15m + 1h + 4h alignment (was 15m + 1h only)
  - Minimum R:R enforced at 1.8 (was implicit)
  - register_loss() bug fixed (missing asset_type arg)
  - Removed redundant pa_long/short_ok check (duplicate of liq/BOS)
  - Score threshold raised: A grade >= 13, A+ >= 17
  - Max 1 open signal per symbol at a time
  - Cooldown 90 min after any loss (was 60)
"""


import os
import time
import csv
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

def now_ct():
    """Return current time in US Central (CT) without requiring tzdata."""
    from datetime import datetime, timedelta
    now_utc = datetime.utcnow()
    year = now_utc.year
    # DST: second Sunday of March -> first Sunday of November
    march1 = datetime(year, 3, 1)
    first_sun_march = march1 + timedelta(days=(6 - march1.weekday()) % 7)
    cdt_start = first_sun_march + timedelta(weeks=1)
    nov1 = datetime(year, 11, 1)
    cdt_end = nov1 + timedelta(days=(6 - nov1.weekday()) % 7)
    is_cdt = cdt_start <= now_utc.replace(hour=2, minute=0, second=0) < cdt_end
    offset = 5 if is_cdt else 6  # CDT=UTC-5, CST=UTC-6
    return now_utc - timedelta(hours=offset)

from dotenv import load_dotenv
from pybit.unified_trading import HTTP

# =========================================================
# LOAD ENV
# =========================================================
load_dotenv()

BOT_TOKEN        = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID          = os.getenv("TELEGRAM_CHAT_ID")
BYBIT_API_KEY    = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")

if not BOT_TOKEN or not CHAT_ID:
    raise ValueError("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in environment variables.")

# =========================================================
# BYBIT SESSION
# =========================================================
bybit = HTTP(
    testnet=False,
    api_key=BYBIT_API_KEY,
    api_secret=BYBIT_API_SECRET,
)

# =========================================================
# SETTINGS
# =========================================================
CHECK_EVERY_SECONDS = 60

SIGNALS_FILE    = "trade_signals_v17.csv"
DAILY_LOCK_FILE = "daily_risk_lock_v17.csv"
PAIR_STATS_FILE = "pair_stats_v17.csv"

sent_alerts: set = set()

# -- Account / Risk ──────────────────────────────────────
ACCOUNT_BALANCE             = 1000.0
RISK_PER_TRADE_PCT          = 0.01          # 1% per trade
MAX_DAILY_RISK_PCT          = 0.03          # 3% daily max
MAX_LOSSES_PER_DAY          = 2
COOLDOWN_AFTER_LOSS_MINUTES = 90            # up from 60
MIN_RR                      = 1.8           # minimum reward:risk ratio
A_PLUS_ONLY_MODE            = False

# -- Coin quality filter ──────────────────────────────────
MIN_COIN_WINRATE_TO_TRADE   = 40.0          # up from 35
MIN_TRADES_FOR_COIN_FILTER  = 8             # up from 5

# -- Signal score thresholds ──────────────────────────────
SCORE_A_PLUS = 17
SCORE_A      = 13

# -- Diagnostic ──────────────────────────────────────────
DIAGNOSTIC_EVERY_N_CYCLES = 0   # 0 = disabled (end-of-day report only)

# -- Watchlists ───────────────────────────────────────────
STOCK_TICKERS = [
    "AAPL", "TSLA", "NVDA", "SPY", "QQQ",
    "AMD",  "META", "AMZN", "MSFT", "PLTR",
]

CRYPTO_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT",
    "DOGEUSDT","BNBUSDT", "AVAXUSDT","LINKUSDT",
    "ADAUSDT", "LTCUSDT", "DOTUSDT", "ATOMUSDT",
    "NEARUSDT","OPUSDT",
]

# -- Scoring weights (fixed, transparent) ─────────────────
WEIGHTS = {
    "trend_4h":    4,
    "trend_1h":    3,
    "trend_15m":   2,
    "adx_ok":      2,
    "adx_dir":     1,
    "liq_sweep":   3,
    "bos":         2,
    "engulf":      2,
    "pin_bar":     2,
    "vwap":        1,
    "rsi":         1,
    "volume":      1,
}

# -- Regime thresholds ────────────────────────────────────
REGIME_ADX_TRENDING = 22
REGIME_ADX_RANGING  = 18

# =========================================================
# TELEGRAM
# =========================================================
def send_telegram(text: str):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        r = requests.post(url, data={"chat_id": CHAT_ID, "text": text}, timeout=10)
        if not r.ok:
            print("Telegram error:", r.text[:200])
    except Exception as e:
        print("Telegram exception:", e)

# =========================================================
# INDICATORS
# =========================================================
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)
    close = df["close"].astype(float)
    prev  = close.shift(1)
    tr    = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()

def compute_adx(df: pd.DataFrame, period: int = 14):
    high  = df["high"].astype(float).reset_index(drop=True)
    low   = df["low"].astype(float).reset_index(drop=True)
    close = df["close"].astype(float).reset_index(drop=True)

    up   = high.diff().clip(lower=0)
    down = (-low.diff()).clip(lower=0)
    mask = up > down
    dm_p = up.where(mask, 0.0)
    dm_m = down.where(~mask, 0.0)

    prev  = close.shift(1)
    tr    = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    atr_s = tr.ewm(com=period - 1, min_periods=period).mean()

    di_p  = 100 * (dm_p.ewm(com=period - 1, min_periods=period).mean() / (atr_s + 1e-9))
    di_m  = 100 * (dm_m.ewm(com=period - 1, min_periods=period).mean() / (atr_s + 1e-9))
    dx    = 100 * ((di_p - di_m).abs() / (di_p + di_m + 1e-9))
    adx   = dx.ewm(com=period - 1, min_periods=period).mean()
    return adx, di_p, di_m

def anchored_vwap(df: pd.DataFrame, lookback: int = 50) -> float:
    d  = df.tail(lookback).copy()
    tp = (d["high"] + d["low"] + d["close"]) / 3.0
    return float((tp * d["volume"]).sum() / (d["volume"].sum() + 1e-9))

# =========================================================
# DATA FETCHERS
# =========================================================
def get_bybit_klines(symbol: str, interval: str, limit: int = 300) -> pd.DataFrame | None:
    try:
        resp = bybit.get_kline(
            category="linear", symbol=symbol,
            interval=interval, limit=limit,
        )
        data = resp["result"]["list"]
        if not data:
            return None
        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume", "turnover"
        ])[["timestamp", "open", "high", "low", "close", "volume"]]
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit="ms")
        df = df.sort_values("timestamp").reset_index(drop=True)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        return df
    except Exception as e:
        print(f"{symbol} Bybit error: {e}")
        return None


def get_yf(ticker: str, interval: str, period: str, retries: int = 3) -> pd.DataFrame:
    """
    Safe yfinance fetch (NO custom session — required for curl_cffi)
    """
    last_exc = None

    for attempt in range(1, retries + 1):
        try:
            tk = yf.Ticker(ticker)
            df = tk.history(period=period, interval=interval, auto_adjust=True)

            if df.empty:
                raise ValueError("Empty dataframe")

            # Normalize to lowercase to match rest of codebase
            df.columns = [c.lower() for c in df.columns]
            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            return df

        except Exception as e:
            last_exc = e
            wait = 2 ** attempt
            time.sleep(wait)

    return pd.DataFrame()
# =========================================================
# MARKET REGIME
# =========================================================
REGIME_TRENDING = "TRENDING"
REGIME_RANGING  = "RANGING"
REGIME_VOLATILE = "VOLATILE"

def detect_regime(df: pd.DataFrame) -> str:
    if len(df) < 50:
        return REGIME_RANGING
    adx_s, _, _ = compute_adx(df)
    adx = float(adx_s.iloc[-1]) if pd.notna(adx_s.iloc[-1]) else 0.0

    atr_s   = compute_atr(df)
    atr_now = float(atr_s.iloc[-1]) if pd.notna(atr_s.iloc[-1]) else 0.0
    atr_75  = float(atr_s.dropna().tail(50).quantile(0.75))
    expanding = atr_now > atr_75

    if adx >= REGIME_ADX_TRENDING:
        return REGIME_TRENDING
    elif adx < REGIME_ADX_RANGING and expanding:
        return REGIME_VOLATILE
    else:
        return REGIME_RANGING

def regime_score_penalty(regime: str) -> float:
    return {REGIME_TRENDING: 0, REGIME_RANGING: 3, REGIME_VOLATILE: 5}.get(regime, 0)

# =========================================================
# PRICE ACTION
# =========================================================
def liq_sweep_long(df: pd.DataFrame, lookback: int = 20) -> bool:
    if len(df) < lookback + 3:
        return False
    confirmed  = df.iloc[:-1]
    recent_low = float(confirmed["low"].iloc[-(lookback + 1):-1].min())
    last_closed = confirmed.iloc[-1]
    return float(last_closed["low"]) < recent_low and float(last_closed["close"]) > recent_low

def liq_sweep_short(df: pd.DataFrame, lookback: int = 20) -> bool:
    if len(df) < lookback + 3:
        return False
    confirmed   = df.iloc[:-1]
    recent_high = float(confirmed["high"].iloc[-(lookback + 1):-1].max())
    last_closed = confirmed.iloc[-1]
    return float(last_closed["high"]) > recent_high and float(last_closed["close"]) < recent_high

def bos_long(df: pd.DataFrame, lookback: int = 10) -> bool:
    if len(df) < lookback + 3:
        return False
    confirmed  = df.iloc[:-1]
    swing_high = float(confirmed["high"].iloc[-(lookback + 2):-2].max())
    return float(confirmed.iloc[-1]["close"]) > swing_high

def bos_short(df: pd.DataFrame, lookback: int = 10) -> bool:
    if len(df) < lookback + 3:
        return False
    confirmed = df.iloc[:-1]
    swing_low = float(confirmed["low"].iloc[-(lookback + 2):-2].min())
    return float(confirmed.iloc[-1]["close"]) < swing_low

def bullish_engulf(df: pd.DataFrame) -> bool:
    if len(df) < 3:
        return False
    prev = df.iloc[-3]
    curr = df.iloc[-2]
    return (
        float(prev["close"]) < float(prev["open"]) and
        float(curr["close"]) > float(curr["open"]) and
        float(curr["open"]) <= float(prev["close"]) and
        float(curr["close"]) >= float(prev["open"])
    )

def bearish_engulf(df: pd.DataFrame) -> bool:
    if len(df) < 3:
        return False
    prev = df.iloc[-3]
    curr = df.iloc[-2]
    return (
        float(prev["close"]) > float(prev["open"]) and
        float(curr["close"]) < float(curr["open"]) and
        float(curr["open"]) >= float(prev["close"]) and
        float(curr["close"]) <= float(prev["open"])
    )

def hammer(df: pd.DataFrame) -> bool:
    c   = df.iloc[-2]
    rng = float(c["high"]) - float(c["low"])
    if rng <= 0:
        return False
    body    = abs(float(c["close"]) - float(c["open"]))
    lo_wick = min(float(c["open"]), float(c["close"])) - float(c["low"])
    hi_wick = float(c["high"]) - max(float(c["open"]), float(c["close"]))
    return lo_wick >= 2.0 * body and hi_wick <= 0.25 * rng and body / rng >= 0.1

def shooting_star(df: pd.DataFrame) -> bool:
    c   = df.iloc[-2]
    rng = float(c["high"]) - float(c["low"])
    if rng <= 0:
        return False
    body    = abs(float(c["close"]) - float(c["open"]))
    hi_wick = float(c["high"]) - max(float(c["open"]), float(c["close"]))
    lo_wick = min(float(c["open"]), float(c["close"])) - float(c["low"])
    return hi_wick >= 2.0 * body and lo_wick <= 0.25 * rng and body / rng >= 0.1

def is_choppy(ema9: float, ema20: float, atr: float) -> bool:
    return atr > 0 and abs(ema9 - ema20) < (atr * 0.15)

# =========================================================
# STOPS / TARGETS
# =========================================================
def smart_stop_long(df: pd.DataFrame, atr: float, lookback: int = 10) -> float:
    confirmed = df.iloc[:-1]
    swing_low = float(confirmed["low"].iloc[-(lookback + 2):-1].min())
    return round(swing_low - 0.2 * atr, 5)

def smart_stop_short(df: pd.DataFrame, atr: float, lookback: int = 10) -> float:
    confirmed  = df.iloc[:-1]
    swing_high = float(confirmed["high"].iloc[-(lookback + 2):-1].max())
    return round(swing_high + 0.2 * atr, 5)

def calc_rr(entry: float, stop: float, target: float) -> float:
    risk = abs(entry - stop)
    if risk <= 0:
        return 0.0
    return round(abs(target - entry) / risk, 2)

# =========================================================
# POSITION SIZING
# =========================================================
_QTY_PRECISION = {
    "BTCUSDT": 3, "ETHUSDT": 2, "BNBUSDT": 2, "LTCUSDT": 2,
    "SOLUSDT": 1, "AVAXUSDT": 1, "LINKUSDT": 1, "ATOMUSDT": 1,
    "NEARUSDT": 1, "OPUSDT": 1,
}
_QTY_FALLBACK = {
    "BTCUSDT": 0.001, "ETHUSDT": 0.01, "SOLUSDT": 1, "XRPUSDT": 25,
    "DOGEUSDT": 200, "BNBUSDT": 0.05, "AVAXUSDT": 2, "LINKUSDT": 2,
    "ADAUSDT": 50, "LTCUSDT": 0.5, "DOTUSDT": 10, "ATOMUSDT": 5,
    "NEARUSDT": 8, "OPUSDT": 20,
}

def get_qty(symbol: str, entry: float, stop: float) -> str:
    risk_usd  = ACCOUNT_BALANCE * RISK_PER_TRADE_PCT
    stop_dist = abs(entry - stop)
    if stop_dist <= 0:
        return str(_QTY_FALLBACK.get(symbol, 1))
    raw  = risk_usd / stop_dist
    prec = _QTY_PRECISION.get(symbol, 0)
    qty  = round(raw, prec) if prec > 0 else int(round(raw))
    return str(qty if qty > 0 else _QTY_FALLBACK.get(symbol, 1))

# =========================================================
# BTC BIAS
# =========================================================
def get_btc_bias() -> str:
    try:
        df = get_bybit_klines("BTCUSDT", "60", 100)
        if df is None or len(df) < 50:
            return "neutral"
        df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
        df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
        price = float(df.iloc[-1]["close"])
        e20   = float(df.iloc[-1]["ema20"])
        e50   = float(df.iloc[-1]["ema50"])
        if price > e20 and e20 > e50:
            return "bull"
        if price < e20 and e20 < e50:
            return "bear"
        return "neutral"
    except:
        return "neutral"

# =========================================================
# SESSION STRENGTH
# =========================================================
def crypto_session() -> str:
    hour = now_ct().hour
    if 2 <= hour <= 4 or 8 <= hour <= 11:
        return "HIGH"
    if 12 <= hour <= 16 or 20 <= hour <= 23:
        return "MID"
    return "LOW"

def stock_market_open() -> bool:
    # Use UTC and manually account for ET offset to avoid tzdata issues on Railway.
    # ET = UTC-5 (EST) or UTC-4 (EDT). We check both to be safe — NYSE is open
    # 9:30–16:00 ET, so we gate at 9:45–15:45 to avoid bad data at open/close edges.
    now_utc = datetime.utcnow()
    if now_utc.weekday() >= 5:   # Saturday=5, Sunday=6
        return False
    # Determine EDT vs EST: EDT runs second Sunday of March through first Sunday of November
    year = now_utc.year
    # Find second Sunday of March
    march1 = datetime(year, 3, 1)
    first_sun_march = march1 + timedelta(days=(6 - march1.weekday()) % 7)
    edt_start = first_sun_march + timedelta(weeks=1)   # second Sunday
    # Find first Sunday of November
    nov1 = datetime(year, 11, 1)
    edt_end = nov1 + timedelta(days=(6 - nov1.weekday()) % 7)  # first Sunday
    is_edt = edt_start <= now_utc.replace(hour=2, minute=0, second=0) < edt_end
    offset = 4 if is_edt else 5
    now_et = now_utc - timedelta(hours=offset)
    mins = now_et.hour * 60 + now_et.minute
    return (9 * 60 + 45) <= mins <= (15 * 60 + 45)

def grade(score: float) -> str:
    if score >= SCORE_A_PLUS:
        return "A+"
    if score >= SCORE_A:
        return "A"
    return "IGNORE"

# =========================================================
# FILE INIT / LOGGING
# =========================================================
_SIGNAL_COLS = [
    "timestamp", "symbol", "asset_type", "side", "timeframe",
    "entry", "stop", "tp1", "tp2",
    "rsi", "vwap", "atr", "adx",
    "score", "regime", "grade", "session",
    "rr", "setup", "status", "closed_at",
]

def ensure_files():
    for path, header in [
        (SIGNALS_FILE,    _SIGNAL_COLS),
        (DAILY_LOCK_FILE, ["date","loss_count_crypto","loss_count_stock","risk_crypto","risk_stock","cooldown_until"]),
        (PAIR_STATS_FILE, ["symbol","total_closed","wins","losses","win_rate"]),
    ]:
        if not os.path.exists(path):
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(header)

def log_signal(
    timestamp, symbol, asset_type, side,
    entry, stop, tp1, tp2,
    rsi, vwap, atr, adx,
    score, regime, sig_grade, session, rr, setup,
):
    ensure_files()
    with open(SIGNALS_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            timestamp, symbol, asset_type, side, "15m/1h/4h",
            entry, stop, tp1, tp2,
            rsi, vwap, atr, adx,
            score, regime, sig_grade, session, rr,
            setup, "OPEN", "",
        ])

# =========================================================
# DAILY RISK LOCK
# =========================================================
def get_today() -> str:
    return now_ct().strftime("%Y-%m-%d")

def read_lock() -> dict:
    ensure_files()
    today = get_today()
    try:
        df  = pd.read_csv(DAILY_LOCK_FILE)
        row = df[df["date"] == today]
        if not row.empty:
            r = row.iloc[-1]
            return {
                "date": today,
                "loss_count_crypto": int(r["loss_count_crypto"]),
                "loss_count_stock":  int(r["loss_count_stock"]),
                "risk_crypto": float(r["risk_crypto"]),
                "risk_stock":  float(r["risk_stock"]),
                "cooldown_until": str(r["cooldown_until"]) if pd.notna(r["cooldown_until"]) else "",
            }
    except:
        pass
    return {"date": today, "loss_count_crypto": 0, "loss_count_stock": 0,
            "risk_crypto": 0.0, "risk_stock": 0.0, "cooldown_until": ""}

def write_lock(lock: dict):
    ensure_files()
    today = get_today()
    rows  = []
    try:
        rows = pd.read_csv(DAILY_LOCK_FILE)[lambda d: d["date"] != today].values.tolist()
    except:
        pass
    rows.append([today, lock["loss_count_crypto"], lock["loss_count_stock"],
                 lock["risk_crypto"], lock["risk_stock"], lock["cooldown_until"]])
    with open(DAILY_LOCK_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date","loss_count_crypto","loss_count_stock","risk_crypto","risk_stock","cooldown_until"])
        w.writerows(rows)

def is_locked(asset_type: str) -> tuple[bool, str]:
    lock    = read_lock()
    max_risk = ACCOUNT_BALANCE * MAX_DAILY_RISK_PCT
    lc = lock[f"loss_count_{asset_type}"]
    rc = lock[f"risk_{asset_type}"]
    if lc >= MAX_LOSSES_PER_DAY:
        return True, f"{asset_type} max losses ({lc})"
    if rc >= max_risk:
        return True, f"{asset_type} max risk ${rc:.2f}"
    if lock["cooldown_until"]:
        try:
            until = datetime.fromisoformat(lock["cooldown_until"])
            now   = now_ct().replace(tzinfo=None)
            if now < until:
                return True, f"Cooldown until {until.strftime('%H:%M')}"
        except:
            pass
    return False, "OK"

def register_loss(asset_type: str):
    lock = read_lock()
    lock[f"loss_count_{asset_type}"] += 1
    lock[f"risk_{asset_type}"] += ACCOUNT_BALANCE * RISK_PER_TRADE_PCT
    cooldown_dt = (
        now_ct().replace(tzinfo=None)
        + timedelta(minutes=COOLDOWN_AFTER_LOSS_MINUTES)
    )
    lock["cooldown_until"] = cooldown_dt.isoformat()
    write_lock(lock)

# =========================================================
# PAIR STATS
# =========================================================
def rebuild_pair_stats():
    try:
        df     = pd.read_csv(SIGNALS_FILE)
        closed = df[df["status"].isin(["WIN", "LOSS"])].copy()
        if closed.empty:
            return
        rows = []
        for sym, grp in closed.groupby("symbol"):
            total = len(grp)
            wins  = (grp["status"] == "WIN").sum()
            rows.append([sym, total, wins, total - wins, round(wins / total * 100, 2)])
        with open(PAIR_STATS_FILE, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["symbol","total_closed","wins","losses","win_rate"])
            w.writerows(rows)
    except:
        pass

def is_blacklisted(symbol: str) -> bool:
    try:
        df  = pd.read_csv(PAIR_STATS_FILE)
        row = df[df["symbol"] == symbol]
        if row.empty:
            return False
        row = row.iloc[-1]
        if int(row["total_closed"]) < MIN_TRADES_FOR_COIN_FILTER:
            return False
        return float(row["win_rate"]) < MIN_COIN_WINRATE_TO_TRADE
    except:
        return False

def has_open_signal(symbol: str) -> bool:
    try:
        df  = pd.read_csv(SIGNALS_FILE)
        row = df[(df["symbol"] == symbol) & (df["status"] == "OPEN")]
        return not row.empty
    except:
        return False

# =========================================================
# OUTCOME CHECKER
# =========================================================
def update_outcomes():
    ensure_files()
    try:
        df = pd.read_csv(SIGNALS_FILE, dtype={"status": str, "closed_at": str})
    except:
        return
    if df.empty:
        return

    df["closed_at"] = df["closed_at"].astype(object)
    updated = False
    now = now_ct().replace(tzinfo=None)

    for i, row in df.iterrows():
        if row["status"] != "OPEN":
            continue
        symbol     = row["symbol"]
        asset_type = row["asset_type"]
        side       = row["side"]
        stop       = float(row["stop"])
        tp2        = float(row["tp2"])

        try:
            signal_time = pd.to_datetime(row["timestamp"])
            sig_naive   = signal_time.tz_localize(None) if signal_time.tzinfo is None else signal_time.tz_convert(None)
        except:
            continue

        if (now - sig_naive).total_seconds() / 3600 > 48:
            df.at[i, "status"]    = "EXPIRED"
            df.at[i, "closed_at"] = now.strftime("%Y-%m-%d %H:%M:%S")
            updated = True
            continue

        try:
            if asset_type == "crypto":
                candles = get_bybit_klines(symbol, "15", 300)
                if candles is None or len(candles) < 5:
                    continue
                candles = candles[candles["timestamp"] > sig_naive]
                for _, c in candles.iterrows():
                    h, l = float(c["high"]), float(c["low"])
                    if side == "BUY":
                        if l <= stop:
                            df.at[i, "status"]    = "LOSS"
                            df.at[i, "closed_at"] = str(c["timestamp"])
                            register_loss(asset_type)
                            updated = True
                            break
                        if h >= tp2:
                            df.at[i, "status"]    = "WIN"
                            df.at[i, "closed_at"] = str(c["timestamp"])
                            updated = True
                            break
                    else:
                        if h >= stop:
                            df.at[i, "status"]    = "LOSS"
                            df.at[i, "closed_at"] = str(c["timestamp"])
                            register_loss(asset_type)
                            updated = True
                            break
                        if l <= tp2:
                            df.at[i, "status"]    = "WIN"
                            df.at[i, "closed_at"] = str(c["timestamp"])
                            updated = True
                            break
            else:
                raw     = get_yf(symbol, "15m", "5d")
                candles = yf_to_std(raw)
                candles = candles[candles["timestamp"] > sig_naive]
                for _, c in candles.iterrows():
                    h, l = float(c["high"]), float(c["low"])
                    if side == "BUY":
                        if l <= stop:
                            df.at[i, "status"]    = "LOSS"
                            df.at[i, "closed_at"] = str(c["timestamp"])
                            register_loss(asset_type)
                            updated = True
                            break
                        if h >= tp2:
                            df.at[i, "status"]    = "WIN"
                            df.at[i, "closed_at"] = str(c["timestamp"])
                            updated = True
                            break
                    else:
                        if h >= stop:
                            df.at[i, "status"]    = "LOSS"
                            df.at[i, "closed_at"] = str(c["timestamp"])
                            register_loss(asset_type)
                            updated = True
                            break
                        if l <= tp2:
                            df.at[i, "status"]    = "WIN"
                            df.at[i, "closed_at"] = str(c["timestamp"])
                            updated = True
                            break
        except Exception as e:
            print(f"Outcome check error {symbol}: {e}")

    if updated:
        df.to_csv(SIGNALS_FILE, index=False)
        rebuild_pair_stats()

# =========================================================
# DAILY STATS REPORT
# =========================================================
def send_daily_report():
    try:
        df     = pd.read_csv(SIGNALS_FILE)
        closed = df[df["status"].isin(["WIN", "LOSS"])].copy()
        if closed.empty:
            return
        total    = len(closed)
        wins     = (closed["status"] == "WIN").sum()
        win_rate = round(wins / total * 100, 1)

        by_coin  = closed.groupby("symbol")["status"].apply(
            lambda x: f"{round((x=='WIN').mean()*100, 0):.0f}% ({len(x)})"
        ).to_dict()
        by_grade = closed.groupby("grade")["status"].apply(
            lambda x: f"{round((x=='WIN').mean()*100, 0):.0f}% ({len(x)})"
        ).to_dict()
        by_regime = closed.groupby("regime")["status"].apply(
            lambda x: f"{round((x=='WIN').mean()*100, 0):.0f}% ({len(x)})"
        ).to_dict() if "regime" in closed.columns else {}

        coin_lines   = "\n".join(f"  {k}: {v}" for k, v in by_coin.items())
        grade_lines  = "\n".join(f"  {k}: {v}" for k, v in by_grade.items())
        regime_lines = "\n".join(f"  {k}: {v}" for k, v in by_regime.items()) or "  N/A"

        msg = (
            f"📊 V17 DAILY REPORT\n\n"
            f"Total: {total} | Wins: {wins} | WR: {win_rate}%\n\n"
            f"🏆 By Coin:\n{coin_lines}\n\n"
            f"🎯 By Grade:\n{grade_lines}\n\n"
            f"📈 By Regime:\n{regime_lines}"
        )
        send_telegram(msg)
    except Exception as e:
        print(f"Stats report error: {e}")

def build_diagnostic(scan_log: list) -> str:
    if not scan_log:
        return "No scan data this cycle."
    best = sorted(scan_log, key=lambda x: max(x["long_score"], x["short_score"]), reverse=True)[:6]
    lines = ["🔍 V17 DIAGNOSTIC — Top Candidates\n"]
    for e in best:
        bl = ", ".join(e["blockers_long"])  or "✅ clear"
        bs = ", ".join(e["blockers_short"]) or "✅ clear"
        lines.append(
            f"{e['symbol']} [{e.get('regime','?')}]\n"
            f"  Long  {e['long_score']:>5.1f} | {bl}\n"
            f"  Short {e['short_score']:>5.1f} | {bs}"
        )
    return "\n".join(lines)

# =========================================================
# CORE SIGNAL BUILDER
# =========================================================
def build_signal(
    symbol:     str,
    asset_type: str,
    df_15m:     pd.DataFrame,
    df_1h:      pd.DataFrame,
    df_4h:      pd.DataFrame,
    session:    str,
    btc_bias:   str,
    scan_log:   list,
):
    if any(len(d) < 80 for d in [df_15m, df_1h, df_4h]):
        return

    regime  = detect_regime(df_15m)
    penalty = regime_score_penalty(regime)

    close   = df_15m["close"].astype(float)
    high    = df_15m["high"].astype(float)
    low     = df_15m["low"].astype(float)
    volume  = df_15m["volume"].astype(float)

    ema9    = close.ewm(span=9,  adjust=False).mean()
    ema20   = close.ewm(span=20, adjust=False).mean()
    rsi_s   = compute_rsi(close)
    atr_s   = compute_atr(df_15m)
    adx_s, di_p, di_m = compute_adx(df_15m)
    vol_avg = volume.rolling(20).mean()

    price   = float(close.iloc[-2])
    e9      = float(ema9.iloc[-2])
    e20     = float(ema20.iloc[-2])
    rsi_val = float(rsi_s.iloc[-2]) if pd.notna(rsi_s.iloc[-2]) else 50.0
    atr_val = float(atr_s.iloc[-2]) if pd.notna(atr_s.iloc[-2]) else 0.0
    adx_val = float(adx_s.iloc[-2]) if pd.notna(adx_s.iloc[-2]) else 0.0
    dip_val = float(di_p.iloc[-2])  if pd.notna(di_p.iloc[-2])  else 0.0
    dim_val = float(di_m.iloc[-2])  if pd.notna(di_m.iloc[-2])  else 0.0
    vol_val = float(volume.iloc[-2])
    vav     = float(vol_avg.iloc[-2]) if pd.notna(vol_avg.iloc[-2]) else 0.0
    vwap    = anchored_vwap(df_15m, lookback=50)

    if atr_val <= 0:
        scan_log.append({"symbol": symbol, "long_score": 0, "short_score": 0,
                         "blockers_long": ["atr=0"], "blockers_short": ["atr=0"], "regime": regime})
        return

    def htf_bull(df_htf: pd.DataFrame) -> bool:
        c = df_htf["close"].astype(float)
        e = c.ewm(span=20, adjust=False).mean()
        return float(c.iloc[-2]) > float(e.iloc[-2])

    t4h_bull = htf_bull(df_4h)
    t1h_bull = htf_bull(df_1h)
    t15_bull = price > e20

    t4h_bear = not t4h_bull
    t1h_bear = not t1h_bull
    t15_bear = not t15_bull

    full_bull = t4h_bull and t1h_bull and t15_bull
    full_bear = t4h_bear and t1h_bear and t15_bear

    adx_ok   = adx_val >= REGIME_ADX_TRENDING
    adx_bull = adx_ok and dip_val > dim_val
    adx_bear = adx_ok and dim_val > dip_val

    liq_long  = liq_sweep_long(df_15m)
    liq_short = liq_sweep_short(df_15m)
    bos_l     = bos_long(df_15m)
    bos_s     = bos_short(df_15m)
    bull_eng  = bullish_engulf(df_15m)
    bear_eng  = bearish_engulf(df_15m)
    pin_bull  = hammer(df_15m)
    pin_bear  = shooting_star(df_15m)

    choppy     = is_choppy(e9, e20, atr_val)
    vol_ok     = vol_val > (vav * 1.3) if vav > 0 else False
    vwap_long  = price > vwap
    vwap_short = price < vwap

    rsi_long  = 45 <= rsi_val <= 65
    rsi_short = 35 <= rsi_val <= 55

    btc_long_ok  = btc_bias in ("bull", "neutral") or symbol == "BTCUSDT"
    btc_short_ok = btc_bias in ("bear", "neutral") or symbol == "BTCUSDT"

    W = WEIGHTS
    long_score = 0.0
    long_score += W["trend_4h"]   * (1 if t4h_bull   else 0)
    long_score += W["trend_1h"]   * (1 if t1h_bull   else 0)
    long_score += W["trend_15m"]  * (1 if t15_bull   else 0)
    long_score += W["adx_ok"]     * (1 if adx_ok     else 0)
    long_score += W["adx_dir"]    * (1 if adx_bull   else 0)
    long_score += W["liq_sweep"]  * (1 if liq_long   else 0)
    long_score += W["bos"]        * (1 if bos_l      else 0)
    long_score += W["engulf"]     * (1 if bull_eng   else 0)
    long_score += W["pin_bar"]    * (1 if pin_bull   else 0)
    long_score += W["vwap"]       * (1 if vwap_long  else 0)
    long_score += W["rsi"]        * (1 if rsi_long   else 0)
    long_score += W["volume"]     * (1 if vol_ok     else 0)

    short_score = 0.0
    short_score += W["trend_4h"]  * (1 if t4h_bear   else 0)
    short_score += W["trend_1h"]  * (1 if t1h_bear   else 0)
    short_score += W["trend_15m"] * (1 if t15_bear   else 0)
    short_score += W["adx_ok"]    * (1 if adx_ok     else 0)
    short_score += W["adx_dir"]   * (1 if adx_bear   else 0)
    short_score += W["liq_sweep"] * (1 if liq_short  else 0)
    short_score += W["bos"]       * (1 if bos_s      else 0)
    short_score += W["engulf"]    * (1 if bear_eng   else 0)
    short_score += W["pin_bar"]   * (1 if pin_bear   else 0)
    short_score += W["vwap"]      * (1 if vwap_short else 0)
    short_score += W["rsi"]       * (1 if rsi_short  else 0)
    short_score += W["volume"]    * (1 if vol_ok     else 0)

    long_score  -= penalty
    short_score -= penalty

    if session == "HIGH":
        long_score  += 1
        short_score += 1
    elif session == "LOW":
        long_score  -= 1
        short_score -= 1

    long_grade  = grade(long_score)
    short_grade = grade(short_score)

    buy_signal = (
        long_grade in (["A+"] if A_PLUS_ONLY_MODE else ["A+", "A"]) and
        full_bull and
        (liq_long or bos_l) and
        (bull_eng or pin_bull) and
        vwap_long and
        adx_ok and
        not choppy and
        btc_long_ok and
        rsi_long
    )

    sell_signal = (
        short_grade in (["A+"] if A_PLUS_ONLY_MODE else ["A+", "A"]) and
        full_bear and
        (liq_short or bos_s) and
        (bear_eng or pin_bear) and
        vwap_short and
        adx_ok and
        not choppy and
        btc_short_ok and
        rsi_short
    )

    def blockers_for(
        score, sig_grade, full_trend, liq, bos_ok, can, vwap_ok,
        adx_pass, chp, btc_ok, rsi_ok, a_plus,
    ):
        b = []
        if sig_grade == "IGNORE":
            b.append(f"score={score:.1f}<{SCORE_A}")
        if not full_trend:
            b.append("trend not aligned")
        if not (liq or bos_ok):
            b.append("no liq/BOS")
        if not can:
            b.append("no candle conf")
        if not vwap_ok:
            b.append("wrong VWAP side")
        if not adx_pass:
            b.append(f"ADX={adx_val:.0f}<{REGIME_ADX_TRENDING}")
        if chp:
            b.append("choppy")
        if not btc_ok:
            b.append(f"BTC={btc_bias}")
        if not rsi_ok:
            b.append(f"RSI={rsi_val:.0f}")
        return b

    bl = blockers_for(long_score,  long_grade,  full_bull, liq_long,  bos_l,
                      bull_eng or pin_bull, vwap_long,  adx_ok, choppy, btc_long_ok,  rsi_long,  A_PLUS_ONLY_MODE)
    bs = blockers_for(short_score, short_grade, full_bear, liq_short, bos_s,
                      bear_eng or pin_bear, vwap_short, adx_ok, choppy, btc_short_ok, rsi_short, A_PLUS_ONLY_MODE)

    scan_log.append({
        "symbol": symbol,
        "long_score": long_score,
        "short_score": short_score,
        "blockers_long": bl,
        "blockers_short": bs,
        "regime": regime,
    })

    ts = now_ct().strftime("%Y-%m-%d %H:%M:%S")

    if buy_signal and not has_open_signal(symbol):
        entry = price
        stop  = smart_stop_long(df_15m, atr_val)
        risk  = abs(entry - stop)
        if risk <= 0:
            return
        tp1 = round(entry + 1.5 * risk, 5)
        tp2 = round(entry + 2.5 * risk, 5)
        rr  = calc_rr(entry, stop, tp2)
        if rr < MIN_RR:

            return
        qty = get_qty(symbol, entry, stop) if asset_type == "crypto" else "N/A"

        key = f"{symbol}-BUY-{df_15m.iloc[-2]['timestamp']}"
        if key not in sent_alerts:
            sent_alerts.add(key)
            pa_str  = f"liq={liq_long} bos={bos_l} engulf={bull_eng} hammer={pin_bull}"
            setup   = f"V17 long [{regime}]: {pa_str}"
            msg = (
                f"🔥 {long_grade} {'CRYPTO' if asset_type=='crypto' else 'STOCK'} LONG\n"
                f"Symbol: {symbol}\n"
                f"Regime: {regime} | Session: {session}\n"
                f"BTC bias: {btc_bias} | 4H: {'▲' if t4h_bull else '▼'} 1H: {'▲' if t1h_bull else '▼'}\n"
                f"──────────────────────\n"
                f"Entry:  {entry:.5f}\n"
                f"Stop:   {stop}  (swing low)\n"
                f"TP1:    {tp1}  (1.5R)\n"
                f"TP2:    {tp2}  (2.5R)\n"
                f"R:R:    {rr}\n"
                f"Qty:    {qty}\n"
                f"──────────────────────\n"
                f"RSI: {rsi_val:.1f} | VWAP: {vwap:.5f} | ATR: {atr_val:.5f}\n"
                f"ADX: {adx_val:.1f} | Score: {long_score:.1f}\n"
                f"PA: {pa_str}"
                + (f"\n──────────────────────\n🚀 BYBIT:\nTRADE {symbol} BUY {qty} SL:{stop} TP:{tp2}" if asset_type == "crypto" else "")
            )
            send_telegram(msg)
            log_signal(ts, symbol, asset_type, "BUY",
                       entry, stop, tp1, tp2,
                       round(rsi_val, 2), round(vwap, 5), round(atr_val, 5), round(adx_val, 1),
                       round(long_score, 1), regime, long_grade, session, rr, setup)
            print(f"✅ {long_grade} BUY: {symbol} score={long_score:.1f} regime={regime} rr={rr}")

    elif sell_signal and not has_open_signal(symbol):
        entry = price
        stop  = smart_stop_short(df_15m, atr_val)
        risk  = abs(stop - entry)
        if risk <= 0:
            return
        tp1 = round(entry - 1.5 * risk, 5)
        tp2 = round(entry - 2.5 * risk, 5)
        rr  = calc_rr(entry, stop, tp2)
        if rr < MIN_RR:

            return
        qty = get_qty(symbol, entry, stop) if asset_type == "crypto" else "N/A"

        key = f"{symbol}-SELL-{df_15m.iloc[-2]['timestamp']}"
        if key not in sent_alerts:
            sent_alerts.add(key)
            pa_str  = f"liq={liq_short} bos={bos_s} engulf={bear_eng} shootstar={pin_bear}"
            setup   = f"V17 short [{regime}]: {pa_str}"
            msg = (
                f"🔻 {short_grade} {'CRYPTO' if asset_type=='crypto' else 'STOCK'} SHORT\n"
                f"Symbol: {symbol}\n"
                f"Regime: {regime} | Session: {session}\n"
                f"BTC bias: {btc_bias} | 4H: {'▲' if t4h_bull else '▼'} 1H: {'▲' if t1h_bull else '▼'}\n"
                f"──────────────────────\n"
                f"Entry:  {entry:.5f}\n"
                f"Stop:   {stop}  (swing high)\n"
                f"TP1:    {tp1}  (1.5R)\n"
                f"TP2:    {tp2}  (2.5R)\n"
                f"R:R:    {rr}\n"
                f"Qty:    {qty}\n"
                f"──────────────────────\n"
                f"RSI: {rsi_val:.1f} | VWAP: {vwap:.5f} | ATR: {atr_val:.5f}\n"
                f"ADX: {adx_val:.1f} | Score: {short_score:.1f}\n"
                f"PA: {pa_str}"
                + (f"\n──────────────────────\n🚀 BYBIT:\nTRADE {symbol} SELL {qty} SL:{stop} TP:{tp2}" if asset_type == "crypto" else "")
            )
            send_telegram(msg)
            log_signal(ts, symbol, asset_type, "SELL",
                       entry, stop, tp1, tp2,
                       round(rsi_val, 2), round(vwap, 5), round(atr_val, 5), round(adx_val, 1),
                       round(short_score, 1), regime, short_grade, session, rr, setup)
            print(f"✅ {short_grade} SELL: {symbol} score={short_score:.1f} regime={regime} rr={rr}")

    # No signal — intentionally silent (reduce noise)

# =========================================================
# CRYPTO SCANNER
# =========================================================
# Cache for slow timeframes — only refresh every N minutes
_slow_cache: dict = {}          # symbol -> {df_1h, df_4h, ts}
SLOW_CACHE_MINUTES = 10         # 1h and 4h data refreshed every 10 min

def scan_crypto(scan_log: list):
    locked, reason = is_locked("crypto")
    if locked:
        return

    session  = crypto_session()
    btc_bias = get_btc_bias()

    now = now_ct()

    for symbol in CRYPTO_SYMBOLS:
        try:
            if is_blacklisted(symbol):
                continue

            # Always fetch 15m (fast timeframe)
            df_15m = get_bybit_klines(symbol, "15", 300)
            time.sleep(0.5)

            # Fetch 1h and 4h from cache if fresh, else re-fetch
            cached = _slow_cache.get(symbol)
            cache_stale = (
                cached is None or
                (now - cached["ts"]).total_seconds() > SLOW_CACHE_MINUTES * 60
            )

            if cache_stale:
                df_1h = get_bybit_klines(symbol, "60",  200)
                time.sleep(0.5)
                df_4h = get_bybit_klines(symbol, "240", 200)
                time.sleep(0.5)
                if df_1h is not None and df_4h is not None:
                    _slow_cache[symbol] = {"df_1h": df_1h, "df_4h": df_4h, "ts": now}
            else:
                df_1h = cached["df_1h"]
                df_4h = cached["df_4h"]

            if any(d is None or len(d) < 80 for d in [df_15m, df_1h, df_4h]):
                continue

            build_signal(symbol, "crypto", df_15m, df_1h, df_4h, session, btc_bias, scan_log)
        except Exception as e:
            print(f"Error {symbol}: {e}")

# =========================================================
# STOCK SCANNER
# =========================================================
def yf_to_std(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Convert yfinance dataframe to standard format used by bot.
    Handles both Title Case (old yf.download) and lowercase (yf.Ticker.history) columns."""
    # Normalize to lowercase so we handle both column styles
    df = df_raw.copy()
    df.columns = [c.lower() for c in df.columns]
    return pd.DataFrame({
        "timestamp": df.index,
        "open":   df["open"].astype(float).values,
        "high":   df["high"].astype(float).values,
        "low":    df["low"].astype(float).values,
        "close":  df["close"].astype(float).values,
        "volume": df["volume"].astype(float).values,
    }).dropna().reset_index(drop=True)

def scan_stocks(scan_log: list):
    locked, reason = is_locked("stock")
    if locked:
        return

    for ticker in STOCK_TICKERS:
        try:
            raw_15m  = get_yf(ticker, "15m", "5d")
            raw_1h   = get_yf(ticker, "60m", "30d")
            if raw_15m.empty or raw_1h.empty:
                continue

            df_15m = yf_to_std(raw_15m)
            df_1h  = yf_to_std(raw_1h)
            raw_4h = (
                raw_1h.resample("4h").agg({
                    "open": "first", "high": "max",
                    "low": "min",    "close": "last",
                    "volume": "sum",
                }).dropna()
            )
            df_4h = yf_to_std(raw_4h)

            if any(len(d) < 40 for d in [df_15m, df_1h, df_4h]):
                continue

            build_signal(ticker, "stock", df_15m, df_1h, df_4h, "STOCK", "neutral", scan_log)
        except Exception as e:
            print(f"Stock error {ticker}: {e}")

# =========================================================
# MAIN LOOP
# =========================================================
def main():
    ensure_files()
    send_telegram(
        "✅ Scanner V17 started\n\n"
        "Changes from V16:\n"
        "• Primary TF → 15m (less noise)\n"
        "• Hard gates: 3TF trend align + liq/BOS + candle + VWAP + ADX\n"
        "• Confirmed-candle-close only (no live candles)\n"
        "• Anchored VWAP (50-bar, not cumsum)\n"
        "• Min R:R 1.8 enforced\n"
        "• BTC filter hardened (bear=no longs)\n"
        "• Max 1 open signal per symbol\n"
        "• register_loss() bug fixed\n"
        "• Vector/dynamic-weight noise removed"
    )


    last_report_day = None
    cycle_count     = 0

    while True:
        cycle_count += 1
        scan_log = []

        try:
            update_outcomes()

            if stock_market_open():
                scan_stocks(scan_log)
            else:
                pass  # market closed

            scan_crypto(scan_log)

            now = now_ct()
            if now.hour == 20 and now.minute < 2 and last_report_day != now.date():
                send_daily_report()
                last_report_day = now.date()

            if DIAGNOSTIC_EVERY_N_CYCLES > 0 and cycle_count % DIAGNOSTIC_EVERY_N_CYCLES == 0:
                send_telegram(build_diagnostic(scan_log))

        except Exception as e:
            import traceback
            print(f"LOOP ERROR: {e}", flush=True)
            traceback.print_exc()

        time.sleep(CHECK_EVERY_SECONDS)

if __name__ == "__main__":
    main()