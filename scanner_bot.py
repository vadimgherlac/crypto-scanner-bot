import os
import time
import csv
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
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
    api_secret=BYBIT_API_SECRET
)

# =========================================================
# SETTINGS
# =========================================================
CHECK_EVERY_SECONDS = 60

SIGNALS_FILE    = "trade_signals_v15.csv"
DAILY_LOCK_FILE = "daily_risk_lock_v15.csv"
PAIR_STATS_FILE = "pair_stats_v15.csv"

sent_alerts = set()

# -------------------------
# ACCOUNT / RISK
# -------------------------
ACCOUNT_BALANCE             = 1000.0
RISK_PER_TRADE_PCT          = 0.01
MAX_DAILY_RISK_PCT          = 0.03
MAX_LOSSES_PER_DAY          = 2
COOLDOWN_AFTER_LOSS_MINUTES = 60
A_PLUS_ONLY_MODE            = False
MIN_COIN_WINRATE_TO_TRADE   = 35.0
MIN_TRADES_FOR_COIN_FILTER  = 5

# -------------------------
# DIAGNOSTIC
# -------------------------
DIAGNOSTIC_EVERY_N_CYCLES = 30

# -------------------------
# WATCHLIST
# -------------------------
STOCK_TICKERS = [
    "AAPL", "TSLA", "NVDA", "SPY", "QQQ",
    "AMD", "META", "AMZN", "MSFT", "PLTR"
]

CRYPTO_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT",
    "DOGEUSDT", "BNBUSDT", "AVAXUSDT", "LINKUSDT",
    "ADAUSDT", "LTCUSDT", "DOTUSDT", "ATOMUSDT",
    "NEARUSDT", "OPUSDT"
]

# =========================================================
# TELEGRAM
# =========================================================
def send_telegram_message(text):
    url     = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text}
    r = requests.post(url, data=payload)
    if not r.ok:
        print("Telegram error:", r.text)

# =========================================================
# HELPERS
# =========================================================
def to_series(data):
    if isinstance(data, pd.DataFrame):
        if data.shape[1] == 1:
            return data.iloc[:, 0]
        raise ValueError("Expected single-column DataFrame.")
    return data

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs       = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_vwap_bybit(df: pd.DataFrame) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    return (tp * df["volume"]).cumsum() / df["volume"].cumsum()

def compute_vwap_yf(df: pd.DataFrame) -> pd.Series:
    high   = to_series(df["High"]).astype(float)
    low    = to_series(df["Low"]).astype(float)
    close  = to_series(df["Close"]).astype(float)
    volume = to_series(df["Volume"]).astype(float)
    tp     = (high + low + close) / 3
    return (tp * volume).cumsum() / volume.cumsum().replace(0, pd.NA)

def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def compute_atr_df(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return compute_atr(
        df["high"].astype(float),
        df["low"].astype(float),
        df["close"].astype(float),
        period
    )

def compute_atr_yf(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return compute_atr(
        to_series(df["High"]).astype(float),
        to_series(df["Low"]).astype(float),
        to_series(df["Close"]).astype(float),
        period
    )

# =========================================================
# ADX
# =========================================================
def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    high  = high.astype(float).reset_index(drop=True)
    low   = low.astype(float).reset_index(drop=True)
    close = close.astype(float).reset_index(drop=True)

    plus_dm  = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm   < 0] = 0
    minus_dm[minus_dm < 0] = 0
    mask = plus_dm > minus_dm
    minus_dm[mask]  = 0
    plus_dm[~mask]  = 0

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)

    atr      = tr.rolling(period).mean()
    plus_di  = 100 * (plus_dm.rolling(period).mean()  / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
    dx       = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
    adx      = dx.rolling(period).mean()
    return adx, plus_di, minus_di

# =========================================================
# RSI DIVERGENCE
# =========================================================
def detect_rsi_divergence(close: pd.Series, rsi: pd.Series, lookback: int = 20):
    if len(close) < lookback + 2:
        return False, False
    c      = close.iloc[-(lookback+2):-1].reset_index(drop=True)
    r      = rsi.iloc[-(lookback+2):-1].reset_index(drop=True)
    c_last = float(close.iloc[-1])
    r_last = float(rsi.iloc[-1])

    price_low         = float(c.min())
    rsi_at_price_low  = float(r[c.idxmin()])
    bull_div          = c_last < price_low and r_last > rsi_at_price_low

    price_high        = float(c.max())
    rsi_at_price_high = float(r[c.idxmax()])
    bear_div          = c_last > price_high and r_last < rsi_at_price_high

    return bull_div, bear_div

# =========================================================
# CANDLESTICK PATTERNS
# =========================================================
def is_bullish_engulfing(df: pd.DataFrame) -> bool:
    if len(df) < 2:
        return False
    prev, curr = df.iloc[-2], df.iloc[-1]
    return (
        prev["close"] < prev["open"] and
        curr["close"] > curr["open"] and
        curr["open"]  < prev["close"] and
        curr["close"] > prev["open"]
    )

def is_bearish_engulfing(df: pd.DataFrame) -> bool:
    if len(df) < 2:
        return False
    prev, curr = df.iloc[-2], df.iloc[-1]
    return (
        prev["close"] > prev["open"] and
        curr["close"] < curr["open"] and
        curr["open"]  > prev["close"] and
        curr["close"] < prev["open"]
    )

def is_hammer(df: pd.DataFrame) -> bool:
    latest = df.iloc[-1]
    body   = abs(latest["close"] - latest["open"])
    rng    = latest["high"] - latest["low"]
    if rng <= 0:
        return False
    lower_wick = min(latest["open"], latest["close"]) - latest["low"]
    upper_wick = latest["high"] - max(latest["open"], latest["close"])
    return lower_wick >= 2 * body and upper_wick <= 0.3 * rng

def is_shooting_star(df: pd.DataFrame) -> bool:
    latest = df.iloc[-1]
    body   = abs(latest["close"] - latest["open"])
    rng    = latest["high"] - latest["low"]
    if rng <= 0:
        return False
    upper_wick = latest["high"] - max(latest["open"], latest["close"])
    lower_wick = min(latest["open"], latest["close"]) - latest["low"]
    return upper_wick >= 2 * body and lower_wick <= 0.3 * rng

def is_inside_bar_breakout_long(df: pd.DataFrame) -> bool:
    if len(df) < 3:
        return False
    mother, inside, current = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    return (
        inside["high"] < mother["high"] and inside["low"] > mother["low"] and
        current["close"] > mother["high"]
    )

def is_inside_bar_breakout_short(df: pd.DataFrame) -> bool:
    if len(df) < 3:
        return False
    mother, inside, current = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    return (
        inside["high"] < mother["high"] and inside["low"] > mother["low"] and
        current["close"] < mother["low"]
    )

# =========================================================
# VOLUME PROFILE (approximate POC)
# =========================================================
def compute_poc(df: pd.DataFrame, bins: int = 20) -> float:
    if len(df) < 20:
        return float(df["close"].iloc[-1])
    price_min = df["low"].min()
    price_max = df["high"].max()
    if price_max <= price_min:
        return float(df["close"].iloc[-1])
    bin_size = (price_max - price_min) / bins
    vol_bins = [0.0] * bins
    for _, row in df.iterrows():
        mid = (row["high"] + row["low"]) / 2
        idx = min(int((mid - price_min) / bin_size), bins - 1)
        vol_bins[idx] += row["volume"]
    poc_idx = vol_bins.index(max(vol_bins))
    return price_min + (poc_idx + 0.5) * bin_size

def near_poc(price: float, poc: float, atr: float) -> bool:
    return abs(price - poc) <= (atr * 0.5)

# =========================================================
# SMART STOP PLACEMENT
# =========================================================
def smart_stop_long(df: pd.DataFrame, atr: float, lookback: int = 10) -> float:
    recent_low = df["low"].iloc[-(lookback+2):-1].min()
    return round(recent_low - (0.25 * atr), 4)

def smart_stop_short(df: pd.DataFrame, atr: float, lookback: int = 10) -> float:
    recent_high = df["high"].iloc[-(lookback+2):-1].max()
    return round(recent_high + (0.25 * atr), 4)

# =========================================================
# EXISTING PRICE ACTION
# =========================================================
def liquidity_sweep_long(df: pd.DataFrame, lookback: int = 20) -> bool:
    if len(df) < lookback + 2:
        return False
    recent_low = df["low"].iloc[-(lookback+2):-2].min()
    latest     = df.iloc[-1]
    return latest["low"] < recent_low and latest["close"] > recent_low

def liquidity_sweep_short(df: pd.DataFrame, lookback: int = 20) -> bool:
    if len(df) < lookback + 2:
        return False
    recent_high = df["high"].iloc[-(lookback+2):-2].max()
    latest      = df.iloc[-1]
    return latest["high"] > recent_high and latest["close"] < recent_high

def break_of_structure_long(df: pd.DataFrame, lookback: int = 10) -> bool:
    if len(df) < lookback + 2:
        return False
    return df["close"].iloc[-1] > df["high"].iloc[-(lookback+2):-2].max()

def break_of_structure_short(df: pd.DataFrame, lookback: int = 10) -> bool:
    if len(df) < lookback + 2:
        return False
    return df["close"].iloc[-1] < df["low"].iloc[-(lookback+2):-2].min()

def strong_bullish_candle(df: pd.DataFrame) -> bool:
    latest = df.iloc[-1]
    body   = abs(latest["close"] - latest["open"])
    rng    = latest["high"] - latest["low"]
    return rng > 0 and latest["close"] > latest["open"] and body / rng >= 0.6

def strong_bearish_candle(df: pd.DataFrame) -> bool:
    latest = df.iloc[-1]
    body   = abs(latest["close"] - latest["open"])
    rng    = latest["high"] - latest["low"]
    return rng > 0 and latest["close"] < latest["open"] and body / rng >= 0.6

def fake_breakdown_trap_long(df: pd.DataFrame, lookback: int = 20) -> bool:
    if len(df) < lookback + 2:
        return False
    recent_low = df["low"].iloc[-(lookback+2):-2].min()
    latest     = df.iloc[-1]
    return latest["low"] < recent_low and latest["close"] > recent_low

def fake_breakout_trap_short(df: pd.DataFrame, lookback: int = 20) -> bool:
    if len(df) < lookback + 2:
        return False
    recent_high = df["high"].iloc[-(lookback+2):-2].max()
    latest      = df.iloc[-1]
    return latest["high"] > recent_high and latest["close"] < recent_high

def enough_room_long(entry: float, target: float, atr: float) -> bool:
    return (target - entry) > (atr * 1.2)

def enough_room_short(entry: float, target: float, atr: float) -> bool:
    return (entry - target) > (atr * 1.2)

def is_choppy_market(ema9: float, ema20: float, atr: float) -> bool:
    if atr <= 0:
        return True
    return abs(ema9 - ema20) < (atr * 0.12)

def grade_signal(score: int) -> str:
    if score >= 14:
        return "A+"
    elif score >= 10:
        return "A"
    elif score >= 7:
        return "B"
    return "IGNORE"

# =========================================================
# SESSION FILTERS
# =========================================================
def get_crypto_session_strength() -> str:
    hour = datetime.now(ZoneInfo("America/Chicago")).hour
    if 2 <= hour <= 11:
        return "HIGH"
    elif 12 <= hour <= 16 or 20 <= hour <= 23:
        return "MID"
    return "LOW"

def is_stock_market_open() -> bool:
    now  = datetime.now(ZoneInfo("America/New_York"))
    if now.weekday() >= 5:
        return False
    mins = now.hour * 60 + now.minute
    return (9 * 60 + 30) <= mins <= (16 * 60)

# =========================================================
# DATA FETCH
# =========================================================
def get_data(ticker: str, interval: str, period: str) -> pd.DataFrame:
    return yf.download(
        ticker, period=period, interval=interval,
        auto_adjust=True, progress=False,
        group_by="column", threads=False,
    )

def get_bybit_klines(symbol: str, interval: str, limit: int = 200):
    try:
        response = bybit.get_kline(
            category="linear", symbol=symbol,
            interval=interval, limit=limit
        )
        data = response["result"]["list"]
        if not data:
            return None
        df = pd.DataFrame(data, columns=[
            "timestamp","open","high","low","close","volume","turnover"
        ])[["timestamp","open","high","low","close","volume"]]
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit="ms")
        df = df.sort_values("timestamp").reset_index(drop=True)
        for col in ["open","high","low","close","volume"]:
            df[col] = df[col].astype(float)
        return df
    except Exception as e:
        print(f"{symbol}: Bybit data error -> {e}")
        return None

# =========================================================
# POSITION SIZE
# =========================================================
def get_crypto_qty(symbol: str, entry: float, stop: float) -> str:
    fallback = {
        "BTCUSDT":0.001,"ETHUSDT":0.01,"SOLUSDT":1,"XRPUSDT":25,
        "DOGEUSDT":200,"BNBUSDT":0.05,"AVAXUSDT":2,"LINKUSDT":2,
        "ADAUSDT":50,"LTCUSDT":0.5,"DOTUSDT":10,"ATOMUSDT":5,
        "NEARUSDT":8,"OPUSDT":20
    }
    stop_dist = abs(entry - stop)
    if stop_dist <= 0:
        return str(fallback.get(symbol, 1))
    raw = (ACCOUNT_BALANCE * RISK_PER_TRADE_PCT) / stop_dist
    if symbol == "BTCUSDT":
        qty = round(raw, 3)
    elif symbol in ["ETHUSDT","BNBUSDT","LTCUSDT"]:
        qty = round(raw, 2)
    elif symbol in ["SOLUSDT","AVAXUSDT","LINKUSDT","ATOMUSDT","NEARUSDT","OPUSDT"]:
        qty = round(raw, 1)
    else:
        qty = round(raw)
    return str(qty if qty > 0 else fallback.get(symbol, 1))

def calculate_rr(entry: float, stop: float, target: float) -> float:
    risk   = abs(entry - stop)
    reward = abs(target - entry)
    return round(reward / risk, 2) if risk > 0 else 0

# =========================================================
# BTC FILTER
# =========================================================
def get_btc_market_bias() -> str:
    try:
        df = get_bybit_klines("BTCUSDT", "15", 200)
        if df is None or len(df) < 50:
            return "neutral"
        df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
        df["vwap"]  = compute_vwap_bybit(df)
        latest      = df.iloc[-1]
        price, ema20, vwap = float(latest["close"]), float(latest["ema20"]), float(latest["vwap"])
        if price > ema20 and price > vwap:
            return "bull"
        elif price < ema20 and price < vwap:
            return "bear"
        return "neutral"
    except Exception as e:
        print(f"BTC filter error: {e}")
        return "neutral"

# =========================================================
# FILE INIT
# =========================================================
def ensure_signal_file():
    if not os.path.exists(SIGNALS_FILE):
        with open(SIGNALS_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "timestamp","symbol","asset_type","side","timeframe",
                "entry","stop","tp1","tp2","rsi","vwap","atr","adx",
                "score","grade","session_strength","rr","setup","status","closed_at"
            ])

def ensure_daily_lock_file():
    if not os.path.exists(DAILY_LOCK_FILE):
        with open(DAILY_LOCK_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["date","loss_count","daily_risk_used","cooldown_until"])

def ensure_pair_stats_file():
    if not os.path.exists(PAIR_STATS_FILE):
        with open(PAIR_STATS_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["symbol","total_closed","wins","losses","win_rate"])

def log_signal(timestamp, symbol, asset_type, side, timeframe,
               entry, stop, tp1, tp2, rsi, vwap, atr, adx,
               score, grade, session_strength, rr, setup):
    ensure_signal_file()
    with open(SIGNALS_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            timestamp, symbol, asset_type, side, timeframe,
            entry, stop, tp1, tp2, rsi, vwap, atr, adx,
            score, grade, session_strength, rr, setup, "OPEN", ""
        ])

# =========================================================
# DAILY RISK LOCK
# =========================================================
def get_today() -> str:
    return datetime.now(ZoneInfo("America/Chicago")).strftime("%Y-%m-%d")

def read_daily_lock() -> dict:
    ensure_daily_lock_file()
    today = get_today()
    try:
        df  = pd.read_csv(DAILY_LOCK_FILE)
        row = df[df["date"] == today]
        if not row.empty:
            row = row.iloc[-1]
            return {
                "date": today,
                "loss_count": int(row["loss_count"]),
                "daily_risk_used": float(row["daily_risk_used"]),
                "cooldown_until": str(row["cooldown_until"]) if pd.notna(row["cooldown_until"]) else ""
            }
    except:
        pass
    return {"date": today, "loss_count": 0, "daily_risk_used": 0.0, "cooldown_until": ""}

def write_daily_lock(loss_count, daily_risk_used, cooldown_until=""):
    ensure_daily_lock_file()
    today = get_today()
    rows  = []
    try:
        df   = pd.read_csv(DAILY_LOCK_FILE)
        rows = df[df["date"] != today].values.tolist()
    except:
        pass
    rows.append([today, loss_count, daily_risk_used, cooldown_until])
    with open(DAILY_LOCK_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date","loss_count","daily_risk_used","cooldown_until"])
        w.writerows(rows)

def is_daily_locked():
    lock     = read_daily_lock()
    max_risk = ACCOUNT_BALANCE * MAX_DAILY_RISK_PCT
    if lock["loss_count"] >= MAX_LOSSES_PER_DAY:
        return True, f"Max losses ({lock['loss_count']})"
    if lock["daily_risk_used"] >= max_risk:
        return True, f"Max risk ${lock['daily_risk_used']:.2f}"
    if lock["cooldown_until"]:
        try:
            until = datetime.fromisoformat(lock["cooldown_until"])
            now   = datetime.now(ZoneInfo("America/Chicago")).replace(tzinfo=None)
            if now < until:
                return True, f"Cooldown until {until}"
        except:
            pass
    return False, "OK"

def register_loss():
    lock     = read_daily_lock()
    cooldown = (
        datetime.now(ZoneInfo("America/Chicago")).replace(tzinfo=None) +
        timedelta(minutes=COOLDOWN_AFTER_LOSS_MINUTES)
    ).isoformat()
    write_daily_lock(
        lock["loss_count"] + 1,
        lock["daily_risk_used"] + ACCOUNT_BALANCE * RISK_PER_TRADE_PCT,
        cooldown
    )

# =========================================================
# PAIR STATS
# =========================================================
def rebuild_pair_stats():
    try:
        df     = pd.read_csv(SIGNALS_FILE)
        closed = df[df["status"].isin(["WIN","LOSS"])].copy()
        if closed.empty:
            return
        rows = []
        for sym, grp in closed.groupby("symbol"):
            total = len(grp)
            wins  = (grp["status"] == "WIN").sum()
            rows.append([sym, total, wins, total-wins, round(wins/total*100, 2)])
        with open(PAIR_STATS_FILE, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["symbol","total_closed","wins","losses","win_rate"])
            w.writerows(rows)
    except:
        pass

def is_coin_blacklisted(symbol: str) -> bool:
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

# =========================================================
# TRADE OUTCOME CHECKER
# =========================================================
def update_signal_results():
    ensure_signal_file()
    try:
        df = pd.read_csv(SIGNALS_FILE)
    except:
        return
    if df.empty:
        return

    updated = False
    now     = datetime.now(ZoneInfo("America/Chicago"))

    for i, row in df.iterrows():
        if row["status"] != "OPEN":
            continue
        symbol      = row["symbol"]
        asset_type  = row["asset_type"]
        side        = row["side"]
        stop        = float(row["stop"])
        tp2         = float(row["tp2"])
        signal_time = pd.to_datetime(row["timestamp"])

        if (now - signal_time.tz_localize(None)).total_seconds() / 3600 > 24:
            df.at[i, "status"]    = "EXPIRED"
            df.at[i, "closed_at"] = now.strftime("%Y-%m-%d %H:%M:%S")
            updated = True
            continue

        try:
            if asset_type == "crypto":
                future  = get_bybit_klines(symbol, "5", 300)
                if future is None or len(future) < 10:
                    continue
                candles = future[future["timestamp"] > signal_time]
                for _, c in candles.iterrows():
                    h, l = float(c["high"]), float(c["low"])
                    if side == "BUY":
                        if l <= stop:
                            df.at[i,"status"]="LOSS"; df.at[i,"closed_at"]=str(c["timestamp"]); register_loss(); updated=True; break
                        if h >= tp2:
                            df.at[i,"status"]="WIN";  df.at[i,"closed_at"]=str(c["timestamp"]); updated=True; break
                    else:
                        if h >= stop:
                            df.at[i,"status"]="LOSS"; df.at[i,"closed_at"]=str(c["timestamp"]); register_loss(); updated=True; break
                        if l <= tp2:
                            df.at[i,"status"]="WIN";  df.at[i,"closed_at"]=str(c["timestamp"]); updated=True; break
            else:
                future  = get_data(symbol, "5m", "5d")
                if future.empty:
                    continue
                candles = future[future.index > signal_time]
                for idx, c in candles.iterrows():
                    h = float(c["High"] if not isinstance(c["High"], pd.Series) else c["High"].iloc[0])
                    l = float(c["Low"]  if not isinstance(c["Low"],  pd.Series) else c["Low"].iloc[0])
                    if side == "BUY":
                        if l <= stop:
                            df.at[i,"status"]="LOSS"; df.at[i,"closed_at"]=str(idx); register_loss(); updated=True; break
                        if h >= tp2:
                            df.at[i,"status"]="WIN";  df.at[i,"closed_at"]=str(idx); updated=True; break
                    else:
                        if h >= stop:
                            df.at[i,"status"]="LOSS"; df.at[i,"closed_at"]=str(idx); register_loss(); updated=True; break
                        if l <= tp2:
                            df.at[i,"status"]="WIN";  df.at[i,"closed_at"]=str(idx); updated=True; break
        except Exception as e:
            print(f"Outcome check error {symbol}: {e}")

    if updated:
        df.to_csv(SIGNALS_FILE, index=False)
        rebuild_pair_stats()

# =========================================================
# DAILY STATS REPORT
# =========================================================
def send_daily_stats_report():
    try:
        df     = pd.read_csv(SIGNALS_FILE)
        closed = df[df["status"].isin(["WIN","LOSS"])].copy()
        if closed.empty:
            return
        total    = len(closed)
        wins     = (closed["status"] == "WIN").sum()
        win_rate = round(wins / total * 100, 2)
        coin_wr  = closed.groupby("symbol")["status"].apply(
            lambda x: round((x=="WIN").mean()*100,1)).sort_values(ascending=False)
        grade_wr = closed.groupby("grade")["status"].apply(
            lambda x: round((x=="WIN").mean()*100,1)).sort_values(ascending=False)
        msg = (
            f"📊 V15 DAILY REPORT\n\n"
            f"Total: {total} | Wins: {wins} | Losses: {total-wins}\n"
            f"Win Rate: {win_rate}%\n\n"
            f"🏆 By Coin:\n" + "\n".join([f"{k}: {v}%" for k,v in coin_wr.head(5).items()]) + "\n\n"
            f"🎯 By Grade:\n" + "\n".join([f"{k}: {v}%" for k,v in grade_wr.items()])
        )
        send_telegram_message(msg)
    except Exception as e:
        print(f"Stats report error: {e}")

# =========================================================
# DIAGNOSTIC
# =========================================================
def build_diagnostic_report(scan_log: list) -> str:
    if not scan_log:
        return "No scan data this cycle."
    best  = sorted(scan_log, key=lambda x: max(x["long_score"], x["short_score"]), reverse=True)[:5]
    lines = ["🔍 V15 DIAGNOSTIC — Top Candidates\n"]
    for e in best:
        bl = ", ".join(e["blockers_long"])  or "✅ clear"
        bs = ", ".join(e["blockers_short"]) or "✅ clear"
        lines.append(
            f"{e['symbol']}\n"
            f"  Long  {e['long_score']:>3} | {bl}\n"
            f"  Short {e['short_score']:>3} | {bs}"
        )
    return "\n".join(lines)

# =========================================================
# CORE SIGNAL BUILDER  (shared by crypto + stocks)
# =========================================================
def build_signal(
    symbol:           str,
    asset_type:       str,
    df_5m:            pd.DataFrame,
    df_15m:           pd.DataFrame,
    df_1h:            pd.DataFrame,
    df_4h:            pd.DataFrame,
    session_strength: str,
    btc_bias:         str,
    scan_log:         list,
):
    # ---------- indicators ----------
    df_5m["ema9"]    = df_5m["close"].ewm(span=9,  adjust=False).mean()
    df_5m["ema20"]   = df_5m["close"].ewm(span=20, adjust=False).mean()
    df_5m["rsi"]     = compute_rsi(df_5m["close"])
    df_5m["vwap"]    = compute_vwap_bybit(df_5m)
    df_5m["atr"]     = compute_atr_df(df_5m)
    df_5m["vol_avg"] = df_5m["volume"].rolling(20).mean()

    df_15m["ema20"] = df_15m["close"].ewm(span=20, adjust=False).mean()
    df_1h["ema20"]  = df_1h["close"].ewm(span=20,  adjust=False).mean()
    df_4h["ema20"]  = df_4h["close"].ewm(span=20,  adjust=False).mean()

    latest  = df_5m.iloc[-1]
    prev    = df_5m.iloc[-2]

    price   = float(latest["close"])
    ema9    = float(latest["ema9"])
    ema20   = float(latest["ema20"])
    rsi     = float(latest["rsi"])
    vwap    = float(latest["vwap"])
    atr     = float(latest["atr"])     if pd.notna(latest["atr"])     else 0
    vol     = float(latest["volume"])
    vol_avg = float(latest["vol_avg"]) if pd.notna(latest["vol_avg"]) else 0

    if atr <= 0:
        scan_log.append({"symbol":symbol,"long_score":0,"short_score":0,
                         "blockers_long":["atr=0"],"blockers_short":["atr=0"]})
        return

    # ---------- trend ----------
    trend_15m_bull = float(df_15m.iloc[-1]["close"]) > float(df_15m.iloc[-1]["ema20"])
    trend_15m_bear = not trend_15m_bull
    trend_1h_bull  = float(df_1h.iloc[-1]["close"])  > float(df_1h.iloc[-1]["ema20"])
    trend_1h_bear  = not trend_1h_bull
    trend_4h_bull  = float(df_4h.iloc[-1]["close"])  > float(df_4h.iloc[-1]["ema20"])
    trend_4h_bear  = not trend_4h_bull

    # ---------- ADX (score only, not a hard gate) ----------
    adx_series, plus_di, minus_di = compute_adx(df_5m["high"], df_5m["low"], df_5m["close"])
    adx_val  = float(adx_series.iloc[-1]) if pd.notna(adx_series.iloc[-1]) else 0
    adx_ok   = adx_val >= 18                           # lowered from 20
    adx_bull = adx_ok and float(plus_di.iloc[-1])  > float(minus_di.iloc[-1])
    adx_bear = adx_ok and float(minus_di.iloc[-1]) > float(plus_di.iloc[-1])

    # ---------- RSI divergence ----------
    bull_div, bear_div = detect_rsi_divergence(df_5m["close"], df_5m["rsi"])

    # ---------- Candlestick patterns ----------
    bull_engulf = is_bullish_engulfing(df_5m)
    bear_engulf = is_bearish_engulfing(df_5m)
    hammer      = is_hammer(df_5m)
    shoot_star  = is_shooting_star(df_5m)
    ib_long     = is_inside_bar_breakout_long(df_5m)
    ib_short    = is_inside_bar_breakout_short(df_5m)

    # ---------- Volume profile ----------
    poc        = compute_poc(df_5m)
    near_value = near_poc(price, poc, atr)

    # ---------- Existing PA ----------
    liq_long    = liquidity_sweep_long(df_5m,  20)
    liq_short   = liquidity_sweep_short(df_5m, 20)
    bos_long    = break_of_structure_long(df_5m,  10)
    bos_short   = break_of_structure_short(df_5m, 10)
    trap_long   = fake_breakdown_trap_long(df_5m,  20)
    trap_short  = fake_breakout_trap_short(df_5m, 20)
    bull_candle = strong_bullish_candle(df_5m)
    bear_candle = strong_bearish_candle(df_5m)

    # At least one PA confirmation required (still a hard gate)
    pa_long_ok  = liq_long  or trap_long  or bos_long
    pa_short_ok = liq_short or trap_short or bos_short

    fresh_bull = float(prev["ema9"]) <= float(prev["ema20"]) and ema9 > ema20
    fresh_bear = float(prev["ema9"]) >= float(prev["ema20"]) and ema9 < ema20

    volume_ok    = vol > (vol_avg * 1.2) if vol_avg > 0 else False
    rsi_buy_ok   = 48 <= rsi <= 70
    rsi_sell_ok  = 30 <= rsi <= 52
    vwap_buy_ok  = price > vwap
    vwap_sell_ok = price < vwap

    target_long  = price + (2.5 * atr)
    target_short = price - (2.5 * atr)
    room_long    = enough_room_long(price,  target_long,  atr)
    room_short   = enough_room_short(price, target_short, atr)
    choppy       = is_choppy_market(ema9, ema20, atr)

    btc_buy_ok  = True if symbol == "BTCUSDT" else btc_bias in ["bull", "neutral"]
    btc_sell_ok = True if symbol == "BTCUSDT" else btc_bias in ["bear", "neutral"]

    # =========================================================
    # SCORE ENGINE
    # ---------------------------------------------------------
    # 4H trend:  score bonus (+2), NOT a hard gate
    # ADX:       score bonus (+1/+1), NOT a hard gate
    # Hard gates: 1H + 15m trend, PA confirm, VWAP, room, not choppy
    # =========================================================
    long_score = 0
    # Trend (up to 6 — 4H is bonus only)
    if trend_4h_bull:   long_score += 2   # bonus, not required
    if trend_1h_bull:   long_score += 2
    if trend_15m_bull:  long_score += 2
    # ADX (bonus only)
    if adx_ok:          long_score += 1
    if adx_bull:        long_score += 1
    # Price action
    if liq_long:        long_score += 2
    if trap_long:       long_score += 1
    if bos_long:        long_score += 1
    if fresh_bull:      long_score += 1
    # Candles
    if bull_engulf:     long_score += 2
    if hammer:          long_score += 1
    if ib_long:         long_score += 1
    if bull_candle:     long_score += 1
    # Confluence
    if vwap_buy_ok:     long_score += 1
    if rsi_buy_ok:      long_score += 1
    if volume_ok:       long_score += 1
    if bull_div:        long_score += 2
    if near_value:      long_score += 1
    if room_long:       long_score += 1
    if btc_buy_ok:      long_score += 1
    # Penalties
    if choppy:          long_score -= 3
    if not adx_ok:      long_score -= 1   # soft penalty only

    if session_strength == "HIGH":  long_score += 1
    elif session_strength == "LOW": long_score -= 1

    short_score = 0
    if trend_4h_bear:   short_score += 2
    if trend_1h_bear:   short_score += 2
    if trend_15m_bear:  short_score += 2
    if adx_ok:          short_score += 1
    if adx_bear:        short_score += 1
    if liq_short:       short_score += 2
    if trap_short:      short_score += 1
    if bos_short:       short_score += 1
    if fresh_bear:      short_score += 1
    if bear_engulf:     short_score += 2
    if shoot_star:      short_score += 1
    if ib_short:        short_score += 1
    if bear_candle:     short_score += 1
    if vwap_sell_ok:    short_score += 1
    if rsi_sell_ok:     short_score += 1
    if volume_ok:       short_score += 1
    if bear_div:        short_score += 2
    if near_value:      short_score += 1
    if room_short:      short_score += 1
    if btc_sell_ok:     short_score += 1
    if choppy:          short_score -= 3
    if not adx_ok:      short_score -= 1
    if session_strength == "HIGH":  short_score += 1
    elif session_strength == "LOW": short_score -= 1

    long_grade  = grade_signal(long_score)
    short_grade = grade_signal(short_score)

    # Eased thresholds: HIGH=8, MID=9, LOW=10
    min_score = 8 if session_strength == "HIGH" else 9 if session_strength == "MID" else 10

    grade_buy_ok  = long_grade  in (["A+"] if A_PLUS_ONLY_MODE else ["A+","A"])
    grade_sell_ok = short_grade in (["A+"] if A_PLUS_ONLY_MODE else ["A+","A"])

    # ---------- HARD GATES ----------
    # 4H and ADX removed as hard requirements — score points only
    # Core gates: 1H + 15m trend, PA, VWAP, room, not choppy
    buy_signal = (
        long_score >= min_score and grade_buy_ok and
        not choppy and
        trend_1h_bull and trend_15m_bull and
        pa_long_ok and vwap_buy_ok and room_long and btc_buy_ok
    )

    sell_signal = (
        short_score >= min_score and grade_sell_ok and
        not choppy and
        trend_1h_bear and trend_15m_bear and
        pa_short_ok and vwap_sell_ok and room_short and btc_sell_ok
    )

    # ---------- diagnostic ----------
    blockers_long  = []
    blockers_short = []

    if long_score < min_score:  blockers_long.append(f"score {long_score}<{min_score}")
    if not grade_buy_ok:        blockers_long.append(f"grade={long_grade}")
    if choppy:                  blockers_long.append("choppy")
    if not trend_1h_bull:       blockers_long.append("1H bear")
    if not trend_15m_bull:      blockers_long.append("15m bear")
    if not pa_long_ok:          blockers_long.append("no PA")
    if not vwap_buy_ok:         blockers_long.append("<VWAP")
    if not room_long:           blockers_long.append("no room")
    if not btc_buy_ok:          blockers_long.append("BTC bear")

    if short_score < min_score: blockers_short.append(f"score {short_score}<{min_score}")
    if not grade_sell_ok:       blockers_short.append(f"grade={short_grade}")
    if choppy:                  blockers_short.append("choppy")
    if not trend_1h_bear:       blockers_short.append("1H bull")
    if not trend_15m_bear:      blockers_short.append("15m bull")
    if not pa_short_ok:         blockers_short.append("no PA")
    if not vwap_sell_ok:        blockers_short.append(">VWAP")
    if not room_short:          blockers_short.append("no room")
    if not btc_sell_ok:         blockers_short.append("BTC bull")

    scan_log.append({
        "symbol": symbol,
        "long_score": long_score,
        "short_score": short_score,
        "blockers_long": blockers_long,
        "blockers_short": blockers_short,
    })

    timestamp = datetime.now(ZoneInfo("America/Chicago")).strftime("%Y-%m-%d %H:%M:%S")

    if buy_signal:
        entry = price
        stop  = smart_stop_long(df_5m, atr)
        risk  = abs(entry - stop)
        tp1   = round(entry + 1.5 * risk, 4)
        tp2   = round(entry + 2.5 * risk, 4)
        rr    = calculate_rr(entry, stop, tp2)
        qty   = get_crypto_qty(symbol, entry, stop) if asset_type == "crypto" else "N/A"
        key   = f"{symbol}-BUY-{str(df_5m.iloc[-1]['timestamp'])}"

        if key not in sent_alerts:
            sent_alerts.add(key)
            pa_str  = f"liq={liq_long} trap={trap_long} bos={bos_long}"
            can_str = f"engulf={bull_engulf} hammer={hammer} IB={ib_long}"
            msg = (
                f"🔥 {long_grade} {'CRYPTO' if asset_type=='crypto' else 'STOCK'} BUY\n"
                f"Symbol: {symbol}\n"
                f"Session: {session_strength} | BTC: {btc_bias}\n"
                f"4H: {'▲' if trend_4h_bull else '▼'} | ADX: {adx_val:.1f}\n"
                f"Entry:  {entry:.4f}\n"
                f"Stop:   {stop}  (swing low)\n"
                f"TP1:    {tp1}  (1.5R)\n"
                f"TP2:    {tp2}  (2.5R)\n"
                f"R:R:    {rr}\n"
                f"Qty:    {qty}\n"
                f"RSI: {rsi:.1f} | VWAP: {vwap:.4f} | ATR: {atr:.4f}\n"
                f"Score: {long_score} | {pa_str}\n"
                f"Candles: {can_str}\n"
                f"Div: bull={bull_div} | POC near={near_value}"
                + (f"\n\n🚀 BYBIT:\nTRADE {symbol} BUY {qty} {stop} {tp2}" if asset_type=="crypto" else "")
            )
            send_telegram_message(msg)
            log_signal(
                timestamp, symbol, asset_type, "BUY", "5m/15m/1h/4h",
                entry, stop, tp1, tp2, round(rsi,2), round(vwap,4),
                round(atr,4), round(adx_val,1), long_score, long_grade,
                session_strength, rr, f"V15 long: {pa_str} {can_str}"
            )
            print(f"✅ {long_grade} BUY: {symbol} score={long_score}")

    elif sell_signal:
        entry = price
        stop  = smart_stop_short(df_5m, atr)
        risk  = abs(stop - entry)
        tp1   = round(entry - 1.5 * risk, 4)
        tp2   = round(entry - 2.5 * risk, 4)
        rr    = calculate_rr(entry, stop, tp2)
        qty   = get_crypto_qty(symbol, entry, stop) if asset_type == "crypto" else "N/A"
        key   = f"{symbol}-SELL-{str(df_5m.iloc[-1]['timestamp'])}"

        if key not in sent_alerts:
            sent_alerts.add(alert_key := key)
            pa_str  = f"liq={liq_short} trap={trap_short} bos={bos_short}"
            can_str = f"engulf={bear_engulf} shootstar={shoot_star} IB={ib_short}"
            msg = (
                f"🔻 {short_grade} {'CRYPTO' if asset_type=='crypto' else 'STOCK'} SELL\n"
                f"Symbol: {symbol}\n"
                f"Session: {session_strength} | BTC: {btc_bias}\n"
                f"4H: {'▲' if trend_4h_bull else '▼'} | ADX: {adx_val:.1f}\n"
                f"Entry:  {entry:.4f}\n"
                f"Stop:   {stop}  (swing high)\n"
                f"TP1:    {tp1}  (1.5R)\n"
                f"TP2:    {tp2}  (2.5R)\n"
                f"R:R:    {rr}\n"
                f"Qty:    {qty}\n"
                f"RSI: {rsi:.1f} | VWAP: {vwap:.4f} | ATR: {atr:.4f}\n"
                f"Score: {short_score} | {pa_str}\n"
                f"Candles: {can_str}\n"
                f"Div: bear={bear_div} | POC near={near_value}"
                + (f"\n\n🚀 BYBIT:\nTRADE {symbol} SELL {qty} {stop} {tp2}" if asset_type=="crypto" else "")
            )
            send_telegram_message(msg)
            log_signal(
                timestamp, symbol, asset_type, "SELL", "5m/15m/1h/4h",
                entry, stop, tp1, tp2, round(rsi,2), round(vwap,4),
                round(atr,4), round(adx_val,1), short_score, short_grade,
                session_strength, rr, f"V15 short: {pa_str} {can_str}"
            )
            print(f"✅ {short_grade} SELL: {symbol} score={short_score}")

    else:
        print(f"{symbol}: no signal | L={long_score} S={short_score} | "
              f"choppy={choppy} pa_l={pa_long_ok} pa_s={pa_short_ok} "
              f"adx={adx_val:.1f} 4H={'▲' if trend_4h_bull else '▼'}")

# =========================================================
# CRYPTO SCANNER
# =========================================================
def scan_crypto_intraday(scan_log: list):
    locked, reason = is_daily_locked()
    if locked:
        print(f"Crypto locked: {reason}")
        return

    session  = get_crypto_session_strength()
    btc_bias = get_btc_market_bias()
    print(f"Session: {session} | BTC: {btc_bias}")

    for symbol in CRYPTO_SYMBOLS:
        try:
            if is_coin_blacklisted(symbol):
                print(f"{symbol}: blacklisted")
                continue

            df_5m  = get_bybit_klines(symbol, "5",   200)
            df_15m = get_bybit_klines(symbol, "15",  200)
            df_1h  = get_bybit_klines(symbol, "60",  200)
            df_4h  = get_bybit_klines(symbol, "240", 200)

            if any(d is None or len(d) < 60 for d in [df_5m, df_15m, df_1h, df_4h]):
                print(f"{symbol}: insufficient data")
                continue

            build_signal(symbol, "crypto", df_5m, df_15m, df_1h, df_4h,
                         session, btc_bias, scan_log)

        except Exception as e:
            print(f"Error {symbol}: {e}")

# =========================================================
# STOCK SCANNER
# =========================================================
def scan_stock(ticker: str, scan_log: list):
    locked, reason = is_daily_locked()
    if locked:
        print(f"Stock locked: {reason}")
        return

    try:
        df_raw_5m  = get_data(ticker, "5m",  "5d")
        df_raw_15m = get_data(ticker, "15m", "10d")
        df_raw_1h  = get_data(ticker, "60m", "30d")

        if any(d.empty for d in [df_raw_5m, df_raw_15m, df_raw_1h]):
            print(f"{ticker}: missing data")
            return

        def yf_to_std(df_raw) -> pd.DataFrame:
            return pd.DataFrame({
                "timestamp": df_raw.index,
                "open":   to_series(df_raw["Open"]).astype(float).values,
                "high":   to_series(df_raw["High"]).astype(float).values,
                "low":    to_series(df_raw["Low"]).astype(float).values,
                "close":  to_series(df_raw["Close"]).astype(float).values,
                "volume": to_series(df_raw["Volume"]).astype(float).values,
            }).dropna()

        df_5m  = yf_to_std(df_raw_5m)
        df_15m = yf_to_std(df_raw_15m)
        df_1h  = yf_to_std(df_raw_1h)

        # Resample 1h → 4h for stocks
        df_raw_4h = df_raw_1h.resample("4h").agg({
            "Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"
        }).dropna()
        df_4h = yf_to_std(df_raw_4h)

        if any(len(d) < 60 for d in [df_5m, df_15m, df_1h, df_4h]):
            print(f"{ticker}: not enough rows")
            return

        build_signal(ticker, "stock", df_5m, df_15m, df_1h, df_4h,
                     "STOCK_SESSION", "neutral", scan_log)

    except Exception as e:
        print(f"Stock error {ticker}: {e}")

# =========================================================
# MAIN
# =========================================================
def main():
    ensure_signal_file()
    ensure_daily_lock_file()
    ensure_pair_stats_file()

    send_telegram_message(
        "✅ Scanner bot V15 started\n"
        "4H = score bonus | ADX = score bonus | "
        "Hard gates: 1H+15m trend, PA, VWAP, room"
    )
    print("Bot V15 started.")

    last_report_day = None
    cycle_count     = 0

    while True:
        print(f"\n--- Cycle {cycle_count + 1} ---")
        scan_log = []

        try:
            update_signal_results()

            if is_stock_market_open():
                for ticker in STOCK_TICKERS:
                    scan_stock(ticker, scan_log)
            else:
                print("Stock market closed.")

            scan_crypto_intraday(scan_log)

            now = datetime.now(ZoneInfo("America/Chicago"))
            if now.hour == 20 and now.minute < 5 and last_report_day != now.date():
                send_daily_stats_report()
                last_report_day = now.date()

            cycle_count += 1
            if DIAGNOSTIC_EVERY_N_CYCLES > 0 and cycle_count % DIAGNOSTIC_EVERY_N_CYCLES == 0:
                send_telegram_message(build_diagnostic_report(scan_log))

        except Exception as e:
            print(f"Main loop error: {e}")

        time.sleep(CHECK_EVERY_SECONDS)

if __name__ == "__main__":
    main()