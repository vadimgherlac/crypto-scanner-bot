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
print("Environment variables loaded.")

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")

print("BOT_TOKEN =", "loaded" if BOT_TOKEN else None)
print("CHAT_ID =", "loaded" if CHAT_ID else None)
print("BYBIT_API_KEY =", "loaded" if BYBIT_API_KEY else None)

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

SIGNALS_FILE = "trade_signals_v12.csv"
STATS_FILE = "trade_stats_v12.csv"
DAILY_LOCK_FILE = "daily_risk_lock_v12.csv"
PAIR_STATS_FILE = "pair_stats_v12.csv"

sent_alerts = set()

# -------------------------
# ACCOUNT / RISK SETTINGS
# -------------------------
ACCOUNT_BALANCE = 1000.0           # CHANGE THIS TO YOUR REAL ACCOUNT
RISK_PER_TRADE_PCT = 0.01          # 1% risk per trade
MAX_DAILY_RISK_PCT = 0.03          # 3% max daily drawdown
MAX_LOSSES_PER_DAY = 2             # stop after 2 losses
COOLDOWN_AFTER_LOSS_MINUTES = 60   # cooldown after loss
A_PLUS_ONLY_MODE = False           # True = only A+ signals
MIN_COIN_WINRATE_TO_TRADE = 35.0   # blacklist weak performers
MIN_TRADES_FOR_COIN_FILTER = 5      # only blacklist after enough history

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

BYBIT_SYMBOL_MAP = {
    "BTC-USD": "BTCUSDT",
    "ETH-USD": "ETHUSDT",
    "SOL-USD": "SOLUSDT",
    "XRP-USD": "XRPUSDT",
    "DOGE-USD": "DOGEUSDT",
    "BNB-USD": "BNBUSDT",
    "AVAX-USD": "AVAXUSDT",
    "LINK-USD": "LINKUSDT",
    "ADA-USD": "ADAUSDT",
    "LTC-USD": "LTCUSDT",
    "DOT-USD": "DOTUSDT",
    "ATOM-USD": "ATOMUSDT",
    "NEAR-USD": "NEARUSDT",
    "OP-USD": "OPUSDT"
}

# =========================================================
# TELEGRAM
# =========================================================
def send_telegram_message(text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text}
    r = requests.post(url, data=payload)
    if not r.ok:
        print("Telegram error:", r.text)
    r.raise_for_status()

# =========================================================
# HELPERS
# =========================================================
def to_series(data):
    if isinstance(data, pd.DataFrame):
        if data.shape[1] == 1:
            return data.iloc[:, 0]
        raise ValueError("Expected single-column data but got multiple columns.")
    return data

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_vwap_bybit(df):
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    return (typical_price * df["volume"]).cumsum() / df["volume"].cumsum()

def compute_vwap_yf(df: pd.DataFrame):
    high = to_series(df["High"]).astype(float)
    low = to_series(df["Low"]).astype(float)
    close = to_series(df["Close"]).astype(float)
    volume = to_series(df["Volume"]).astype(float)
    typical_price = (high + low + close) / 3
    cumulative_tpv = (typical_price * volume).cumsum()
    cumulative_volume = volume.cumsum().replace(0, pd.NA)
    return cumulative_tpv / cumulative_volume

def compute_atr_df(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def compute_atr_yf(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = to_series(df["High"]).astype(float)
    low = to_series(df["Low"]).astype(float)
    close = to_series(df["Close"]).astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# =========================================================
# PRICE ACTION / SMART MONEY LOGIC
# =========================================================
def liquidity_sweep_long(df: pd.DataFrame, lookback: int = 20) -> bool:
    if len(df) < lookback + 2:
        return False
    recent_low = df["low"].iloc[-(lookback+2):-2].min()
    latest = df.iloc[-1]
    return latest["low"] < recent_low and latest["close"] > recent_low

def liquidity_sweep_short(df: pd.DataFrame, lookback: int = 20) -> bool:
    if len(df) < lookback + 2:
        return False
    recent_high = df["high"].iloc[-(lookback+2):-2].max()
    latest = df.iloc[-1]
    return latest["high"] > recent_high and latest["close"] < recent_high

def break_of_structure_long(df: pd.DataFrame, lookback: int = 10) -> bool:
    if len(df) < lookback + 2:
        return False
    prior_high = df["high"].iloc[-(lookback+2):-2].max()
    return df["close"].iloc[-1] > prior_high

def break_of_structure_short(df: pd.DataFrame, lookback: int = 10) -> bool:
    if len(df) < lookback + 2:
        return False
    prior_low = df["low"].iloc[-(lookback+2):-2].min()
    return df["close"].iloc[-1] < prior_low

def strong_bullish_candle(df: pd.DataFrame) -> bool:
    latest = df.iloc[-1]
    body = abs(latest["close"] - latest["open"])
    rng = latest["high"] - latest["low"]
    return rng > 0 and latest["close"] > latest["open"] and body / rng >= 0.6

def strong_bearish_candle(df: pd.DataFrame) -> bool:
    latest = df.iloc[-1]
    body = abs(latest["close"] - latest["open"])
    rng = latest["high"] - latest["low"]
    return rng > 0 and latest["close"] < latest["open"] and body / rng >= 0.6

def fake_breakout_trap_short(df: pd.DataFrame, lookback: int = 20) -> bool:
    if len(df) < lookback + 2:
        return False
    recent_high = df["high"].iloc[-(lookback+2):-2].max()
    latest = df.iloc[-1]
    return latest["high"] > recent_high and latest["close"] < recent_high

def fake_breakdown_trap_long(df: pd.DataFrame, lookback: int = 20) -> bool:
    if len(df) < lookback + 2:
        return False
    recent_low = df["low"].iloc[-(lookback+2):-2].min()
    latest = df.iloc[-1]
    return latest["low"] < recent_low and latest["close"] > recent_low

def enough_room_long(entry: float, recent_high: float, atr: float) -> bool:
    return (recent_high - entry) > (atr * 1.2)

def enough_room_short(entry: float, recent_low: float, atr: float) -> bool:
    return (entry - recent_low) > (atr * 1.2)

def is_choppy_market(price: float, ema9: float, ema20: float, atr: float) -> bool:
    if atr <= 0:
        return True
    return abs(ema9 - ema20) < (atr * 0.15)

def grade_signal(score: int) -> str:
    if score >= 10:
        return "A+"
    elif score >= 8:
        return "A"
    elif score >= 6:
        return "B"
    return "IGNORE"

# =========================================================
# SESSION FILTERS
# =========================================================
def get_crypto_session_strength():
    now = datetime.now(ZoneInfo("America/Chicago"))
    hour = now.hour

    if 2 <= hour <= 11:
        return "HIGH"
    elif 12 <= hour <= 16 or 20 <= hour <= 23:
        return "MID"
    else:
        return "LOW"

def is_stock_market_open() -> bool:
    now = datetime.now(ZoneInfo("America/New_York"))
    if now.weekday() >= 5:
        return False
    current_minutes = now.hour * 60 + now.minute
    return (9 * 60 + 30) <= current_minutes <= (16 * 60)

# =========================================================
# DATA FETCH
# =========================================================
def get_data(ticker: str, interval: str, period: str) -> pd.DataFrame:
    return yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=False,
    )

def get_bybit_klines(symbol: str, interval: str, limit: int = 200):
    try:
        response = bybit.get_kline(
            category="linear",
            symbol=symbol,
            interval=interval,
            limit=limit
        )
        data = response["result"]["list"]
        if not data:
            return None

        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume", "turnover"
        ])
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit="ms")
        df = df.sort_values("timestamp").reset_index(drop=True)

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)

        return df
    except Exception as e:
        print(f"{symbol}: Bybit data error -> {e}")
        return None

# =========================================================
# POSITION SIZE / RISK
# =========================================================
def get_crypto_qty(bybit_symbol: str, entry: float = None, stop: float = None) -> str:
    """
    Risk-based quantity sizing.
    qty = risk_dollars / stop_distance
    """
    fallback_qty_map = {
        "BTCUSDT": 0.001,
        "ETHUSDT": 0.01,
        "SOLUSDT": 1,
        "XRPUSDT": 25,
        "DOGEUSDT": 200,
        "BNBUSDT": 0.05,
        "AVAXUSDT": 2,
        "LINKUSDT": 2,
        "ADAUSDT": 50,
        "LTCUSDT": 0.5,
        "DOTUSDT": 10,
        "ATOMUSDT": 5,
        "NEARUSDT": 8,
        "OPUSDT": 20
    }

    if entry is None or stop is None:
        return str(fallback_qty_map.get(bybit_symbol, 1))

    risk_dollars = ACCOUNT_BALANCE * RISK_PER_TRADE_PCT
    stop_distance = abs(entry - stop)

    if stop_distance <= 0:
        return str(fallback_qty_map.get(bybit_symbol, 1))

    raw_qty = risk_dollars / stop_distance

    # simple rounding rules
    if bybit_symbol == "BTCUSDT":
        qty = round(raw_qty, 3)
    elif bybit_symbol in ["ETHUSDT", "BNBUSDT", "LTCUSDT"]:
        qty = round(raw_qty, 2)
    elif bybit_symbol in ["SOLUSDT", "AVAXUSDT", "LINKUSDT", "ATOMUSDT", "NEARUSDT", "OPUSDT"]:
        qty = round(raw_qty, 1)
    else:
        qty = round(raw_qty)

    if qty <= 0:
        qty = fallback_qty_map.get(bybit_symbol, 1)

    return str(qty)

def calculate_risk_reward(entry: float, stop: float, target: float) -> float:
    risk = abs(entry - stop)
    reward = abs(target - entry)
    if risk <= 0:
        return 0
    return round(reward / risk, 2)

# =========================================================
# BTC FILTER
# =========================================================
def get_btc_market_bias():
    try:
        df_15m = get_bybit_klines("BTCUSDT", "15", 200)
        if df_15m is None or len(df_15m) < 50:
            return "neutral"

        df_15m["ema20"] = df_15m["close"].ewm(span=20, adjust=False).mean()
        df_15m["vwap"] = compute_vwap_bybit(df_15m)

        latest = df_15m.iloc[-1]
        price = float(latest["close"])
        ema20 = float(latest["ema20"])
        vwap = float(latest["vwap"])

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
        with open(SIGNALS_FILE, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([
                "timestamp", "symbol", "asset_type", "side", "timeframe",
                "entry", "stop", "target", "rsi", "vwap", "atr",
                "score", "grade", "session_strength", "rr",
                "setup", "status", "closed_at"
            ])

def ensure_daily_lock_file():
    if not os.path.exists(DAILY_LOCK_FILE):
        with open(DAILY_LOCK_FILE, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([
                "date", "loss_count", "daily_risk_used", "cooldown_until"
            ])

def ensure_pair_stats_file():
    if not os.path.exists(PAIR_STATS_FILE):
        with open(PAIR_STATS_FILE, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([
                "symbol", "total_closed", "wins", "losses", "win_rate"
            ])

# =========================================================
# LOGGING
# =========================================================
def log_signal(timestamp, symbol, asset_type, side, timeframe, entry, stop, target,
               rsi, vwap, atr, score, grade, session_strength, rr, setup):
    ensure_signal_file()
    with open(SIGNALS_FILE, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            timestamp, symbol, asset_type, side, timeframe,
            entry, stop, target, rsi, vwap, atr,
            score, grade, session_strength, rr,
            setup, "OPEN", ""
        ])

# =========================================================
# DAILY RISK LOCK ENGINE
# =========================================================
def get_today_date_str():
    return datetime.now(ZoneInfo("America/Chicago")).strftime("%Y-%m-%d")

def read_daily_lock():
    ensure_daily_lock_file()
    today = get_today_date_str()

    try:
        df = pd.read_csv(DAILY_LOCK_FILE)
    except:
        return {"date": today, "loss_count": 0, "daily_risk_used": 0.0, "cooldown_until": ""}

    row = df[df["date"] == today]
    if row.empty:
        return {"date": today, "loss_count": 0, "daily_risk_used": 0.0, "cooldown_until": ""}

    row = row.iloc[-1]
    return {
        "date": row["date"],
        "loss_count": int(row["loss_count"]),
        "daily_risk_used": float(row["daily_risk_used"]),
        "cooldown_until": str(row["cooldown_until"]) if pd.notna(row["cooldown_until"]) else ""
    }

def write_daily_lock(loss_count: int, daily_risk_used: float, cooldown_until: str = ""):
    ensure_daily_lock_file()
    today = get_today_date_str()

    rows = []
    if os.path.exists(DAILY_LOCK_FILE):
        try:
            df = pd.read_csv(DAILY_LOCK_FILE)
            rows = df[df["date"] != today].values.tolist()
        except:
            rows = []

    rows.append([today, loss_count, daily_risk_used, cooldown_until])

    with open(DAILY_LOCK_FILE, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["date", "loss_count", "daily_risk_used", "cooldown_until"])
        writer.writerows(rows)

def is_daily_locked():
    lock = read_daily_lock()

    max_daily_risk_dollars = ACCOUNT_BALANCE * MAX_DAILY_RISK_PCT

    if lock["loss_count"] >= MAX_LOSSES_PER_DAY:
        return True, f"Daily lock active: max losses reached ({lock['loss_count']})"

    if lock["daily_risk_used"] >= max_daily_risk_dollars:
        return True, f"Daily lock active: max daily risk reached (${lock['daily_risk_used']:.2f})"

    if lock["cooldown_until"]:
        try:
            cooldown_until = datetime.fromisoformat(lock["cooldown_until"])
            now = datetime.now(ZoneInfo("America/Chicago")).replace(tzinfo=None)
            if now < cooldown_until:
                return True, f"Cooldown active until {cooldown_until}"
        except:
            pass

    return False, "OK"

def register_loss_to_daily_lock():
    lock = read_daily_lock()
    risk_dollars = ACCOUNT_BALANCE * RISK_PER_TRADE_PCT

    new_loss_count = lock["loss_count"] + 1
    new_daily_risk = lock["daily_risk_used"] + risk_dollars
    cooldown_until = (
        datetime.now(ZoneInfo("America/Chicago")).replace(tzinfo=None) +
        timedelta(minutes=COOLDOWN_AFTER_LOSS_MINUTES)
    ).isoformat()

    write_daily_lock(new_loss_count, new_daily_risk, cooldown_until)

def register_win_no_lock_change():
    # You can keep losses only; no reset here.
    pass

# =========================================================
# PAIR LEARNING / BLACKLIST
# =========================================================
def rebuild_pair_stats():
    ensure_signal_file()
    ensure_pair_stats_file()

    try:
        df = pd.read_csv(SIGNALS_FILE)
    except:
        return

    closed = df[df["status"].isin(["WIN", "LOSS"])].copy()
    if closed.empty:
        return

    rows = []
    for symbol, group in closed.groupby("symbol"):
        total = len(group)
        wins = len(group[group["status"] == "WIN"])
        losses = len(group[group["status"] == "LOSS"])
        win_rate = round((wins / total) * 100, 2) if total > 0 else 0.0
        rows.append([symbol, total, wins, losses, win_rate])

    with open(PAIR_STATS_FILE, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["symbol", "total_closed", "wins", "losses", "win_rate"])
        writer.writerows(rows)

def get_pair_stats(symbol: str):
    ensure_pair_stats_file()
    try:
        df = pd.read_csv(PAIR_STATS_FILE)
    except:
        return None

    row = df[df["symbol"] == symbol]
    if row.empty:
        return None

    row = row.iloc[-1]
    return {
        "symbol": row["symbol"],
        "total_closed": int(row["total_closed"]),
        "wins": int(row["wins"]),
        "losses": int(row["losses"]),
        "win_rate": float(row["win_rate"])
    }

def is_coin_blacklisted(symbol: str) -> bool:
    stats = get_pair_stats(symbol)
    if not stats:
        return False

    if stats["total_closed"] < MIN_TRADES_FOR_COIN_FILTER:
        return False

    return stats["win_rate"] < MIN_COIN_WINRATE_TO_TRADE

# =========================================================
# TRADE OUTCOME CHECKER
# =========================================================
def get_future_price_data(symbol: str, asset_type: str):
    try:
        if asset_type == "crypto":
            return get_bybit_klines(symbol, "5", 300)
        else:
            return get_data(symbol, interval="5m", period="5d")
    except Exception as e:
        print(f"Future data fetch error for {symbol}: {e}")
        return None

def update_signal_results():
    ensure_signal_file()

    try:
        df = pd.read_csv(SIGNALS_FILE)
    except Exception as e:
        print(f"Error reading signals file: {e}")
        return

    if df.empty:
        return

    updated = False
    now = datetime.now(ZoneInfo("America/Chicago"))

    for i, row in df.iterrows():
        if row["status"] != "OPEN":
            continue

        symbol = row["symbol"]
        asset_type = row["asset_type"]
        side = row["side"]
        stop = float(row["stop"])
        target = float(row["target"])
        signal_time = pd.to_datetime(row["timestamp"])

        age_hours = (now - signal_time.tz_localize(None)).total_seconds() / 3600
        if age_hours > 24:
            df.at[i, "status"] = "EXPIRED"
            df.at[i, "closed_at"] = now.strftime("%Y-%m-%d %H:%M:%S")
            updated = True
            continue

        future_df = get_future_price_data(symbol, asset_type)
        if future_df is None or len(future_df) < 10:
            continue

        if asset_type == "crypto":
            candles = future_df[future_df["timestamp"] > signal_time]
            if candles.empty:
                continue

            for _, candle in candles.iterrows():
                high = float(candle["high"])
                low = float(candle["low"])

                if side == "BUY":
                    if low <= stop:
                        df.at[i, "status"] = "LOSS"
                        df.at[i, "closed_at"] = str(candle["timestamp"])
                        updated = True
                        register_loss_to_daily_lock()
                        break
                    if high >= target:
                        df.at[i, "status"] = "WIN"
                        df.at[i, "closed_at"] = str(candle["timestamp"])
                        updated = True
                        register_win_no_lock_change()
                        break

                elif side == "SELL":
                    if high >= stop:
                        df.at[i, "status"] = "LOSS"
                        df.at[i, "closed_at"] = str(candle["timestamp"])
                        updated = True
                        register_loss_to_daily_lock()
                        break
                    if low <= target:
                        df.at[i, "status"] = "WIN"
                        df.at[i, "closed_at"] = str(candle["timestamp"])
                        updated = True
                        register_win_no_lock_change()
                        break

        else:
            candles = future_df[future_df.index > signal_time]
            if candles.empty:
                continue

            for idx, candle in candles.iterrows():
                high = float(candle["High"])
                low = float(candle["Low"])

                if side == "BUY":
                    if low <= stop:
                        df.at[i, "status"] = "LOSS"
                        df.at[i, "closed_at"] = str(idx)
                        updated = True
                        register_loss_to_daily_lock()
                        break
                    if high >= target:
                        df.at[i, "status"] = "WIN"
                        df.at[i, "closed_at"] = str(idx)
                        updated = True
                        register_win_no_lock_change()
                        break

                elif side == "SELL":
                    if high >= stop:
                        df.at[i, "status"] = "LOSS"
                        df.at[i, "closed_at"] = str(idx)
                        updated = True
                        register_loss_to_daily_lock()
                        break
                    if low <= target:
                        df.at[i, "status"] = "WIN"
                        df.at[i, "closed_at"] = str(idx)
                        updated = True
                        register_win_no_lock_change()
                        break

    if updated:
        df.to_csv(SIGNALS_FILE, index=False)
        rebuild_pair_stats()
        print("Updated signal results and rebuilt pair stats.")

# =========================================================
# PERFORMANCE STATS
# =========================================================
def build_stats():
    ensure_signal_file()

    try:
        df = pd.read_csv(SIGNALS_FILE)
    except Exception as e:
        print(f"Error reading signal file for stats: {e}")
        return None

    closed = df[df["status"].isin(["WIN", "LOSS"])].copy()
    if closed.empty:
        return None

    total = len(closed)
    wins = len(closed[closed["status"] == "WIN"])
    losses = len(closed[closed["status"] == "LOSS"])
    win_rate = round((wins / total) * 100, 2) if total > 0 else 0

    summary = {
        "total_trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate
    }

    return summary, closed

def send_daily_stats_report():
    stats = build_stats()
    if stats is None:
        return

    summary, closed = stats

    coin_stats = closed.groupby("symbol")["status"].apply(
        lambda x: round((x == "WIN").mean() * 100, 1)
    ).sort_values(ascending=False)

    grade_stats = closed.groupby("grade")["status"].apply(
        lambda x: round((x == "WIN").mean() * 100, 1)
    ).sort_values(ascending=False)

    session_stats = closed.groupby("session_strength")["status"].apply(
        lambda x: round((x == "WIN").mean() * 100, 1)
    ).sort_values(ascending=False)

    top_coins = "\n".join([f"{k}: {v}%" for k, v in coin_stats.head(5).items()])
    top_grades = "\n".join([f"{k}: {v}%" for k, v in grade_stats.items()])
    top_sessions = "\n".join([f"{k}: {v}%" for k, v in session_stats.items()])

    msg = (
        f"📊 V12 DEBT-SURVIVAL REPORT\n\n"
        f"Total Closed Trades: {summary['total_trades']}\n"
        f"Wins: {summary['wins']}\n"
        f"Losses: {summary['losses']}\n"
        f"Win Rate: {summary['win_rate']}%\n\n"
        f"🏆 Top Coins:\n{top_coins if top_coins else 'No data'}\n\n"
        f"🎯 Grade Performance:\n{top_grades if top_grades else 'No data'}\n\n"
        f"⏰ Session Performance:\n{top_sessions if top_sessions else 'No data'}"
    )

    send_telegram_message(msg)

# =========================================================
# CRYPTO SCANNER
# =========================================================
def scan_crypto_intraday():
    locked, lock_reason = is_daily_locked()
    if locked:
        print(f"Crypto scan blocked: {lock_reason}")
        return

    session_strength = get_crypto_session_strength()
    print(f"Crypto session strength: {session_strength}")

    btc_bias = get_btc_market_bias()
    print(f"BTC market bias: {btc_bias}")

    for symbol in CRYPTO_SYMBOLS:
        try:
            if is_coin_blacklisted(symbol):
                print(f"{symbol}: blacklisted by learning engine (low historical win rate)")
                continue

            df_5m = get_bybit_klines(symbol, "5", 200)
            df_15m = get_bybit_klines(symbol, "15", 200)
            df_1h = get_bybit_klines(symbol, "60", 200)

            if (
                df_5m is None or df_15m is None or df_1h is None or
                len(df_5m) < 60 or len(df_15m) < 60 or len(df_1h) < 60
            ):
                print(f"{symbol}: missing timeframe data")
                continue

            df_5m["ema9"] = df_5m["close"].ewm(span=9, adjust=False).mean()
            df_5m["ema20"] = df_5m["close"].ewm(span=20, adjust=False).mean()
            df_5m["rsi"] = compute_rsi(df_5m["close"])
            df_5m["vwap"] = compute_vwap_bybit(df_5m)
            df_5m["atr"] = compute_atr_df(df_5m, 14)
            df_5m["vol_avg"] = df_5m["volume"].rolling(20).mean()

            df_15m["ema20"] = df_15m["close"].ewm(span=20, adjust=False).mean()
            df_1h["ema20"] = df_1h["close"].ewm(span=20, adjust=False).mean()

            latest_5m = df_5m.iloc[-1]
            prev_5m = df_5m.iloc[-2]
            latest_15m = df_15m.iloc[-1]
            latest_1h = df_1h.iloc[-1]

            price = float(latest_5m["close"])
            ema9 = float(latest_5m["ema9"])
            ema20 = float(latest_5m["ema20"])
            rsi = float(latest_5m["rsi"])
            vwap = float(latest_5m["vwap"])
            atr = float(latest_5m["atr"]) if pd.notna(latest_5m["atr"]) else 0
            vol = float(latest_5m["volume"])
            vol_avg = float(latest_5m["vol_avg"]) if pd.notna(latest_5m["vol_avg"]) else 0

            trend_15m_bull = latest_15m["close"] > latest_15m["ema20"]
            trend_15m_bear = latest_15m["close"] < latest_15m["ema20"]
            trend_1h_bull = latest_1h["close"] > latest_1h["ema20"]
            trend_1h_bear = latest_1h["close"] < latest_1h["ema20"]

            fresh_bull = prev_5m["ema9"] <= prev_5m["ema20"] and latest_5m["ema9"] > latest_5m["ema20"]
            fresh_bear = prev_5m["ema9"] >= prev_5m["ema20"] and latest_5m["ema9"] < latest_5m["ema20"]

            volume_ok = vol > (vol_avg * 1.25) if vol_avg > 0 else False
            rsi_buy_ok = 50 <= rsi <= 68
            rsi_sell_ok = 32 <= rsi <= 50
            vwap_buy_ok = price > vwap
            vwap_sell_ok = price < vwap

            liq_long = liquidity_sweep_long(df_5m, 20)
            liq_short = liquidity_sweep_short(df_5m, 20)
            bos_long = break_of_structure_long(df_5m, 10)
            bos_short = break_of_structure_short(df_5m, 10)
            trap_long = fake_breakdown_trap_long(df_5m, 20)
            trap_short = fake_breakout_trap_short(df_5m, 20)
            bull_candle = strong_bullish_candle(df_5m)
            bear_candle = strong_bearish_candle(df_5m)

            recent_high = df_5m["high"].iloc[-22:-2].max()
            recent_low = df_5m["low"].iloc[-22:-2].min()

            room_long = enough_room_long(price, recent_high + atr, atr) if atr > 0 else False
            room_short = enough_room_short(price, recent_low - atr, atr) if atr > 0 else False
            choppy = is_choppy_market(price, ema9, ema20, atr)

            btc_buy_ok = True if symbol == "BTCUSDT" else btc_bias in ["bull", "neutral"]
            btc_sell_ok = True if symbol == "BTCUSDT" else btc_bias in ["bear", "neutral"]

            # -------------------------
            # SCORE ENGINE
            # -------------------------
            long_score = 0
            if trend_1h_bull: long_score += 2
            if trend_15m_bull: long_score += 2
            if liq_long: long_score += 2
            if trap_long: long_score += 1
            if bos_long: long_score += 1
            if fresh_bull: long_score += 1
            if vwap_buy_ok: long_score += 1
            if rsi_buy_ok: long_score += 1
            if volume_ok: long_score += 1
            if bull_candle: long_score += 1
            if room_long: long_score += 1
            if btc_buy_ok: long_score += 1
            if choppy: long_score -= 3

            short_score = 0
            if trend_1h_bear: short_score += 2
            if trend_15m_bear: short_score += 2
            if liq_short: short_score += 2
            if trap_short: short_score += 1
            if bos_short: short_score += 1
            if fresh_bear: short_score += 1
            if vwap_sell_ok: short_score += 1
            if rsi_sell_ok: short_score += 1
            if volume_ok: short_score += 1
            if bear_candle: short_score += 1
            if room_short: short_score += 1
            if btc_sell_ok: short_score += 1
            if choppy: short_score -= 3

            # Session boost/penalty
            if session_strength == "HIGH":
                long_score += 1
                short_score += 1
            elif session_strength == "LOW":
                long_score -= 1
                short_score -= 1

            long_grade = grade_signal(long_score)
            short_grade = grade_signal(short_score)

            # harder outside best hours
            min_score_required = 8 if session_strength == "HIGH" else 9 if session_strength == "MID" else 10

            # dead hours need real confirmation
            extra_low_session_long_ok = True
            if session_strength == "LOW":
                extra_low_session_long_ok = bos_long or trap_long or bull_candle

            extra_low_session_short_ok = True
            if session_strength == "LOW":
                extra_low_session_short_ok = bos_short or trap_short or bear_candle

            # A+ only mode
            grade_buy_ok = long_grade == "A+" if A_PLUS_ONLY_MODE else long_grade in ["A+", "A"]
            grade_sell_ok = short_grade == "A+" if A_PLUS_ONLY_MODE else short_grade in ["A+", "A"]

            buy_signal = (
                long_score >= min_score_required and
                grade_buy_ok and
                atr > 0 and
                not choppy and
                trend_1h_bull and
                trend_15m_bull and
                liq_long and
                fresh_bull and
                vwap_buy_ok and
                room_long and
                btc_buy_ok and
                extra_low_session_long_ok
            )

            sell_signal = (
                short_score >= min_score_required and
                grade_sell_ok and
                atr > 0 and
                not choppy and
                trend_1h_bear and
                trend_15m_bear and
                liq_short and
                fresh_bear and
                vwap_sell_ok and
                room_short and
                btc_sell_ok and
                extra_low_session_short_ok
            )

            if buy_signal:
                entry = price
                stop = round(entry - (1.2 * atr), 4)
                target = round(entry + (2.2 * atr), 4)
                rr = calculate_risk_reward(entry, stop, target)
                qty = get_crypto_qty(symbol, entry, stop)
                timestamp = datetime.now(ZoneInfo("America/Chicago")).strftime("%Y-%m-%d %H:%M:%S")
                alert_key = f"{symbol}-BUY-{str(df_5m.iloc[-1]['timestamp'])}"

                if alert_key not in sent_alerts:
                    sent_alerts.add(alert_key)

                    msg = (
                        f"🔥 {long_grade} ELITE CRYPTO BUY\n"
                        f"Ticker: {symbol}\n"
                        f"Session: {session_strength}\n"
                        f"Entry: {entry:.4f}\n"
                        f"Stop: {stop}\n"
                        f"Target: {target}\n"
                        f"R:R: {rr}\n"
                        f"Qty: {qty}\n"
                        f"RSI: {rsi:.2f}\n"
                        f"VWAP: {vwap:.4f}\n"
                        f"ATR: {atr:.4f}\n"
                        f"BTC Filter: {btc_bias}\n"
                        f"Score: {long_score}\n\n"
                        f"🚀 BYBIT APPROVAL COMMAND:\n"
                        f"TRADE {symbol} BUY {qty} {stop} {target}"
                    )

                    send_telegram_message(msg)
                    log_signal(
                        timestamp, symbol, "crypto", "BUY", "5m/15m/1h",
                        entry, stop, target, round(rsi, 2), round(vwap, 4),
                        round(atr, 4), long_score, long_grade, session_strength, rr,
                        "Elite long: liquidity + structure + BTC filter + ATR + session filter"
                    )
                    print(f"Sent {long_grade} BUY alert for {symbol}")

            elif sell_signal:
                entry = price
                stop = round(entry + (1.2 * atr), 4)
                target = round(entry - (2.2 * atr), 4)
                rr = calculate_risk_reward(entry, stop, target)
                qty = get_crypto_qty(symbol, entry, stop)
                timestamp = datetime.now(ZoneInfo("America/Chicago")).strftime("%Y-%m-%d %H:%M:%S")
                alert_key = f"{symbol}-SELL-{str(df_5m.iloc[-1]['timestamp'])}"

                if alert_key not in sent_alerts:
                    sent_alerts.add(alert_key)

                    msg = (
                        f"🔻 {short_grade} ELITE CRYPTO SELL\n"
                        f"Ticker: {symbol}\n"
                        f"Session: {session_strength}\n"
                        f"Entry: {entry:.4f}\n"
                        f"Stop: {stop}\n"
                        f"Target: {target}\n"
                        f"R:R: {rr}\n"
                        f"Qty: {qty}\n"
                        f"RSI: {rsi:.2f}\n"
                        f"VWAP: {vwap:.4f}\n"
                        f"ATR: {atr:.4f}\n"
                        f"BTC Filter: {btc_bias}\n"
                        f"Score: {short_score}\n\n"
                        f"🚀 BYBIT APPROVAL COMMAND:\n"
                        f"TRADE {symbol} SELL {qty} {stop} {target}"
                    )

                    send_telegram_message(msg)
                    log_signal(
                        timestamp, symbol, "crypto", "SELL", "5m/15m/1h",
                        entry, stop, target, round(rsi, 2), round(vwap, 4),
                        round(atr, 4), short_score, short_grade, session_strength, rr,
                        "Elite short: liquidity + structure + BTC filter + ATR + session filter"
                    )
                    print(f"Sent {short_grade} SELL alert for {symbol}")

            else:
                print(
                    f"{symbol}: no elite setup | "
                    f"price={price:.4f}, ema9={ema9:.4f}, ema20={ema20:.4f}, "
                    f"rsi={rsi:.2f}, vwap={vwap:.4f}, atr={atr:.4f}, "
                    f"long_score={long_score}, short_score={short_score}, choppy={choppy}"
                )

        except Exception as e:
            print(f"Error on {symbol}: {e}")

# =========================================================
# STOCK SCANNER
# =========================================================
def analyze_intraday_symbol(ticker: str, asset_type: str):
    locked, lock_reason = is_daily_locked()
    if locked:
        print(f"Stock scan blocked: {lock_reason}")
        return

    df_5m = get_data(ticker, interval="5m", period="5d")
    df_15m = get_data(ticker, interval="15m", period="10d")
    df_1h = get_data(ticker, interval="60m", period="30d")

    if df_5m.empty or df_15m.empty or df_1h.empty:
        print(f"{ticker}: missing timeframe data")
        return

    close_5m = to_series(df_5m["Close"]).dropna()
    volume_5m = to_series(df_5m["Volume"]).dropna()
    close_15m = to_series(df_15m["Close"]).dropna()
    close_1h = to_series(df_1h["Close"]).dropna()

    if len(close_5m) < 60 or len(volume_5m) < 60 or len(close_15m) < 60 or len(close_1h) < 60:
        print(f"{ticker}: not enough cleaned data")
        return

    ema9_5m = close_5m.ewm(span=9, adjust=False).mean()
    ema20_5m = close_5m.ewm(span=20, adjust=False).mean()
    rsi_5m = compute_rsi(close_5m, 14)
    vol_avg_5m = volume_5m.rolling(20).mean()
    vwap_5m = compute_vwap_yf(df_5m)
    atr_5m = compute_atr_yf(df_5m, 14)

    ema20_15m = close_15m.ewm(span=20, adjust=False).mean()
    ema20_1h = close_1h.ewm(span=20, adjust=False).mean()

    last_price = float(close_5m.iloc[-1])
    last_ema9_5m = float(ema9_5m.iloc[-1])
    last_ema20_5m = float(ema20_5m.iloc[-1])
    prev_ema9_5m = float(ema9_5m.iloc[-2])
    prev_ema20_5m = float(ema20_5m.iloc[-2])
    last_rsi_5m = float(rsi_5m.iloc[-1])
    last_vol_5m = float(volume_5m.iloc[-1])
    last_vol_avg_5m = float(vol_avg_5m.iloc[-1]) if pd.notna(vol_avg_5m.iloc[-1]) else 0
    last_vwap_5m = float(vwap_5m.iloc[-1]) if pd.notna(vwap_5m.iloc[-1]) else last_price
    last_atr_5m = float(atr_5m.iloc[-1]) if pd.notna(atr_5m.iloc[-1]) else 0

    trend_15m_bull = float(close_15m.iloc[-1]) > float(ema20_15m.iloc[-1])
    trend_15m_bear = float(close_15m.iloc[-1]) < float(ema20_15m.iloc[-1])
    trend_1h_bull = float(close_1h.iloc[-1]) > float(ema20_1h.iloc[-1])
    trend_1h_bear = float(close_1h.iloc[-1]) < float(ema20_1h.iloc[-1])

    fresh_bullish_5m = prev_ema9_5m <= prev_ema20_5m and last_ema9_5m > last_ema20_5m
    fresh_bearish_5m = prev_ema9_5m >= prev_ema20_5m and last_ema9_5m < last_ema20_5m

    df_liq = pd.DataFrame({
        "open": to_series(df_5m["Open"]).astype(float),
        "high": to_series(df_5m["High"]).astype(float),
        "low": to_series(df_5m["Low"]).astype(float),
        "close": to_series(df_5m["Close"]).astype(float),
        "volume": to_series(df_5m["Volume"]).astype(float),
    })

    liq_long = liquidity_sweep_long(df_liq, 20)
    liq_short = liquidity_sweep_short(df_liq, 20)
    bos_long = break_of_structure_long(df_liq, 10)
    bos_short = break_of_structure_short(df_liq, 10)
    trap_long = fake_breakdown_trap_long(df_liq, 20)
    trap_short = fake_breakout_trap_short(df_liq, 20)
    bull_candle = strong_bullish_candle(df_liq)
    bear_candle = strong_bearish_candle(df_liq)

    recent_high = df_liq["high"].iloc[-22:-2].max()
    recent_low = df_liq["low"].iloc[-22:-2].min()

    volume_ok = last_vol_5m > (last_vol_avg_5m * 1.25) if last_vol_avg_5m > 0 else False
    rsi_buy_ok = 52 <= last_rsi_5m <= 68
    rsi_sell_ok = 32 <= last_rsi_5m <= 50
    vwap_buy_ok = last_price > last_vwap_5m
    vwap_sell_ok = last_price < last_vwap_5m
    room_long = enough_room_long(last_price, recent_high + last_atr_5m, last_atr_5m) if last_atr_5m > 0 else False
    room_short = enough_room_short(last_price, recent_low - last_atr_5m, last_atr_5m) if last_atr_5m > 0 else False
    choppy = is_choppy_market(last_price, last_ema9_5m, last_ema20_5m, last_atr_5m)

    long_score = 0
    if trend_1h_bull: long_score += 2
    if trend_15m_bull: long_score += 2
    if liq_long: long_score += 2
    if trap_long: long_score += 1
    if bos_long: long_score += 1
    if fresh_bullish_5m: long_score += 1
    if vwap_buy_ok: long_score += 1
    if rsi_buy_ok: long_score += 1
    if volume_ok: long_score += 1
    if bull_candle: long_score += 1
    if room_long: long_score += 1
    if choppy: long_score -= 3

    short_score = 0
    if trend_1h_bear: short_score += 2
    if trend_15m_bear: short_score += 2
    if liq_short: short_score += 2
    if trap_short: short_score += 1
    if bos_short: short_score += 1
    if fresh_bearish_5m: short_score += 1
    if vwap_sell_ok: short_score += 1
    if rsi_sell_ok: short_score += 1
    if volume_ok: short_score += 1
    if bear_candle: short_score += 1
    if room_short: short_score += 1
    if choppy: short_score -= 3

    long_grade = grade_signal(long_score)
    short_grade = grade_signal(short_score)
    timestamp = datetime.now(ZoneInfo("America/Chicago")).strftime("%Y-%m-%d %H:%M:%S")

    grade_buy_ok = long_grade == "A+" if A_PLUS_ONLY_MODE else long_grade in ["A+", "A"]
    grade_sell_ok = short_grade == "A+" if A_PLUS_ONLY_MODE else short_grade in ["A+", "A"]

    bullish = (
        grade_buy_ok and last_atr_5m > 0 and not choppy and
        trend_1h_bull and trend_15m_bull and liq_long and fresh_bullish_5m and
        vwap_buy_ok and room_long
    )

    bearish = (
        grade_sell_ok and last_atr_5m > 0 and not choppy and
        trend_1h_bear and trend_15m_bear and liq_short and fresh_bearish_5m and
        vwap_sell_ok and room_short
    )

    if bullish:
        alert_key = f"{ticker}-BUY-{str(close_5m.index[-1])}"
        if alert_key not in sent_alerts:
            sent_alerts.add(alert_key)

            stop = round(last_price - (1.2 * last_atr_5m), 4)
            target = round(last_price + (2.2 * last_atr_5m), 4)
            rr = calculate_risk_reward(last_price, stop, target)
            setup = "Elite stock long: 1H+15m trend + 5m liquidity reclaim + BOS + VWAP + ATR"

            msg = (
                f"🔥 {long_grade} ELITE STOCK BUY\n"
                f"Ticker: {ticker}\n"
                f"Entry: {last_price:.4f}\n"
                f"Stop: {stop:.4f}\n"
                f"Target: {target:.4f}\n"
                f"R:R: {rr}\n"
                f"RSI(5m): {last_rsi_5m:.2f}\n"
                f"VWAP: {last_vwap_5m:.4f}\n"
                f"ATR: {last_atr_5m:.4f}\n"
                f"Score: {long_score}"
            )

            send_telegram_message(msg)
            log_signal(
                timestamp, ticker, "stock", "BUY", "5m/15m/1h",
                round(last_price, 4), stop, target, round(last_rsi_5m, 2),
                round(last_vwap_5m, 4), round(last_atr_5m, 4),
                long_score, long_grade, "STOCK_SESSION", rr, setup
            )
            print(f"Sent elite BUY alert for {ticker}")

    elif bearish:
        alert_key = f"{ticker}-SELL-{str(close_5m.index[-1])}"
        if alert_key not in sent_alerts:
            sent_alerts.add(alert_key)

            stop = round(last_price + (1.2 * last_atr_5m), 4)
            target = round(last_price - (2.2 * last_atr_5m), 4)
            rr = calculate_risk_reward(last_price, stop, target)
            setup = "Elite stock short: 1H+15m trend + 5m liquidity rejection + BOS + VWAP + ATR"

            msg = (
                f"🔻 {short_grade} ELITE STOCK SELL\n"
                f"Ticker: {ticker}\n"
                f"Entry: {last_price:.4f}\n"
                f"Stop: {stop:.4f}\n"
                f"Target: {target:.4f}\n"
                f"R:R: {rr}\n"
                f"RSI(5m): {last_rsi_5m:.2f}\n"
                f"VWAP: {last_vwap_5m:.4f}\n"
                f"ATR: {last_atr_5m:.4f}\n"
                f"Score: {short_score}"
            )

            send_telegram_message(msg)
            log_signal(
                timestamp, ticker, "stock", "SELL", "5m/15m/1h",
                round(last_price, 4), stop, target, round(last_rsi_5m, 2),
                round(last_vwap_5m, 4), round(last_atr_5m, 4),
                short_score, short_grade, "STOCK_SESSION", rr, setup
            )
            print(f"Sent elite SELL alert for {ticker}")

    else:
        print(
            f"{ticker}: no elite setup | "
            f"price={last_price:.4f}, ema9={last_ema9_5m:.4f}, ema20={last_ema20_5m:.4f}, "
            f"rsi={last_rsi_5m:.2f}, vwap={last_vwap_5m:.4f}, atr={last_atr_5m:.4f}, "
            f"long_score={long_score}, short_score={short_score}, choppy={choppy}"
        )

# =========================================================
# MAIN
# =========================================================
def main():
    if not BOT_TOKEN or not CHAT_ID:
        raise ValueError("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID first.")

    ensure_signal_file()
    ensure_daily_lock_file()
    ensure_pair_stats_file()

    send_telegram_message("✅ Scanner bot V12 DEBT-SURVIVAL started")
    print("Bot started successfully. Entering main loop...")

    last_report_day = None

    while True:
        print("Running new scan cycle...")

        try:
            # 1) update old trade outcomes
            update_signal_results()

            # 2) stocks (only market hours)
            if is_stock_market_open():
                for ticker in STOCK_TICKERS:
                    try:
                        analyze_intraday_symbol(ticker, "stock")
                    except Exception as e:
                        print(f"Error on stock {ticker}: {e}")
            else:
                print("Stock market is closed. Skipping stock scan.")

            # 3) crypto (24/7)
            try:
                scan_crypto_intraday()
            except Exception as e:
                print(f"Error in crypto scan: {e}")

            # 4) daily report once per day
            now = datetime.now(ZoneInfo("America/Chicago"))
            if now.hour == 20 and now.minute < 5:
                if last_report_day != now.date():
                    send_daily_stats_report()
                    last_report_day = now.date()

        except Exception as e:
            print(f"Main loop error: {e}")

        time.sleep(CHECK_EVERY_SECONDS)

if __name__ == "__main__":
    main()