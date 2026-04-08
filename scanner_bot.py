import os
import time
import csv
import os
import time
import csv
import uuid
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, timezone
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
CHECK_EVERY_SECONDS = 120

TRADE_ALERTS_FILE = "trade_alerts.csv"
SIGNALS_FILE = "signals_log.csv"
BYBIT_TRADES_FILE = "bybit_closed_trades.csv"
PERFORMANCE_FILE = "performance_summary.csv"

sent_alerts = set()

# ===== WATCHLISTS =====
STOCK_TICKERS = [
    "AAPL", "TSLA", "NVDA", "SPY", "QQQ",
    "AMD", "META", "AMZN", "MSFT", "PLTR"
]

# Debt survival mode = tighter crypto list
CRYPTO_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT"
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
    payload = {
        "chat_id": CHAT_ID,
        "text": text
    }
    r = requests.post(url, data=payload, timeout=20)
    if not r.ok:
        print("Telegram error:", r.text)
    r.raise_for_status()

# =========================================================
# FILE HELPERS
# =========================================================
def ensure_csv(file_path, headers):
    if not os.path.exists(file_path):
        with open(file_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

def init_files():
    ensure_csv(TRADE_ALERTS_FILE, [
        "timestamp", "ticker", "asset_type", "side", "timeframe",
        "entry", "stop_or_risk", "target", "rsi", "setup", "grade"
    ])

    ensure_csv(SIGNALS_FILE, [
        "signal_id", "timestamp", "symbol", "side", "grade", "score",
        "entry", "stop", "target", "rsi", "atr", "vwap",
        "volume_spike", "breakout", "choppy",
        "trend_15m", "trend_1h", "status", "resolved_at", "notes"
    ])

    ensure_csv(BYBIT_TRADES_FILE, [
        "trade_id", "symbol", "side", "qty",
        "avg_entry", "avg_exit", "realized_pnl",
        "opened_at", "closed_at", "raw_json"
    ])

    ensure_csv(PERFORMANCE_FILE, [
        "generated_at", "metric", "group", "value"
    ])

# =========================================================
# INDICATORS
# =========================================================
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_atr(df, period=14):
    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()
    return atr

def compute_vwap_bybit(df):
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    return (typical_price * df["volume"]).cumsum() / df["volume"].cumsum()

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def to_series(data):
    if isinstance(data, pd.DataFrame):
        if data.shape[1] == 1:
            return data.iloc[:, 0]
        raise ValueError("Expected single-column data but got multiple columns.")
    return data

def compute_vwap(df: pd.DataFrame) -> pd.Series:
    high = to_series(df["High"]).astype(float)
    low = to_series(df["Low"]).astype(float)
    close = to_series(df["Close"]).astype(float)
    volume = to_series(df["Volume"]).astype(float)

    typical_price = (high + low + close) / 3
    cumulative_tpv = (typical_price * volume).cumsum()
    cumulative_volume = volume.cumsum().replace(0, pd.NA)
    vwap = cumulative_tpv / cumulative_volume
    return vwap

# =========================================================
# MARKET FILTERS
# =========================================================
def is_stock_market_open() -> bool:
    now = datetime.now(ZoneInfo("America/New_York"))
    if now.weekday() >= 5:
        return False
    current_minutes = now.hour * 60 + now.minute
    market_open_minutes = 9 * 60 + 30
    market_close_minutes = 16 * 60
    return market_open_minutes <= current_minutes <= market_close_minutes

def is_choppy_market(df):
    if len(df) < 30:
        return True

    recent = df.tail(20)
    price_range = recent["high"].max() - recent["low"].min()
    avg_close = recent["close"].mean()

    if avg_close == 0:
        return True

    range_pct = price_range / avg_close
    return range_pct < 0.004  # 0.4%

def breakout_long(df, lookback=10):
    if len(df) < lookback + 2:
        return False
    latest_close = df["close"].iloc[-1]
    recent_high = df["high"].iloc[-(lookback+1):-1].max()
    return latest_close > recent_high

def breakout_short(df, lookback=10):
    if len(df) < lookback + 2:
        return False
    latest_close = df["close"].iloc[-1]
    recent_low = df["low"].iloc[-(lookback+1):-1].min()
    return latest_close < recent_low

def grade_signal(score):
    # A+ only if you want max strictness:
    if score >= 10:
        return "A+"
    elif score >= 9:
        return "B"
    return None

# =========================================================
# DATA FETCH
# =========================================================
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
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit="ms", utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)

        return df

    except Exception as e:
        print(f"{symbol}: Bybit data error -> {e}")
        return None

def get_data(ticker: str, interval: str, period: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=False,
    )
    return df

# =========================================================
# POSITION SIZE
# =========================================================
def get_crypto_qty(bybit_symbol: str) -> str:
    qty_map = {
        "BTCUSDT": "0.001",
        "ETHUSDT": "0.01",
        "SOLUSDT": "1",
        "XRPUSDT": "25",
        "DOGEUSDT": "200",
        "BNBUSDT": "0.05",
        "AVAXUSDT": "2",
        "LINKUSDT": "2",
        "ADAUSDT": "50",
        "LTCUSDT": "0.5",
        "DOTUSDT": "10",
        "ATOMUSDT": "5",
        "NEARUSDT": "8",
        "OPUSDT": "20"
    }
    return qty_map.get(bybit_symbol, "1")

# =========================================================
# LOGGING
# =========================================================
def log_alert(timestamp, ticker, asset_type, side, timeframe, entry, stop_or_risk, target, rsi, setup, grade=""):
    with open(TRADE_ALERTS_FILE, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            timestamp, ticker, asset_type, side, timeframe,
            entry, stop_or_risk, target, rsi, setup, grade
        ])

def log_signal(
    symbol, side, grade, score, entry, stop, target, rsi, atr, vwap,
    volume_spike, breakout, choppy, trend_15m, trend_1h
):
    signal_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now(ZoneInfo("America/Chicago")).strftime("%Y-%m-%d %H:%M:%S")

    with open(SIGNALS_FILE, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            signal_id, timestamp, symbol, side, grade, score,
            round(entry, 4), round(stop, 4), round(target, 4),
            round(rsi, 2), round(atr, 4), round(vwap, 4),
            str(volume_spike), str(breakout), str(choppy),
            trend_15m, trend_1h, "OPEN", "", ""
        ])

    return signal_id

# =========================================================
# SIGNAL PERFORMANCE TRACKING
# =========================================================
def update_signal_results():
    """
    Check unresolved signals and mark them WIN / LOSS / EXPIRED
    based on whether target or stop was hit first after signal time.
    """
    try:
        if not os.path.exists(SIGNALS_FILE):
            return

        df = pd.read_csv(SIGNALS_FILE)
        if df.empty:
            return

        updated = False

        for idx, row in df.iterrows():
            if str(row["status"]).upper() != "OPEN":
                continue

            symbol = row["symbol"]
            side = row["side"]
            entry = float(row["entry"])
            stop = float(row["stop"])
            target = float(row["target"])

            # Parse signal time
            try:
                signal_time = pd.to_datetime(row["timestamp"]).tz_localize("America/Chicago").tz_convert("UTC")
            except Exception:
                signal_time = pd.to_datetime(row["timestamp"], utc=True)

            # Pull recent 5m candles after signal
            df_5m = get_bybit_klines(symbol, "5", 300)
            if df_5m is None or df_5m.empty:
                continue

            future_df = df_5m[df_5m["timestamp"] >= signal_time].copy()
            if future_df.empty:
                continue

            # Give signals a max life window to resolve
            max_bars = 36  # ~3 hours on 5m
            future_df = future_df.head(max_bars)

            resolved = False
            result = None

            for _, candle in future_df.iterrows():
                high = float(candle["high"])
                low = float(candle["low"])

                if side == "BUY":
                    hit_target = high >= target
                    hit_stop = low <= stop

                    if hit_target and hit_stop:
                        # Conservative: count as loss if both hit same candle
                        result = "LOSS"
                        resolved = True
                        break
                    elif hit_target:
                        result = "WIN"
                        resolved = True
                        break
                    elif hit_stop:
                        result = "LOSS"
                        resolved = True
                        break

                elif side == "SELL":
                    hit_target = low <= target
                    hit_stop = high >= stop

                    if hit_target and hit_stop:
                        result = "LOSS"
                        resolved = True
                        break
                    elif hit_target:
                        result = "WIN"
                        resolved = True
                        break
                    elif hit_stop:
                        result = "LOSS"
                        resolved = True
                        break

            if resolved:
                df.at[idx, "status"] = result
                df.at[idx, "resolved_at"] = datetime.now(ZoneInfo("America/Chicago")).strftime("%Y-%m-%d %H:%M:%S")
                updated = True
            else:
                # If signal is old enough and still unresolved, expire it
                age_minutes = (datetime.now(timezone.utc) - signal_time.to_pydatetime()).total_seconds() / 60
                if age_minutes > 180:  # 3 hours
                    df.at[idx, "status"] = "EXPIRED"
                    df.at[idx, "resolved_at"] = datetime.now(ZoneInfo("America/Chicago")).strftime("%Y-%m-%d %H:%M:%S")
                    updated = True

        if updated:
            df.to_csv(SIGNALS_FILE, index=False)
            print("Updated signal results.")

    except Exception as e:
        print(f"update_signal_results error: {e}")

# =========================================================
# BYBIT CLOSED TRADE HISTORY
# =========================================================
def fetch_bybit_closed_trades():
    """
    Pull recent closed PnL / trade history from Bybit and append new rows.
    NOTE:
    Bybit API fields can vary slightly by account type / endpoint.
    This function is built defensively.
    """
    try:
        existing_ids = set()
        if os.path.exists(BYBIT_TRADES_FILE):
            existing_df = pd.read_csv(BYBIT_TRADES_FILE)
            if not existing_df.empty and "trade_id" in existing_df.columns:
                existing_ids = set(existing_df["trade_id"].astype(str).tolist())

        # Pull realized PnL records
        response = bybit.get_closed_pnl(
            category="linear",
            limit=50
        )

        items = response.get("result", {}).get("list", [])
        if not items:
            print("No closed Bybit trades found.")
            return

        new_rows = []

        for item in items:
            trade_id = str(
                item.get("orderId")
                or item.get("execId")
                or item.get("id")
                or item.get("createdTime")
                or uuid.uuid4()
            )

            if trade_id in existing_ids:
                continue

            symbol = item.get("symbol", "")
            side = item.get("side", "")
            qty = item.get("qty") or item.get("closedSize") or item.get("size") or ""
            avg_entry = item.get("avgEntryPrice") or item.get("entryPrice") or ""
            avg_exit = item.get("avgExitPrice") or item.get("exitPrice") or item.get("markPrice") or ""
            realized_pnl = item.get("closedPnl") or item.get("realisedPnl") or item.get("realizedPnl") or ""

            opened_at = item.get("createdTime") or item.get("updatedTime") or ""
            closed_at = item.get("updatedTime") or item.get("createdTime") or ""

            def fmt_ms(ms):
                try:
                    return datetime.fromtimestamp(int(ms)/1000, tz=ZoneInfo("America/Chicago")).strftime("%Y-%m-%d %H:%M:%S")
                except:
                    return str(ms)

            new_rows.append([
                trade_id,
                symbol,
                side,
                qty,
                avg_entry,
                avg_exit,
                realized_pnl,
                fmt_ms(opened_at),
                fmt_ms(closed_at),
                str(item)
            ])

        if new_rows:
            with open(BYBIT_TRADES_FILE, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerows(new_rows)

            print(f"Saved {len(new_rows)} new Bybit closed trades.")
        else:
            print("No new Bybit closed trades.")

    except Exception as e:
        print(f"fetch_bybit_closed_trades error: {e}")

# =========================================================
# PERFORMANCE ENGINE
# =========================================================
def generate_performance_summary():
    """
    Builds a quick performance summary from:
    1) signal outcomes
    2) actual Bybit closed trades
    """
    try:
        rows = []
        now_str = datetime.now(ZoneInfo("America/Chicago")).strftime("%Y-%m-%d %H:%M:%S")

        # -------------------------
        # SIGNAL STATS
        # -------------------------
        if os.path.exists(SIGNALS_FILE):
            sig = pd.read_csv(SIGNALS_FILE)

            if not sig.empty:
                resolved = sig[sig["status"].isin(["WIN", "LOSS"])].copy()

                if not resolved.empty:
                    total = len(resolved)
                    wins = len(resolved[resolved["status"] == "WIN"])
                    losses = len(resolved[resolved["status"] == "LOSS"])
                    win_rate = round((wins / total) * 100, 2) if total > 0 else 0

                    rows.extend([
                        [now_str, "signal_total_resolved", "all", total],
                        [now_str, "signal_wins", "all", wins],
                        [now_str, "signal_losses", "all", losses],
                        [now_str, "signal_win_rate", "all", win_rate],
                    ])

                    # By symbol
                    by_symbol = resolved.groupby("symbol")["status"].apply(
                        lambda x: round((x == "WIN").mean() * 100, 2)
                    )
                    for symbol, wr in by_symbol.items():
                        rows.append([now_str, "signal_win_rate", symbol, wr])

                    # By grade
                    by_grade = resolved.groupby("grade")["status"].apply(
                        lambda x: round((x == "WIN").mean() * 100, 2)
                    )
                    for grade, wr in by_grade.items():
                        rows.append([now_str, "signal_win_rate_grade", grade, wr])

                    # By side
                    by_side = resolved.groupby("side")["status"].apply(
                        lambda x: round((x == "WIN").mean() * 100, 2)
                    )
                    for side, wr in by_side.items():
                        rows.append([now_str, "signal_win_rate_side", side, wr])

        # -------------------------
        # REAL BYBIT TRADE STATS
        # -------------------------
        if os.path.exists(BYBIT_TRADES_FILE):
            trades = pd.read_csv(BYBIT_TRADES_FILE)

            if not trades.empty and "realized_pnl" in trades.columns:
                trades["realized_pnl"] = pd.to_numeric(trades["realized_pnl"], errors="coerce").fillna(0)

                total_trades = len(trades)
                pnl_sum = round(trades["realized_pnl"].sum(), 4)
                pnl_avg = round(trades["realized_pnl"].mean(), 4) if total_trades > 0 else 0
                wins = len(trades[trades["realized_pnl"] > 0])
                losses = len(trades[trades["realized_pnl"] <= 0])
                win_rate = round((wins / total_trades) * 100, 2) if total_trades > 0 else 0

                rows.extend([
                    [now_str, "bybit_total_closed_trades", "all", total_trades],
                    [now_str, "bybit_total_realized_pnl", "all", pnl_sum],
                    [now_str, "bybit_avg_realized_pnl", "all", pnl_avg],
                    [now_str, "bybit_win_rate", "all", win_rate],
                ])

                by_symbol = trades.groupby("symbol")["realized_pnl"].agg(["sum", "mean", "count"])
                for symbol, vals in by_symbol.iterrows():
                    rows.append([now_str, "bybit_symbol_pnl_sum", symbol, round(vals["sum"], 4)])
                    rows.append([now_str, "bybit_symbol_pnl_avg", symbol, round(vals["mean"], 4)])
                    rows.append([now_str, "bybit_symbol_trade_count", symbol, int(vals["count"])])

        if rows:
            with open(PERFORMANCE_FILE, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["generated_at", "metric", "group", "value"])
                writer.writerows(rows)

            print("Performance summary generated.")

    except Exception as e:
        print(f"generate_performance_summary error: {e}")

# =========================================================
# ADAPTIVE FILTERING
# =========================================================
def get_symbol_signal_win_rate(symbol):
    """
    Returns recent resolved signal win rate for symbol.
    If not enough data, returns None.
    """
    try:
        if not os.path.exists(SIGNALS_FILE):
            return None

        df = pd.read_csv(SIGNALS_FILE)
        if df.empty:
            return None

        df = df[(df["symbol"] == symbol) & (df["status"].isin(["WIN", "LOSS"]))].copy()
        if len(df) < 5:
            return None

        recent = df.tail(20)
        win_rate = (recent["status"] == "WIN").mean() * 100
        return round(win_rate, 2)
    except:
        return None

def should_allow_symbol(symbol):
    """
    Adaptive suppression:
    If a symbol has enough recent history and it's underperforming badly,
    suppress it.
    """
    wr = get_symbol_signal_win_rate(symbol)
    if wr is None:
        return True  # not enough data yet

    # Suppress if recent performance is ugly
    if wr < 35:
        print(f"{symbol}: suppressed by adaptive filter (recent WR {wr}%)")
        return False

    return True

def should_allow_grade(grade):
    """
    Conservative mode:
    If you want ONLY A+ for now, return grade == "A+"
    """
    return grade == "A+"  # debt survival mode

# =========================================================
# CRYPTO SNIPER SCANNER
# =========================================================
def scan_crypto_intraday():
    for symbol in CRYPTO_SYMBOLS:
        try:
            if not should_allow_symbol(symbol):
                continue

            df_5m = get_bybit_klines(symbol, "5", 250)
            df_15m = get_bybit_klines(symbol, "15", 250)
            df_1h = get_bybit_klines(symbol, "60", 250)

            if (
                df_5m is None or df_15m is None or df_1h is None or
                len(df_5m) < 60 or len(df_15m) < 60 or len(df_1h) < 60
            ):
                print(f"{symbol}: missing Bybit timeframe data")
                continue

            # ===== INDICATORS =====
            df_5m["ema9"] = df_5m["close"].ewm(span=9, adjust=False).mean()
            df_5m["ema20"] = df_5m["close"].ewm(span=20, adjust=False).mean()
            df_5m["rsi"] = compute_rsi(df_5m["close"])
            df_5m["vwap"] = compute_vwap_bybit(df_5m)
            df_5m["atr"] = compute_atr(df_5m, 14)
            df_5m["vol_avg20"] = df_5m["volume"].rolling(20).mean()

            df_15m["ema20"] = df_15m["close"].ewm(span=20, adjust=False).mean()
            df_15m["ema50"] = df_15m["close"].ewm(span=50, adjust=False).mean()

            df_1h["ema20"] = df_1h["close"].ewm(span=20, adjust=False).mean()
            df_1h["ema50"] = df_1h["close"].ewm(span=50, adjust=False).mean()

            latest_5m = df_5m.iloc[-1]
            prev_5m = df_5m.iloc[-2]
            latest_15m = df_15m.iloc[-1]
            latest_1h = df_1h.iloc[-1]

            price = float(latest_5m["close"])
            ema9 = float(latest_5m["ema9"])
            ema20 = float(latest_5m["ema20"])
            rsi = float(latest_5m["rsi"])
            vwap = float(latest_5m["vwap"])
            atr = float(latest_5m["atr"]) if pd.notna(latest_5m["atr"]) else None
            volume = float(latest_5m["volume"])
            vol_avg = float(latest_5m["vol_avg20"]) if pd.notna(latest_5m["vol_avg20"]) else 0

            if atr is None or atr <= 0:
                print(f"{symbol}: ATR unavailable")
                continue

            # ===== TREND FILTERS =====
            trend_15m_bull = latest_15m["close"] > latest_15m["ema20"] and latest_15m["ema20"] > latest_15m["ema50"]
            trend_15m_bear = latest_15m["close"] < latest_15m["ema20"] and latest_15m["ema20"] < latest_15m["ema50"]

            trend_1h_bull = latest_1h["close"] > latest_1h["ema20"] and latest_1h["ema20"] > latest_1h["ema50"]
            trend_1h_bear = latest_1h["close"] < latest_1h["ema20"] and latest_1h["ema20"] < latest_1h["ema50"]

            # ===== 5M ENTRY STRUCTURE =====
            fresh_bull_cross = prev_5m["ema9"] <= prev_5m["ema20"] and latest_5m["ema9"] > latest_5m["ema20"]
            fresh_bear_cross = prev_5m["ema9"] >= prev_5m["ema20"] and latest_5m["ema9"] < latest_5m["ema20"]

            ema_bull_structure = price > ema9 > ema20
            ema_bear_structure = price < ema9 < ema20

            volume_spike = volume > (vol_avg * 1.3) if vol_avg > 0 else False

            breakout_bull = breakout_long(df_5m, lookback=10)
            breakout_bear = breakout_short(df_5m, lookback=10)

            choppy = is_choppy_market(df_5m)

            # ===== SCORING =====
            long_score = 0
            short_score = 0

            if trend_15m_bull:
                long_score += 2
            if trend_1h_bull:
                long_score += 2
            if fresh_bull_cross:
                long_score += 2
            if ema_bull_structure:
                long_score += 1
            if 52 <= rsi <= 67:
                long_score += 1
            if price > vwap:
                long_score += 1
            if volume_spike:
                long_score += 2
            if breakout_bull:
                long_score += 2
            if not choppy:
                long_score += 1

            if trend_15m_bear:
                short_score += 2
            if trend_1h_bear:
                short_score += 2
            if fresh_bear_cross:
                short_score += 2
            if ema_bear_structure:
                short_score += 1
            if 33 <= rsi <= 48:
                short_score += 1
            if price < vwap:
                short_score += 1
            if volume_spike:
                short_score += 2
            if breakout_bear:
                short_score += 2
            if not choppy:
                short_score += 1

            long_grade = grade_signal(long_score)
            short_grade = grade_signal(short_score)

            timestamp = datetime.now(ZoneInfo("America/Chicago")).strftime("%Y-%m-%d %H:%M:%S")

            # ===== LONG SIGNAL =====
            if long_grade and should_allow_grade(long_grade):
                entry = price
                stop = round(entry - (atr * 1.2), 4)
                target = round(entry + (atr * 2.2), 4)
                qty = get_crypto_qty(symbol)

                rr = (target - entry) / (entry - stop) if (entry - stop) > 0 else 0
                if rr < 1.7:
                    print(f"{symbol}: long skipped, RR too low")
                    continue

                alert_key = f"{symbol}-BUY-{df_5m.iloc[-1]['timestamp']}-{long_grade}"
                if alert_key not in sent_alerts:
                    sent_alerts.add(alert_key)

                    setup = (
                        f"Sniper long | score={long_score} | "
                        f"15m trend + 1h trend + EMA structure + RSI + VWAP + "
                        f"{'volume spike' if volume_spike else 'no vol spike'} + "
                        f"{'breakout' if breakout_bull else 'no breakout'}"
                    )

                    signal_id = log_signal(
                        symbol=symbol,
                        side="BUY",
                        grade=long_grade,
                        score=long_score,
                        entry=entry,
                        stop=stop,
                        target=target,
                        rsi=rsi,
                        atr=atr,
                        vwap=vwap,
                        volume_spike=volume_spike,
                        breakout=breakout_bull,
                        choppy=choppy,
                        trend_15m="bull",
                        trend_1h="bull"
                    )

                    message = (
                        f"🔥 {long_grade} CRYPTO BUY SIGNAL\n"
                        f"Signal ID: {signal_id}\n"
                        f"Ticker: {symbol}\n"
                        f"Entry: {entry:.4f}\n"
                        f"Stop: {stop}\n"
                        f"Target: {target}\n"
                        f"RR: {rr:.2f}\n"
                        f"Score: {long_score}/14\n"
                        f"RSI: {rsi:.2f}\n"
                        f"VWAP: {vwap:.4f}\n"
                        f"ATR: {atr:.4f}\n"
                        f"Volume Spike: {'YES' if volume_spike else 'NO'}\n"
                        f"Breakout: {'YES' if breakout_bull else 'NO'}\n"
                        f"Choppy Market: {'YES' if choppy else 'NO'}\n"
                        f"Trend: 15m + 1h bullish\n\n"
                        f"🚀 BYBIT APPROVAL COMMAND:\n"
                        f"TRADE {symbol} BUY {qty} {stop} {target}"
                    )

                    send_telegram_message(message)
                    log_alert(timestamp, symbol, "crypto", "BUY", "5m/15m/1h", round(entry, 4), stop, target, round(rsi, 2), setup, long_grade)
                    print(f"Sent {long_grade} BUY alert for {symbol}")

            # ===== SHORT SIGNAL =====
            elif short_grade and should_allow_grade(short_grade):
                entry = price
                stop = round(entry + (atr * 1.2), 4)
                target = round(entry - (atr * 2.2), 4)
                qty = get_crypto_qty(symbol)

                rr = (entry - target) / (stop - entry) if (stop - entry) > 0 else 0
                if rr < 1.7:
                    print(f"{symbol}: short skipped, RR too low")
                    continue

                alert_key = f"{symbol}-SELL-{df_5m.iloc[-1]['timestamp']}-{short_grade}"
                if alert_key not in sent_alerts:
                    sent_alerts.add(alert_key)

                    setup = (
                        f"Sniper short | score={short_score} | "
                        f"15m trend + 1h trend + EMA structure + RSI + VWAP + "
                        f"{'volume spike' if volume_spike else 'no vol spike'} + "
                        f"{'breakout' if breakout_bear else 'no breakout'}"
                    )

                    signal_id = log_signal(
                        symbol=symbol,
                        side="SELL",
                        grade=short_grade,
                        score=short_score,
                        entry=entry,
                        stop=stop,
                        target=target,
                        rsi=rsi,
                        atr=atr,
                        vwap=vwap,
                        volume_spike=volume_spike,
                        breakout=breakout_bear,
                        choppy=choppy,
                        trend_15m="bear",
                        trend_1h="bear"
                    )

                    message = (
                        f"🔻 {short_grade} CRYPTO SELL SIGNAL\n"
                        f"Signal ID: {signal_id}\n"
                        f"Ticker: {symbol}\n"
                        f"Entry: {entry:.4f}\n"
                        f"Stop: {stop}\n"
                        f"Target: {target}\n"
                        f"RR: {rr:.2f}\n"
                        f"Score: {short_score}/14\n"
                        f"RSI: {rsi:.2f}\n"
                        f"VWAP: {vwap:.4f}\n"
                        f"ATR: {atr:.4f}\n"
                        f"Volume Spike: {'YES' if volume_spike else 'NO'}\n"
                        f"Breakout: {'YES' if breakout_bear else 'NO'}\n"
                        f"Choppy Market: {'YES' if choppy else 'NO'}\n"
                        f"Trend: 15m + 1h bearish\n\n"
                        f"🚀 BYBIT APPROVAL COMMAND:\n"
                        f"TRADE {symbol} SELL {qty} {stop} {target}"
                    )

                    send_telegram_message(message)
                    log_alert(timestamp, symbol, "crypto", "SELL", "5m/15m/1h", round(entry, 4), stop, target, round(rsi, 2), setup, short_grade)
                    print(f"Sent {short_grade} SELL alert for {symbol}")

            else:
                print(
                    f"{symbol}: no sniper setup | "
                    f"price={price:.4f}, "
                    f"long_score={long_score}, short_score={short_score}, "
                    f"rsi={rsi:.2f}, atr={atr:.4f}, "
                    f"vol_spike={'YES' if volume_spike else 'NO'}, "
                    f"choppy={'YES' if choppy else 'NO'}"
                )

        except Exception as e:
            print(f"Error on {symbol}: {e}")

# =========================================================
# STOCK SCANNER (unchanged core)
# =========================================================
def analyze_intraday_symbol(ticker: str, asset_type: str):
    df_5m = get_data(ticker, interval="5m", period="5d")
    df_15m = get_data(ticker, interval="15m", period="10d")

    if df_5m.empty or df_15m.empty:
        print(f"{ticker}: missing timeframe data")
        return

    close_5m = to_series(df_5m["Close"]).dropna()
    volume_5m = to_series(df_5m["Volume"]).dropna()
    close_15m = to_series(df_15m["Close"]).dropna()

    if len(close_5m) < 30 or len(volume_5m) < 30 or len(close_15m) < 30:
        print(f"{ticker}: not enough cleaned data")
        return

    ema9_5m = close_5m.ewm(span=9, adjust=False).mean()
    ema20_5m = close_5m.ewm(span=20, adjust=False).mean()
    rsi_5m = calculate_rsi(close_5m, 14)
    vol_avg_5m = volume_5m.rolling(20).mean()
    vwap_5m = compute_vwap(df_5m)

    ema9_15m = close_15m.ewm(span=9, adjust=False).mean()
    ema20_15m = close_15m.ewm(span=20, adjust=False).mean()

    last_price = float(close_5m.iloc[-1])
    last_ema9_5m = float(ema9_5m.iloc[-1])
    last_ema20_5m = float(ema20_5m.iloc[-1])
    prev_ema9_5m = float(ema9_5m.iloc[-2])
    prev_ema20_5m = float(ema20_5m.iloc[-2])
    last_rsi_5m = float(rsi_5m.iloc[-1])
    last_vol_5m = float(volume_5m.iloc[-1])
    last_vol_avg_5m = float(vol_avg_5m.iloc[-1]) if pd.notna(vol_avg_5m.iloc[-1]) else 0
    last_vwap_5m = float(vwap_5m.iloc[-1]) if pd.notna(vwap_5m.iloc[-1]) else last_price

    last_ema9_15m = float(ema9_15m.iloc[-1])
    last_ema20_15m = float(ema20_15m.iloc[-1])

    trend_bullish_15m = last_ema9_15m > last_ema20_15m
    trend_bearish_15m = last_ema9_15m < last_ema20_15m

    fresh_bullish_5m = prev_ema9_5m <= prev_ema20_5m and last_ema9_5m > last_ema20_5m
    fresh_bearish_5m = prev_ema9_5m >= prev_ema20_5m and last_ema9_5m < last_ema20_5m

    bullish = (
        trend_bullish_15m
        and fresh_bullish_5m
        and 52 <= last_rsi_5m <= 68
        and last_vol_5m > last_vol_avg_5m
        and last_price > last_vwap_5m
    )

    bearish = (
        trend_bearish_15m
        and fresh_bearish_5m
        and 32 <= last_rsi_5m <= 48
        and last_vol_5m > last_vol_avg_5m
        and last_price < last_vwap_5m
    )

    asset_label = "stock" if asset_type == "stock" else "crypto"
    timestamp = datetime.now(ZoneInfo("America/Chicago")).strftime("%Y-%m-%d %H:%M:%S")

    if bullish:
        alert_key = f"{ticker}-INTRADAY-BUY-{str(close_5m.index[-1])}"
        if alert_key not in sent_alerts:
            sent_alerts.add(alert_key)
            stop = round(last_price * 0.992, 4)
            target = round(last_price * 1.015, 4)
            setup = "15m bullish trend + 5m fresh bullish crossover + RSI/volume/VWAP confirmation"

            msg = (
                f"🔥 POTENTIAL INTRADAY {asset_label.upper()} BUY\n"
                f"Ticker: {ticker}\n"
                f"Timeframe: 5m entry / 15m trend\n"
                f"Entry: {last_price:.4f}\n"
                f"Stop: {stop:.4f}\n"
                f"Target: {target:.4f}\n"
                f"RSI(5m): {last_rsi_5m:.2f}\n"
                f"VWAP: {last_vwap_5m:.4f}\n"
                f"Setup: {setup}"
            )

            send_telegram_message(msg)
            log_alert(timestamp, ticker, asset_label, "BUY", "5m/15m", round(last_price, 4), stop, target, round(last_rsi_5m, 2), setup)
            print(f"Sent intraday BUY alert for {ticker}")

    elif bearish:
        alert_key = f"{ticker}-INTRADAY-SELL-{str(close_5m.index[-1])}"
        if alert_key not in sent_alerts:
            sent_alerts.add(alert_key)
            risk = round(last_price * 1.008, 4)
            target = round(last_price * 0.985, 4)
            setup = "15m bearish trend + 5m fresh bearish crossover + RSI/volume/VWAP confirmation"

            msg = (
                f"🔥 POTENTIAL INTRADAY {asset_label.upper()} SELL\n"
                f"Ticker: {ticker}\n"
                f"Timeframe: 5m entry / 15m trend\n"
                f"Entry: {last_price:.4f}\n"
                f"Risk Line: {risk:.4f}\n"
                f"Target: {target:.4f}\n"
                f"RSI(5m): {last_rsi_5m:.2f}\n"
                f"VWAP: {last_vwap_5m:.4f}\n"
                f"Setup: {setup}"
            )

            send_telegram_message(msg)
            log_alert(timestamp, ticker, asset_label, "SELL", "5m/15m", round(last_price, 4), risk, target, round(last_rsi_5m, 2), setup)
            print(f"Sent intraday SELL alert for {ticker}")

    else:
        print(
            f"{ticker}: no intraday setup | "
            f"price={last_price:.4f}, 5m_ema9={last_ema9_5m:.4f}, 5m_ema20={last_ema20_5m:.4f}, "
            f"15m_trend={'bull' if trend_bullish_15m else 'bear' if trend_bearish_15m else 'flat'}, "
            f"rsi_5m={last_rsi_5m:.2f}, vwap={last_vwap_5m:.4f}"
        )

# =========================================================
# DAILY / PERIODIC STATUS PUSH
# =========================================================
def maybe_send_performance_snapshot():
    """
    Sends a lightweight summary to Telegram once in a while.
    For now, every run is okay, but you can make this smarter later.
    """
    try:
        if not os.path.exists(PERFORMANCE_FILE):
            return

        perf = pd.read_csv(PERFORMANCE_FILE)
        if perf.empty:
            return

        def get_metric(metric, group="all", default="N/A"):
            row = perf[(perf["metric"] == metric) & (perf["group"] == group)]
            if row.empty:
                return default
            return row.iloc[-1]["value"]

        signal_wr = get_metric("signal_win_rate")
        bybit_wr = get_metric("bybit_win_rate")
        bybit_pnl = get_metric("bybit_total_realized_pnl")

        print(f"Snapshot -> signal_wr={signal_wr}, bybit_wr={bybit_wr}, bybit_pnl={bybit_pnl}")

    except Exception as e:
        print(f"maybe_send_performance_snapshot error: {e}")

# =========================================================
# MAIN LOOP
# =========================================================
def main():
    if not BOT_TOKEN or not CHAT_ID:
        raise ValueError("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID first.")

    init_files()

    send_telegram_message("✅ Scanner bot v8 started (Full tracking / Sniper Mode / Approval mode)")
    print("Bot started successfully. Entering main loop...")

    while True:
        print("\n==============================")
        print("Running new scan cycle...")
        print("==============================")

        try:
            # 1) Update old signals first
            update_signal_results()

            # 2) Pull Bybit closed trades
            fetch_bybit_closed_trades()

            # 3) Rebuild performance stats
            generate_performance_summary()

            # 4) STOCKS
            if is_stock_market_open():
                for ticker in STOCK_TICKERS:
                    try:
                        analyze_intraday_symbol(ticker, "stock")
                    except Exception as e:
                        print(f"Error on stock {ticker}: {e}")
            else:
                print("Stock market is closed. Skipping stock intraday scan.")

            # 5) CRYPTO
            try:
                scan_crypto_intraday()
            except Exception as e:
                print(f"Error in crypto Bybit scan: {e}")

            # 6) Optional status check
            maybe_send_performance_snapshot()

        except Exception as e:
            print(f"Main loop error: {e}")

        time.sleep(CHECK_EVERY_SECONDS)

if __name__ == "__main__":
    main()