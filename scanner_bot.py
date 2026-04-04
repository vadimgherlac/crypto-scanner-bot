import os
import time
import json
import csv
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
from pybit.unified_trading import HTTP

# ===== LOAD ENV =====
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

# ===== BYBIT SESSION =====
bybit = HTTP(
    testnet=False,
    api_key=BYBIT_API_KEY,
    api_secret=BYBIT_API_SECRET
)

# ===== SETTINGS =====
CHECK_EVERY_SECONDS = 60
LOG_FILE = "trade_alerts.csv"
sent_alerts = set()

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

def send_telegram_message(text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": text
    }
    r = requests.post(url, data=payload)
    if not r.ok:
        print("Telegram error:", r.text)
    r.raise_for_status()
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_vwap_bybit(df):
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    return (typical_price * df["volume"]).cumsum() / df["volume"].cumsum()

def scan_crypto_intraday():
    for symbol in CRYPTO_SYMBOLS:
        try:
            df_5m = get_bybit_klines(symbol, "5", 200)
            df_15m = get_bybit_klines(symbol, "15", 200)

            if df_5m is None or df_15m is None or len(df_5m) < 30 or len(df_15m) < 30:
                print(f"{symbol}: missing Bybit timeframe data")
                continue

            # Indicators
            df_5m["ema9"] = df_5m["close"].ewm(span=9).mean()
            df_5m["ema20"] = df_5m["close"].ewm(span=20).mean()
            df_5m["rsi"] = compute_rsi(df_5m["close"])
            df_5m["vwap"] = compute_vwap_bybit(df_5m)

            df_15m["ema20"] = df_15m["close"].ewm(span=20).mean()

            latest_5m = df_5m.iloc[-1]
            prev_5m = df_5m.iloc[-2]
            latest_15m = df_15m.iloc[-1]

            price = float(latest_5m["close"])
            ema9 = float(latest_5m["ema9"])
            ema20 = float(latest_5m["ema20"])
            rsi = float(latest_5m["rsi"])
            vwap = float(latest_5m["vwap"])
            trend_15m = "bull" if latest_15m["close"] > latest_15m["ema20"] else "bear"

            buy_signal = (
                trend_15m == "bull" and
                prev_5m["ema9"] <= prev_5m["ema20"] and
                latest_5m["ema9"] > latest_5m["ema20"] and
                45 <= rsi <= 70 and
                price > vwap
            )

            sell_signal = (
                trend_15m == "bear" and
                prev_5m["ema9"] >= prev_5m["ema20"] and
                latest_5m["ema9"] < latest_5m["ema20"] and
                30 <= rsi <= 55 and
                price < vwap
            )

            if buy_signal:
                entry = price
                stop = round(entry * 0.98, 4)
                target = round(entry * 1.04, 4)
                qty = get_crypto_qty(symbol)

                message = (
                    f"🔥 POTENTIAL INTRADAY CRYPTO BUY\n"
                    f"Ticker: {symbol}\n"
                    f"Entry: {entry:.4f}\n"
                    f"Stop: {stop}\n"
                    f"Target: {target}\n\n"
                    f"🚀 BYBIT APPROVAL COMMAND:\n"
                    f"TRADE {symbol} BUY {qty} {stop} {target}"
                )

                send_telegram_message(message)
                print(f"Sent BUY alert for {symbol}")

            elif sell_signal:
                entry = price
                stop = round(entry * 1.02, 4)
                target = round(entry * 0.96, 4)
                qty = get_crypto_qty(symbol)

                message = (
                    f"🔻 POTENTIAL INTRADAY CRYPTO SELL\n"
                    f"Ticker: {symbol}\n"
                    f"Entry: {entry:.4f}\n"
                    f"Stop: {stop}\n"
                    f"Target: {target}\n\n"
                    f"🚀 BYBIT APPROVAL COMMAND:\n"
                    f"TRADE {symbol} SELL {qty} {stop} {target}"
                )

                send_telegram_message(message)
                print(f"Sent SELL alert for {symbol}")

            else:
                print(
                    f"{symbol}: no intraday setup | "
                    f"price={price:.4f}, "
                    f"5m_ema9={ema9:.4f}, "
                    f"5m_ema20={ema20:.4f}, "
                    f"15m_trend={trend_15m}, "
                    f"rsi_5m={rsi:.2f}, "
                    f"vwap={vwap:.4f}"
                )

        except Exception as e:
            print(f"Error on {symbol}: {e}")

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

def log_alert(timestamp, ticker, asset_type, side, timeframe, entry, stop_or_risk, target, rsi, setup):
    file_exists = os.path.exists(LOG_FILE)

    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "ticker",
                "asset_type",
                "side",
                "timeframe",
                "entry",
                "stop_or_risk",
                "target",
                "rsi",
                "setup"
            ])

        writer.writerow([
            timestamp,
            ticker,
            asset_type,
            side,
            timeframe,
            entry,
            stop_or_risk,
            target,
            rsi,
            setup
        ])


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


def is_stock_market_open() -> bool:
    now = datetime.now(ZoneInfo("America/New_York"))
    if now.weekday() >= 5:
        return False
    current_minutes = now.hour * 60 + now.minute
    market_open_minutes = 9 * 60 + 30
    market_close_minutes = 16 * 60
    return market_open_minutes <= current_minutes <= market_close_minutes


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

            if asset_type == "crypto":
                bybit_symbol = BYBIT_SYMBOL_MAP.get(ticker, "")
                qty = get_crypto_qty(bybit_symbol)
                msg += (
                    f"\n\n🚀 BYBIT APPROVAL COMMAND:\n"
                    f"TRADE {bybit_symbol} BUY {qty} {stop} {target}"
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

            if asset_type == "crypto":
                bybit_symbol = BYBIT_SYMBOL_MAP.get(ticker, "")
                qty = get_crypto_qty(bybit_symbol)
                msg += (
                    f"\n\n🚀 BYBIT APPROVAL COMMAND:\n"
                    f"TRADE {bybit_symbol} SELL {qty} {risk} {target}"
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


def main():
    if not BOT_TOKEN or not CHAT_ID:
        raise ValueError("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID first.")

    send_telegram_message("✅ Scanner bot v6 started (Telegram approval mode)")
    print("Bot started successfully. Entering main loop...")

    while True:
        print("Running new scan cycle...")
        try:
            if is_stock_market_open():
                for ticker in STOCK_TICKERS:
                    try:
                        analyze_intraday_symbol(ticker, "stock")
                    except Exception as e:
                        print(f"Error on stock {ticker}: {e}")
            else:
                print("Stock market is closed. Skipping stock intraday scan.")

            try:
                scan_crypto_intraday()
            except Exception as e:
                print(f"Error in crypto Bybit scan: {e}")

        except Exception as e:
            print(f"Main loop error: {e}")

        time.sleep(CHECK_EVERY_SECONDS)


if __name__ == "__main__":
    main()