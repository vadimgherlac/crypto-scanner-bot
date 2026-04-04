import os
import time
import requests
import pandas as pd
import yfinance as yf

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

TICKERS = ["AAPL", "TSLA", "NVDA", "SPY", "QQQ"]
CHECK_EVERY_SECONDS = 300

sent_alerts = set()

def send_telegram_message(text: str) -> None:
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text}
    r = requests.post(url, json=payload, timeout=20)
    r.raise_for_status()

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def check_symbol(ticker: str):
    df = yf.download(ticker, period="5d", interval="5m", auto_adjust=True, progress=False)

    if df.empty or len(df) < 30:
        return

    close = df["Close"].copy()
    volume = df["Volume"].copy()

    ema9 = close.ewm(span=9).mean()
    ema20 = close.ewm(span=20).mean()
    rsi = calculate_rsi(close, 14)
    vol_avg = volume.rolling(20).mean()

    last_price = float(close.iloc[-1])
    last_ema9 = float(ema9.iloc[-1])
    last_ema20 = float(ema20.iloc[-1])
    last_rsi = float(rsi.iloc[-1])
    last_vol = float(volume.iloc[-1])
    last_vol_avg = float(vol_avg.iloc[-1]) if pd.notna(vol_avg.iloc[-1]) else 0

    bullish = last_ema9 > last_ema20 and last_rsi > 50 and last_vol > last_vol_avg

    if bullish:
        alert_key = f"{ticker}-BUY-{df.index[-1]}"
        if alert_key not in sent_alerts:
            sent_alerts.add(alert_key)
            stop = round(last_price * 0.98, 2)
            target = round(last_price * 1.04, 2)
            msg = (
                f"🚨 BUY SIGNAL\n"
                f"Ticker: {ticker}\n"
                f"Entry: {last_price:.2f}\n"
                f"Stop: {stop:.2f}\n"
                f"Target: {target:.2f}\n"
                f"Setup: EMA9 > EMA20, RSI > 50, volume spike"
            )
            send_telegram_message(msg)

def main():
    if not BOT_TOKEN or not CHAT_ID:
        raise ValueError("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID first.")

    send_telegram_message("✅ Scanner bot started")

    while True:
        for ticker in TICKERS:
            try:
                check_symbol(ticker)
            except Exception as e:
                print(f"Error on {ticker}: {e}")
        time.sleep(CHECK_EVERY_SECONDS)

if __name__ == "__main__":
    main()