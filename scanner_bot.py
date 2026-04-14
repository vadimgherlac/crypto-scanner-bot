import os
import time
import csv
import requests
import numpy as np
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

SIGNALS_FILE    = "trade_signals_v16.csv"
DAILY_LOCK_FILE = "daily_risk_lock_v16.csv"
PAIR_STATS_FILE = "pair_stats_v16.csv"

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
# VECTOR SETTINGS
# -------------------------
# Minimum cosine similarity to historical winners to count as pattern match
PATTERN_SIMILARITY_THRESHOLD = 0.72
# Minimum closed trades before dynamic weights activate
MIN_TRADES_FOR_DYNAMIC_WEIGHTS = 10
# How much the vector confidence can add to the rule score (max bonus)
MAX_VECTOR_BONUS = 4
# Min vector confidence (0-1) required to fire a signal
MIN_VECTOR_CONFIDENCE = 0.35

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

def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def compute_atr_df(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return compute_atr(df["high"].astype(float), df["low"].astype(float), df["close"].astype(float), period)

def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    high  = high.astype(float).reset_index(drop=True)
    low   = low.astype(float).reset_index(drop=True)
    close = close.astype(float).reset_index(drop=True)
    plus_dm  = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    mask = plus_dm > minus_dm
    minus_dm[mask]  = 0
    plus_dm[~mask]  = 0
    prev_close = close.shift(1)
    tr = pd.concat([high-low,(high-prev_close).abs(),(low-prev_close).abs()],axis=1).max(axis=1)
    atr      = tr.rolling(period).mean()
    plus_di  = 100 * (plus_dm.rolling(period).mean()  / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
    dx       = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9))
    adx      = dx.rolling(period).mean()
    return adx, plus_di, minus_di

def get_data(ticker: str, interval: str, period: str) -> pd.DataFrame:
    df = yf.download(
        ticker, period=period, interval=interval,
        auto_adjust=True, progress=False,
        group_by="column", threads=False,
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

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
# ██╗   ██╗███████╗ ██████╗████████╗ ██████╗ ██████╗
# ██║   ██║██╔════╝██╔════╝╚══██╔══╝██╔═══██╗██╔══██╗
# ██║   ██║█████╗  ██║        ██║   ██║   ██║██████╔╝
# ╚██╗ ██╔╝██╔══╝  ██║        ██║   ██║   ██║██╔══██╗
#  ╚████╔╝ ███████╗╚██████╗   ██║   ╚██████╔╝██║  ██║
#   ╚═══╝  ╚══════╝ ╚═════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝
# ALGORITHMICS ENGINE
# =========================================================

# ---------------------------------------------------------
# 1. MARKET REGIME DETECTION
# ---------------------------------------------------------
# Classifies current market into one of three regimes:
#   TRENDING  — strong directional move, ADX high, ATR expanding
#   RANGING   — price oscillating, ADX low, ATR flat/low
#   VOLATILE  — large erratic swings, ATR high but ADX low
#
# Why it matters:
#   - In RANGING markets, breakout signals are usually fake
#   - In VOLATILE markets, stops get hit randomly
#   - In TRENDING markets, momentum signals are most reliable
#   Signal thresholds and score weights adjust per regime.
# ---------------------------------------------------------

REGIME_TRENDING  = "TRENDING"
REGIME_RANGING   = "RANGING"
REGIME_VOLATILE  = "VOLATILE"

def detect_market_regime(df: pd.DataFrame) -> tuple[str, dict]:
    """
    Returns (regime_str, regime_meta) where regime_meta contains
    the raw values used for classification.
    """
    if len(df) < 50:
        return REGIME_RANGING, {}

    close  = df["close"].astype(float)
    high   = df["high"].astype(float)
    low    = df["low"].astype(float)

    # ADX
    adx_series, plus_di, minus_di = compute_adx(high, low, close)
    adx_val = float(adx_series.iloc[-1]) if pd.notna(adx_series.iloc[-1]) else 0

    # ATR percentile (current ATR vs its own 50-bar history)
    atr_series   = compute_atr(high, low, close, 14)
    atr_now      = float(atr_series.iloc[-1]) if pd.notna(atr_series.iloc[-1]) else 0
    atr_history  = atr_series.dropna().tail(50)
    atr_pct      = float(np.percentile(atr_history, 75)) if len(atr_history) > 10 else atr_now
    atr_expanded = atr_now > atr_pct   # ATR above its own 75th pct = expanding

    # EMA slope (rate of change of EMA20 over last 5 bars)
    ema20       = close.ewm(span=20, adjust=False).mean()
    ema_slope   = (float(ema20.iloc[-1]) - float(ema20.iloc[-6])) / (float(ema20.iloc[-6]) + 1e-9)
    strong_slope = abs(ema_slope) > 0.002   # >0.2% move in EMA over 5 bars

    meta = {
        "adx": round(adx_val, 1),
        "atr_expanded": atr_expanded,
        "ema_slope": round(ema_slope * 100, 3),
        "strong_slope": strong_slope,
    }

    if adx_val >= 25 and strong_slope:
        return REGIME_TRENDING, meta
    elif adx_val < 20 and atr_expanded:
        return REGIME_VOLATILE, meta
    elif adx_val < 20:
        return REGIME_RANGING, meta
    else:
        # ADX 20-25, mild trend
        return REGIME_TRENDING, meta

def regime_score_multiplier(regime: str) -> float:
    """
    Multiplier applied to the final rule score based on regime.
    Trending = full credit, Ranging = penalised, Volatile = heavily penalised.
    """
    return {
        REGIME_TRENDING:  1.0,
        REGIME_RANGING:   0.75,
        REGIME_VOLATILE:  0.6,
    }.get(regime, 1.0)

def regime_min_score(base_min: int, regime: str) -> int:
    """Raise the bar in regimes where signals are less reliable."""
    delta = {
        REGIME_TRENDING:  0,
        REGIME_RANGING:   2,
        REGIME_VOLATILE:  3,
    }.get(regime, 0)
    return base_min + delta

# ---------------------------------------------------------
# 2. FEATURE VECTOR BUILDER
# ---------------------------------------------------------
# Encodes the current market state as a fixed-length numeric
# vector for use in pattern matching and dynamic weights.
#
# Features (all normalised 0-1 or -1 to 1):
#   [0]  RSI / 100
#   [1]  (price - vwap) / atr           — VWAP distance in ATR units
#   [2]  (ema9 - ema20) / atr           — EMA spread in ATR units
#   [3]  volume / vol_avg - 1           — Volume surge ratio
#   [4]  atr / price                    — ATR as % of price (volatility)
#   [5]  (price - low_20) / (high_20 - low_20)  — position in 20-bar range
#   [6]  adx / 100
#   [7]  ema_slope (clipped -1..1)
#   [8]  1 if liq_sweep else 0
#   [9]  1 if bos else 0
#   [10] 1 if bull/bear candle else 0
#   [11] 1 if rsi_div else 0
# ---------------------------------------------------------

FEATURE_NAMES = [
    "rsi_norm", "vwap_dist", "ema_spread", "vol_surge",
    "atr_pct", "range_pos", "adx_norm", "ema_slope",
    "liq_sweep", "bos", "strong_candle", "rsi_div"
]
N_FEATURES = len(FEATURE_NAMES)

def build_feature_vector(
    price:      float,
    rsi:        float,
    vwap:       float,
    ema9:       float,
    ema20:      float,
    atr:        float,
    vol:        float,
    vol_avg:    float,
    adx:        float,
    ema_slope:  float,
    high_20:    float,
    low_20:     float,
    liq_sweep:  bool,
    bos:        bool,
    strong_can: bool,
    rsi_div:    bool,
    side:       str,        # "BUY" or "SELL" — flips sign-sensitive features
) -> np.ndarray:
    """
    Returns a normalised feature vector. Sign-sensitive features
    are flipped for SELL so cosine similarity works direction-aware.
    """
    direction = 1.0 if side == "BUY" else -1.0

    rsi_norm   = rsi / 100.0
    vwap_dist  = ((price - vwap) / (atr + 1e-9)) * direction
    ema_spread = ((ema9 - ema20) / (atr + 1e-9)) * direction
    vol_surge  = min((vol / (vol_avg + 1e-9)) - 1.0, 3.0)  # cap at 3x
    atr_pct    = min(atr / (price + 1e-9), 0.1)            # cap at 10%
    range_pos  = (price - low_20) / (high_20 - low_20 + 1e-9)
    if side == "SELL":
        range_pos = 1.0 - range_pos    # flip: SELL prefers top of range
    adx_norm   = min(adx / 100.0, 1.0)
    slope_clip = max(min(ema_slope * direction, 1.0), -1.0)
    liq_f      = 1.0 if liq_sweep else 0.0
    bos_f      = 1.0 if bos else 0.0
    can_f      = 1.0 if strong_can else 0.0
    div_f      = 1.0 if rsi_div else 0.0

    vec = np.array([
        rsi_norm, vwap_dist, ema_spread, vol_surge,
        atr_pct, range_pos, adx_norm, slope_clip,
        liq_f, bos_f, can_f, div_f
    ], dtype=float)

    # L2 normalise so cosine similarity = dot product
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-9:
        return 0.0
    return float(np.dot(a, b) / denom)

# ---------------------------------------------------------
# 3. HISTORICAL PATTERN LIBRARY
# ---------------------------------------------------------
# Loads closed WIN signals from the CSV, encodes each as a
# feature vector, and stores them as a matrix for fast
# similarity search.
# ---------------------------------------------------------

class PatternLibrary:
    """
    Maintains a matrix of normalised feature vectors for all
    historical WIN trades. Used for cosine similarity search.
    """
    def __init__(self):
        self.win_vectors:  list[np.ndarray] = []
        self.loss_vectors: list[np.ndarray] = []
        self._last_loaded: datetime | None  = None

    def _should_reload(self) -> bool:
        if self._last_loaded is None:
            return True
        return (datetime.now() - self._last_loaded).seconds > 300  # reload every 5 min

    def load(self):
        if not self._should_reload():
            return
        self.win_vectors  = []
        self.loss_vectors = []

        if not os.path.exists(SIGNALS_FILE):
            self._last_loaded = datetime.now()
            return

        try:
            df     = pd.read_csv(SIGNALS_FILE)
            closed = df[df["status"].isin(["WIN","LOSS"])].copy()
            if closed.empty:
                self._last_loaded = datetime.now()
                return

            for _, row in closed.iterrows():
                try:
                    # Reconstruct a feature vector from logged values
                    # (we don't have all fields but use what we have)
                    entry   = float(row["entry"])
                    stop    = float(row["stop"])
                    rsi     = float(row["rsi"])
                    vwap    = float(row["vwap"])
                    atr     = float(row["atr"])
                    adx     = float(row.get("adx", 25))
                    side    = str(row["side"])
                    score   = float(row["score"])

                    # Approximate missing fields from what we have
                    ema_spread_approx = (entry - stop) / (atr + 1e-9) * (1 if side=="BUY" else -1)
                    rr = abs(float(row.get("rr", 2.0)))

                    vec = np.array([
                        rsi / 100.0,
                        (entry - vwap) / (atr + 1e-9) * (1 if side=="BUY" else -1),
                        ema_spread_approx,
                        0.5,                          # vol surge unknown
                        min(atr / (entry + 1e-9), 0.1),
                        0.5,                          # range pos unknown
                        min(adx / 100.0, 1.0),
                        0.0,                          # ema slope unknown
                        1.0 if score >= 10 else 0.0,  # proxy for liq sweep
                        1.0 if score >= 12 else 0.0,  # proxy for BOS
                        0.5,
                        0.0,
                    ], dtype=float)

                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        vec = vec / norm

                    if row["status"] == "WIN":
                        self.win_vectors.append(vec)
                    else:
                        self.loss_vectors.append(vec)
                except:
                    continue

            self._last_loaded = datetime.now()
            print(f"PatternLibrary: {len(self.win_vectors)} wins, {len(self.loss_vectors)} losses loaded")

        except Exception as e:
            print(f"PatternLibrary load error: {e}")
            self._last_loaded = datetime.now()

    def pattern_confidence(self, query_vec: np.ndarray, top_k: int = 5) -> float:
        """
        Returns a confidence score 0.0-1.0 based on how similar
        the current setup is to historical winners vs losers.

        Formula:
          conf = avg_win_sim / (avg_win_sim + avg_loss_sim + eps)

        Falls back to 0.5 (neutral) if not enough history.
        """
        self.load()

        if len(self.win_vectors) < 3:
            return 0.5   # neutral — not enough history yet

        # Top-K similarity to winners
        win_sims  = sorted(
            [cosine_similarity(query_vec, w) for w in self.win_vectors],
            reverse=True
        )[:top_k]
        avg_win = float(np.mean(win_sims))

        # Top-K similarity to losers
        if len(self.loss_vectors) >= 3:
            loss_sims = sorted(
                [cosine_similarity(query_vec, l) for l in self.loss_vectors],
                reverse=True
            )[:top_k]
            avg_loss = float(np.mean(loss_sims))
        else:
            avg_loss = 0.0

        conf = avg_win / (avg_win + avg_loss + 1e-9)
        return round(float(np.clip(conf, 0.0, 1.0)), 3)

# Global pattern library (loaded once, reloaded every 5 min)
PATTERN_LIB = PatternLibrary()

# ---------------------------------------------------------
# 4. DYNAMIC WEIGHT ENGINE
# ---------------------------------------------------------
# Reads closed trades and computes per-feature win rates.
# Features that have historically predicted wins get higher
# weight in the score. Features that haven't, get lower weight.
#
# Default weights = V15 fixed weights.
# Dynamic weights kick in after MIN_TRADES_FOR_DYNAMIC_WEIGHTS.
# ---------------------------------------------------------

# Base weights matching V15 scoring (feature → base points)
BASE_WEIGHTS = {
    "trend_4h":    2,
    "trend_1h":    2,
    "trend_15m":   2,
    "adx_ok":      1,
    "adx_dir":     1,
    "liq":         2,
    "trap":        1,
    "bos":         1,
    "fresh_cross": 1,
    "engulf":      2,
    "pin_bar":     1,
    "inside_bar":  1,
    "strong_can":  1,
    "vwap":        1,
    "rsi":         1,
    "volume":      1,
    "rsi_div":     2,
    "near_poc":    1,
    "room":        1,
    "btc_filter":  1,
}

class DynamicWeightEngine:
    """
    Tracks feature-level performance and adjusts weights.
    Weights are updated every cycle from the signals CSV.
    """
    def __init__(self):
        self.weights: dict[str, float] = dict(BASE_WEIGHTS)
        self._last_updated: datetime | None = None

    def _should_update(self) -> bool:
        if self._last_updated is None:
            return True
        return (datetime.now() - self._last_updated).seconds > 300

    def update(self):
        if not self._should_update():
            return
        if not os.path.exists(SIGNALS_FILE):
            self._last_updated = datetime.now()
            return

        try:
            df     = pd.read_csv(SIGNALS_FILE)
            closed = df[df["status"].isin(["WIN","LOSS"])].copy()

            if len(closed) < MIN_TRADES_FOR_DYNAMIC_WEIGHTS:
                self._last_updated = datetime.now()
                return

            # Parse setup strings to extract which features fired
            # Format: "V16 long: liq=True trap=False bos=True ..."
            feature_wins  = {k: 0 for k in BASE_WEIGHTS}
            feature_total = {k: 0 for k in BASE_WEIGHTS}

            for _, row in closed.iterrows():
                setup  = str(row.get("setup",""))
                is_win = row["status"] == "WIN"
                score  = float(row.get("score", 0))

                # Use score as proxy for feature presence
                # (exact feature tracking added in V16 setup string)
                for feat in BASE_WEIGHTS:
                    if feat in setup:
                        feature_total[feat] += 1
                        if is_win:
                            feature_wins[feat] += 1

            # Compute new weights: features with >60% win rate get boosted
            # features with <40% win rate get reduced
            new_weights = {}
            for feat, base in BASE_WEIGHTS.items():
                total = feature_total[feat]
                if total < 5:
                    new_weights[feat] = base   # not enough data, keep base
                    continue
                win_rate = feature_wins[feat] / total
                # Scale: 0.5 win rate → 1.0x, 1.0 → 1.5x, 0.0 → 0.5x
                multiplier = 0.5 + win_rate
                new_weights[feat] = round(base * multiplier, 2)

            self.weights = new_weights
            self._last_updated = datetime.now()
            print(f"DynamicWeights updated from {len(closed)} trades")

        except Exception as e:
            print(f"DynamicWeight update error: {e}")
            self._last_updated = datetime.now()

    def get(self, feature: str) -> float:
        return self.weights.get(feature, BASE_WEIGHTS.get(feature, 1))

# Global weight engine
WEIGHT_ENGINE = DynamicWeightEngine()

# =========================================================
# EXISTING PRICE ACTION FUNCTIONS
# =========================================================
def compute_rsi_series(series: pd.Series, period: int = 14) -> pd.Series:
    return compute_rsi(series, period)

def compute_vwap_yf(df: pd.DataFrame) -> pd.Series:
    high   = to_series(df["High"]).astype(float)
    low    = to_series(df["Low"]).astype(float)
    close  = to_series(df["Close"]).astype(float)
    volume = to_series(df["Volume"]).astype(float)
    tp     = (high + low + close) / 3
    return (tp * volume).cumsum() / volume.cumsum().replace(0, pd.NA)

def liquidity_sweep_long(df: pd.DataFrame, lookback: int = 20) -> bool:
    if len(df) < lookback + 2: return False
    recent_low = df["low"].iloc[-(lookback+2):-2].min()
    latest     = df.iloc[-1]
    return latest["low"] < recent_low and latest["close"] > recent_low

def liquidity_sweep_short(df: pd.DataFrame, lookback: int = 20) -> bool:
    if len(df) < lookback + 2: return False
    recent_high = df["high"].iloc[-(lookback+2):-2].max()
    latest      = df.iloc[-1]
    return latest["high"] > recent_high and latest["close"] < recent_high

def break_of_structure_long(df: pd.DataFrame, lookback: int = 10) -> bool:
    if len(df) < lookback + 2: return False
    return df["close"].iloc[-1] > df["high"].iloc[-(lookback+2):-2].max()

def break_of_structure_short(df: pd.DataFrame, lookback: int = 10) -> bool:
    if len(df) < lookback + 2: return False
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
    if len(df) < lookback + 2: return False
    recent_low = df["low"].iloc[-(lookback+2):-2].min()
    latest     = df.iloc[-1]
    return latest["low"] < recent_low and latest["close"] > recent_low

def fake_breakout_trap_short(df: pd.DataFrame, lookback: int = 20) -> bool:
    if len(df) < lookback + 2: return False
    recent_high = df["high"].iloc[-(lookback+2):-2].max()
    latest      = df.iloc[-1]
    return latest["high"] > recent_high and latest["close"] < recent_high

def is_bullish_engulfing(df: pd.DataFrame) -> bool:
    if len(df) < 2: return False
    prev, curr = df.iloc[-2], df.iloc[-1]
    return (prev["close"] < prev["open"] and curr["close"] > curr["open"] and
            curr["open"] < prev["close"] and curr["close"] > prev["open"])

def is_bearish_engulfing(df: pd.DataFrame) -> bool:
    if len(df) < 2: return False
    prev, curr = df.iloc[-2], df.iloc[-1]
    return (prev["close"] > prev["open"] and curr["close"] < curr["open"] and
            curr["open"] > prev["close"] and curr["close"] < prev["open"])

def is_hammer(df: pd.DataFrame) -> bool:
    latest = df.iloc[-1]
    body   = abs(latest["close"] - latest["open"])
    rng    = latest["high"] - latest["low"]
    if rng <= 0: return False
    lower_wick = min(latest["open"], latest["close"]) - latest["low"]
    upper_wick = latest["high"] - max(latest["open"], latest["close"])
    return lower_wick >= 2 * body and upper_wick <= 0.3 * rng

def is_shooting_star(df: pd.DataFrame) -> bool:
    latest = df.iloc[-1]
    body   = abs(latest["close"] - latest["open"])
    rng    = latest["high"] - latest["low"]
    if rng <= 0: return False
    upper_wick = latest["high"] - max(latest["open"], latest["close"])
    lower_wick = min(latest["open"], latest["close"]) - latest["low"]
    return upper_wick >= 2 * body and lower_wick <= 0.3 * rng

def is_inside_bar_breakout_long(df: pd.DataFrame) -> bool:
    if len(df) < 3: return False
    mother, inside, current = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    return (inside["high"] < mother["high"] and inside["low"] > mother["low"] and
            current["close"] > mother["high"])

def is_inside_bar_breakout_short(df: pd.DataFrame) -> bool:
    if len(df) < 3: return False
    mother, inside, current = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    return (inside["high"] < mother["high"] and inside["low"] > mother["low"] and
            current["close"] < mother["low"])

def detect_rsi_divergence(close: pd.Series, rsi: pd.Series, lookback: int = 20):
    if len(close) < lookback + 2: return False, False
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

def compute_poc(df: pd.DataFrame, bins: int = 20) -> float:
    if len(df) < 20: return float(df["close"].iloc[-1])
    price_min = df["low"].min()
    price_max = df["high"].max()
    if price_max <= price_min: return float(df["close"].iloc[-1])
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

def smart_stop_long(df: pd.DataFrame, atr: float, lookback: int = 10) -> float:
    recent_low = df["low"].iloc[-(lookback+2):-1].min()
    return round(recent_low - (0.25 * atr), 4)

def smart_stop_short(df: pd.DataFrame, atr: float, lookback: int = 10) -> float:
    recent_high = df["high"].iloc[-(lookback+2):-1].max()
    return round(recent_high + (0.25 * atr), 4)

def is_choppy_market(ema9: float, ema20: float, atr: float) -> bool:
    if atr <= 0: return True
    return abs(ema9 - ema20) < (atr * 0.12)

def enough_room_long(entry: float, target: float, atr: float) -> bool:
    return (target - entry) > (atr * 1.2)

def enough_room_short(entry: float, target: float, atr: float) -> bool:
    return (entry - target) > (atr * 1.2)

def grade_signal(score: float) -> str:
    if score >= 14: return "A+"
    elif score >= 10: return "A"
    elif score >= 7:  return "B"
    return "IGNORE"

# =========================================================
# SESSION / POSITION / RISK HELPERS
# =========================================================
def get_crypto_session_strength() -> str:
    hour = datetime.now(ZoneInfo("America/Chicago")).hour
    if 2 <= hour <= 11:   return "HIGH"
    elif 12 <= hour <= 16 or 20 <= hour <= 23: return "MID"
    return "LOW"

def is_stock_market_open() -> bool:
    now  = datetime.now(ZoneInfo("America/New_York"))
    if now.weekday() >= 5: return False
    mins = now.hour * 60 + now.minute
    return (9*60+30) <= mins <= (16*60)

def get_btc_market_bias() -> str:
    try:
        df = get_bybit_klines("BTCUSDT", "15", 200)
        if df is None or len(df) < 50: return "neutral"
        df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
        df["vwap"]  = compute_vwap_bybit(df)
        latest      = df.iloc[-1]
        price, ema20, vwap = float(latest["close"]), float(latest["ema20"]), float(latest["vwap"])
        if price > ema20 and price > vwap: return "bull"
        elif price < ema20 and price < vwap: return "bear"
        return "neutral"
    except: return "neutral"

def get_crypto_qty(symbol: str, entry: float, stop: float) -> str:
    fallback = {
        "BTCUSDT":0.001,"ETHUSDT":0.01,"SOLUSDT":1,"XRPUSDT":25,
        "DOGEUSDT":200,"BNBUSDT":0.05,"AVAXUSDT":2,"LINKUSDT":2,
        "ADAUSDT":50,"LTCUSDT":0.5,"DOTUSDT":10,"ATOMUSDT":5,
        "NEARUSDT":8,"OPUSDT":20
    }
    stop_dist = abs(entry - stop)
    if stop_dist <= 0: return str(fallback.get(symbol, 1))
    raw = (ACCOUNT_BALANCE * RISK_PER_TRADE_PCT) / stop_dist
    if symbol == "BTCUSDT": qty = round(raw, 3)
    elif symbol in ["ETHUSDT","BNBUSDT","LTCUSDT"]: qty = round(raw, 2)
    elif symbol in ["SOLUSDT","AVAXUSDT","LINKUSDT","ATOMUSDT","NEARUSDT","OPUSDT"]: qty = round(raw, 1)
    else: qty = round(raw)
    return str(qty if qty > 0 else fallback.get(symbol, 1))

def calculate_rr(entry: float, stop: float, target: float) -> float:
    risk = abs(entry - stop)
    return round(abs(target - entry) / risk, 2) if risk > 0 else 0

# =========================================================
# FILE INIT / LOGGING
# =========================================================
def ensure_signal_file():
    if not os.path.exists(SIGNALS_FILE):
        with open(SIGNALS_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "timestamp","symbol","asset_type","side","timeframe",
                "entry","stop","tp1","tp2","rsi","vwap","atr","adx",
                "score","vector_conf","regime","grade","session_strength",
                "rr","setup","status","closed_at"
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
               score, vector_conf, regime, grade, session_strength, rr, setup):
    ensure_signal_file()
    with open(SIGNALS_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            timestamp, symbol, asset_type, side, timeframe,
            entry, stop, tp1, tp2, rsi, vwap, atr, adx,
            score, vector_conf, regime, grade, session_strength, rr,
            setup, "OPEN", ""
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
    except: pass
    return {"date": today, "loss_count": 0, "daily_risk_used": 0.0, "cooldown_until": ""}

def write_daily_lock(loss_count, daily_risk_used, cooldown_until=""):
    ensure_daily_lock_file()
    today = get_today()
    rows  = []
    try:
        df   = pd.read_csv(DAILY_LOCK_FILE)
        rows = df[df["date"] != today].values.tolist()
    except: pass
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
            if datetime.now(ZoneInfo("America/Chicago")).replace(tzinfo=None) < until:
                return True, f"Cooldown until {until}"
        except: pass
    return False, "OK"

def register_loss():
    lock     = read_daily_lock()
    cooldown = (datetime.now(ZoneInfo("America/Chicago")).replace(tzinfo=None) +
                timedelta(minutes=COOLDOWN_AFTER_LOSS_MINUTES)).isoformat()
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
        if closed.empty: return
        rows = []
        for sym, grp in closed.groupby("symbol"):
            total = len(grp)
            wins  = (grp["status"] == "WIN").sum()
            rows.append([sym, total, wins, total-wins, round(wins/total*100, 2)])
        with open(PAIR_STATS_FILE, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["symbol","total_closed","wins","losses","win_rate"])
            w.writerows(rows)
    except: pass

def is_coin_blacklisted(symbol: str) -> bool:
    try:
        df  = pd.read_csv(PAIR_STATS_FILE)
        row = df[df["symbol"] == symbol]
        if row.empty: return False
        row = row.iloc[-1]
        if int(row["total_closed"]) < MIN_TRADES_FOR_COIN_FILTER: return False
        return float(row["win_rate"]) < MIN_COIN_WINRATE_TO_TRADE
    except: return False

# =========================================================
# TRADE OUTCOME CHECKER
# =========================================================
def update_signal_results():
    ensure_signal_file()
    try: df = pd.read_csv(SIGNALS_FILE)
    except: return
    if df.empty: return

    updated = False
    now     = datetime.now(ZoneInfo("America/Chicago")).replace(tzinfo=None)

    for i, row in df.iterrows():
        if row["status"] != "OPEN": continue
        symbol      = row["symbol"]
        asset_type  = row["asset_type"]
        side        = row["side"]
        stop        = float(row["stop"])
        tp2         = float(row["tp2"])
        signal_time = pd.to_datetime(row["timestamp"])

        signal_time_naive = signal_time.tz_localize(None) if signal_time.tzinfo is None else signal_time.tz_convert(None)
        if (now - signal_time_naive).total_seconds() / 3600 > 24:
            df.at[i,"status"]    = "EXPIRED"
            df.at[i,"closed_at"] = now.strftime("%Y-%m-%d %H:%M:%S")
            updated = True
            continue

        try:
            if asset_type == "crypto":
                future  = get_bybit_klines(symbol, "5", 300)
                if future is None or len(future) < 10: continue
                candles = future[future["timestamp"] > signal_time]
                for _, c in candles.iterrows():
                    h, l = float(c["high"]), float(c["low"])
                    if side == "BUY":
                        if l <= stop:   df.at[i,"status"]="LOSS"; df.at[i,"closed_at"]=str(c["timestamp"]); register_loss(); updated=True; break
                        if h >= tp2:    df.at[i,"status"]="WIN";  df.at[i,"closed_at"]=str(c["timestamp"]); updated=True; break
                    else:
                        if h >= stop:   df.at[i,"status"]="LOSS"; df.at[i,"closed_at"]=str(c["timestamp"]); register_loss(); updated=True; break
                        if l <= tp2:    df.at[i,"status"]="WIN";  df.at[i,"closed_at"]=str(c["timestamp"]); updated=True; break
            else:
                future  = get_data(symbol, "5m", "5d")
                if future.empty: continue
                candles = future[future.index > signal_time]
                for idx, c in candles.iterrows():
                    h = float(c["High"]); l = float(c["Low"])
                    if side == "BUY":
                        if l <= stop:   df.at[i,"status"]="LOSS"; df.at[i,"closed_at"]=str(idx); register_loss(); updated=True; break
                        if h >= tp2:    df.at[i,"status"]="WIN";  df.at[i,"closed_at"]=str(idx); updated=True; break
                    else:
                        if h >= stop:   df.at[i,"status"]="LOSS"; df.at[i,"closed_at"]=str(idx); register_loss(); updated=True; break
                        if l <= tp2:    df.at[i,"status"]="WIN";  df.at[i,"closed_at"]=str(idx); updated=True; break
        except Exception as e:
            print(f"Outcome check error {symbol}: {e}")

    if updated:
        df.to_csv(SIGNALS_FILE, index=False)
        rebuild_pair_stats()

# =========================================================
# DAILY STATS + DIAGNOSTIC
# =========================================================
def send_daily_stats_report():
    try:
        df     = pd.read_csv(SIGNALS_FILE)
        closed = df[df["status"].isin(["WIN","LOSS"])].copy()
        if closed.empty: return
        total    = len(closed)
        wins     = (closed["status"] == "WIN").sum()
        win_rate = round(wins / total * 100, 2)
        coin_wr  = closed.groupby("symbol")["status"].apply(
            lambda x: round((x=="WIN").mean()*100,1)).sort_values(ascending=False)
        grade_wr = closed.groupby("grade")["status"].apply(
            lambda x: round((x=="WIN").mean()*100,1)).sort_values(ascending=False)
        regime_wr = closed.groupby("regime")["status"].apply(
            lambda x: round((x=="WIN").mean()*100,1)).sort_values(ascending=False) \
            if "regime" in closed.columns else pd.Series()

        msg = (
            f"📊 V16 DAILY REPORT\n\n"
            f"Total: {total} | Wins: {wins} | Losses: {total-wins}\n"
            f"Win Rate: {win_rate}%\n\n"
            f"🏆 By Coin:\n" + "\n".join([f"{k}: {v}%" for k,v in coin_wr.head(5).items()]) + "\n\n"
            f"🎯 By Grade:\n" + "\n".join([f"{k}: {v}%" for k,v in grade_wr.items()]) + "\n\n"
            f"📈 By Regime:\n" + ("\n".join([f"{k}: {v}%" for k,v in regime_wr.items()]) if not regime_wr.empty else "N/A")
        )
        send_telegram_message(msg)
    except Exception as e:
        print(f"Stats report error: {e}")

def build_diagnostic_report(scan_log: list) -> str:
    if not scan_log: return "No scan data this cycle."
    best  = sorted(scan_log, key=lambda x: max(x["long_score"], x["short_score"]), reverse=True)[:5]
    lines = ["🔍 V16 DIAGNOSTIC — Top Candidates\n"]
    for e in best:
        bl = ", ".join(e["blockers_long"])  or "✅ clear"
        bs = ", ".join(e["blockers_short"]) or "✅ clear"
        vc = e.get("vector_conf", "?")
        re = e.get("regime", "?")
        lines.append(
            f"{e['symbol']} [{re}] vec={vc}\n"
            f"  Long  {e['long_score']:>5.1f} | {bl}\n"
            f"  Short {e['short_score']:>5.1f} | {bs}"
        )
    return "\n".join(lines)

# =========================================================
# CORE SIGNAL BUILDER  (V15 rules + V16 vector layer)
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
    # ── Update vector engines (cached, runs at most every 5 min) ──
    WEIGHT_ENGINE.update()
    W = WEIGHT_ENGINE  # shorthand

    # ── Base indicators ──
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
                         "blockers_long":["atr=0"],"blockers_short":["atr=0"],
                         "vector_conf":0,"regime":"UNKNOWN"})
        return

    # ── STEP 1: REGIME DETECTION ──
    regime, regime_meta = detect_market_regime(df_5m)
    score_mult  = regime_score_multiplier(regime)
    ema_slope   = regime_meta.get("ema_slope", 0.0)
    adx_val_reg = regime_meta.get("adx", 0.0)

    # ── Trend ──
    trend_15m_bull = float(df_15m.iloc[-1]["close"]) > float(df_15m.iloc[-1]["ema20"])
    trend_15m_bear = not trend_15m_bull
    trend_1h_bull  = float(df_1h.iloc[-1]["close"])  > float(df_1h.iloc[-1]["ema20"])
    trend_1h_bear  = not trend_1h_bull
    trend_4h_bull  = float(df_4h.iloc[-1]["close"])  > float(df_4h.iloc[-1]["ema20"])
    trend_4h_bear  = not trend_4h_bull

    # ── ADX ──
    adx_series, plus_di, minus_di = compute_adx(df_5m["high"], df_5m["low"], df_5m["close"])
    adx_val  = float(adx_series.iloc[-1]) if pd.notna(adx_series.iloc[-1]) else 0
    adx_ok   = adx_val >= 18
    adx_bull = adx_ok and float(plus_di.iloc[-1]) > float(minus_di.iloc[-1])
    adx_bear = adx_ok and float(minus_di.iloc[-1]) > float(plus_di.iloc[-1])

    # ── PA signals ──
    liq_long    = liquidity_sweep_long(df_5m,  20)
    liq_short   = liquidity_sweep_short(df_5m, 20)
    bos_long    = break_of_structure_long(df_5m,  10)
    bos_short   = break_of_structure_short(df_5m, 10)
    trap_long   = fake_breakdown_trap_long(df_5m,  20)
    trap_short  = fake_breakout_trap_short(df_5m, 20)
    bull_candle = strong_bullish_candle(df_5m)
    bear_candle = strong_bearish_candle(df_5m)
    bull_engulf = is_bullish_engulfing(df_5m)
    bear_engulf = is_bearish_engulfing(df_5m)
    hammer      = is_hammer(df_5m)
    shoot_star  = is_shooting_star(df_5m)
    ib_long     = is_inside_bar_breakout_long(df_5m)
    ib_short    = is_inside_bar_breakout_short(df_5m)
    bull_div, bear_div = detect_rsi_divergence(df_5m["close"], df_5m["rsi"])

    poc        = compute_poc(df_5m)
    near_value = near_poc(price, poc, atr)

    # PA recent (last 3 bars)
    def _pa_long_recent(lookback=3):
        for i in range(lookback):
            end = len(df_5m) - i
            if end < 30: break
            low_w  = df_5m["low"].iloc[end-21:end-1]
            high_w = df_5m["high"].iloc[end-12:end-1]
            cl     = df_5m["close"].iloc[end-1]
            lo     = df_5m["low"].iloc[end-1]
            if (lo < low_w.min() and cl > low_w.min()) or cl > high_w.max():
                return True
        return False

    def _pa_short_recent(lookback=3):
        for i in range(lookback):
            end = len(df_5m) - i
            if end < 30: break
            high_w = df_5m["high"].iloc[end-21:end-1]
            low_w  = df_5m["low"].iloc[end-12:end-1]
            cl     = df_5m["close"].iloc[end-1]
            hi     = df_5m["high"].iloc[end-1]
            if (hi > high_w.max() and cl < high_w.max()) or cl < low_w.min():
                return True
        return False

    pa_long_ok  = _pa_long_recent()
    pa_short_ok = _pa_short_recent()

    fresh_bull = float(prev["ema9"]) <= float(prev["ema20"]) and ema9 > ema20
    fresh_bear = float(prev["ema9"]) >= float(prev["ema20"]) and ema9 < ema20

    volume_ok    = vol > (vol_avg * 1.2) if vol_avg > 0 else False
    rsi_buy_ok   = 48 <= rsi <= 70
    rsi_sell_ok  = 30 <= rsi <= 52
    vwap_buy_ok  = price > vwap
    vwap_sell_ok = price < vwap

    high_20  = df_5m["high"].iloc[-22:-2].max()
    low_20   = df_5m["low"].iloc[-22:-2].min()

    target_long  = price + (2.5 * atr)
    target_short = price - (2.5 * atr)
    room_long    = enough_room_long(price,  target_long,  atr)
    room_short   = enough_room_short(price, target_short, atr)
    choppy       = is_choppy_market(ema9, ema20, atr)

    btc_buy_ok  = True if symbol == "BTCUSDT" else btc_bias in ["bull","neutral"]
    btc_sell_ok = True if symbol == "BTCUSDT" else btc_bias in ["bear","neutral"]

    # ── STEP 2: RULE SCORE (dynamic weights) ──
    long_score = 0.0
    long_score += W.get("trend_4h")    * (1 if trend_4h_bull  else 0)
    long_score += W.get("trend_1h")    * (1 if trend_1h_bull  else 0)
    long_score += W.get("trend_15m")   * (1 if trend_15m_bull else 0)
    long_score += W.get("adx_ok")      * (1 if adx_ok         else 0)
    long_score += W.get("adx_dir")     * (1 if adx_bull       else 0)
    long_score += W.get("liq")         * (1 if liq_long        else 0)
    long_score += W.get("trap")        * (1 if trap_long       else 0)
    long_score += W.get("bos")         * (1 if bos_long        else 0)
    long_score += W.get("fresh_cross") * (1 if fresh_bull      else 0)
    long_score += W.get("engulf")      * (1 if bull_engulf     else 0)
    long_score += W.get("pin_bar")     * (1 if hammer          else 0)
    long_score += W.get("inside_bar")  * (1 if ib_long         else 0)
    long_score += W.get("strong_can")  * (1 if bull_candle     else 0)
    long_score += W.get("vwap")        * (1 if vwap_buy_ok     else 0)
    long_score += W.get("rsi")         * (1 if rsi_buy_ok      else 0)
    long_score += W.get("volume")      * (1 if volume_ok       else 0)
    long_score += W.get("rsi_div")     * (1 if bull_div        else 0)
    long_score += W.get("near_poc")    * (1 if near_value      else 0)
    long_score += W.get("room")        * (1 if room_long       else 0)
    long_score += W.get("btc_filter")  * (1 if btc_buy_ok      else 0)
    if choppy:   long_score -= 3
    if not adx_ok: long_score -= 1
    if session_strength == "HIGH":  long_score += 1
    elif session_strength == "LOW": long_score -= 1

    short_score = 0.0
    short_score += W.get("trend_4h")    * (1 if trend_4h_bear  else 0)
    short_score += W.get("trend_1h")    * (1 if trend_1h_bear  else 0)
    short_score += W.get("trend_15m")   * (1 if trend_15m_bear else 0)
    short_score += W.get("adx_ok")      * (1 if adx_ok         else 0)
    short_score += W.get("adx_dir")     * (1 if adx_bear       else 0)
    short_score += W.get("liq")         * (1 if liq_short       else 0)
    short_score += W.get("trap")        * (1 if trap_short      else 0)
    short_score += W.get("bos")         * (1 if bos_short       else 0)
    short_score += W.get("fresh_cross") * (1 if fresh_bear      else 0)
    short_score += W.get("engulf")      * (1 if bear_engulf     else 0)
    short_score += W.get("pin_bar")     * (1 if shoot_star      else 0)
    short_score += W.get("inside_bar")  * (1 if ib_short        else 0)
    short_score += W.get("strong_can")  * (1 if bear_candle     else 0)
    short_score += W.get("vwap")        * (1 if vwap_sell_ok    else 0)
    short_score += W.get("rsi")         * (1 if rsi_sell_ok     else 0)
    short_score += W.get("volume")      * (1 if volume_ok       else 0)
    short_score += W.get("rsi_div")     * (1 if bear_div        else 0)
    short_score += W.get("near_poc")    * (1 if near_value      else 0)
    short_score += W.get("room")        * (1 if room_short      else 0)
    short_score += W.get("btc_filter")  * (1 if btc_sell_ok     else 0)
    if choppy:    short_score -= 3
    if not adx_ok: short_score -= 1
    if session_strength == "HIGH":  short_score += 1
    elif session_strength == "LOW": short_score -= 1

    # Apply regime multiplier to rule scores
    long_score  = long_score  * score_mult
    short_score = short_score * score_mult

    # ── STEP 3: VECTOR CONFIDENCE ──
    # Build feature vectors for both directions, get pattern confidence
    long_vec = build_feature_vector(
        price, rsi, vwap, ema9, ema20, atr, vol, vol_avg, adx_val, ema_slope,
        high_20, low_20, liq_long or trap_long, bos_long,
        bull_candle or bull_engulf, bull_div, "BUY"
    )
    short_vec = build_feature_vector(
        price, rsi, vwap, ema9, ema20, atr, vol, vol_avg, adx_val, ema_slope,
        high_20, low_20, liq_short or trap_short, bos_short,
        bear_candle or bear_engulf, bear_div, "SELL"
    )

    long_conf  = PATTERN_LIB.pattern_confidence(long_vec)
    short_conf = PATTERN_LIB.pattern_confidence(short_vec)

    # Vector bonus: up to MAX_VECTOR_BONUS points added to rule score
    long_bonus  = round(long_conf  * MAX_VECTOR_BONUS, 2)
    short_bonus = round(short_conf * MAX_VECTOR_BONUS, 2)

    long_score_final  = long_score  + long_bonus
    short_score_final = short_score + short_bonus

    long_grade  = grade_signal(long_score_final)
    short_grade = grade_signal(short_score_final)

    # ── STEP 4: THRESHOLDS (regime-aware) ──
    base_min    = 8 if session_strength == "HIGH" else 9 if session_strength == "MID" else 10
    min_score_l = regime_min_score(base_min, regime)
    min_score_s = regime_min_score(base_min, regime)

    grade_buy_ok  = long_grade  in (["A+"] if A_PLUS_ONLY_MODE else ["A+","A"])
    grade_sell_ok = short_grade in (["A+"] if A_PLUS_ONLY_MODE else ["A+","A"])

    # ── STEP 5: SIGNAL GATES ──
    # Hard gates (non-negotiable):
    #   1H + 15m trend, PA confirm, VWAP side, room, not choppy, BTC filter
    # Vector gate:
    #   confidence must be >= MIN_VECTOR_CONFIDENCE (default 0.35)
    #   (0.5 = neutral = no history yet, so this passes until history builds)
    buy_signal = (
        long_score_final >= min_score_l and grade_buy_ok and
        not choppy and
        trend_1h_bull and trend_15m_bull and
        pa_long_ok and vwap_buy_ok and room_long and btc_buy_ok and
        long_conf >= MIN_VECTOR_CONFIDENCE
    )

    sell_signal = (
        short_score_final >= min_score_s and grade_sell_ok and
        not choppy and
        trend_1h_bear and trend_15m_bear and
        pa_short_ok and vwap_sell_ok and room_short and btc_sell_ok and
        short_conf >= MIN_VECTOR_CONFIDENCE
    )

    # ── Diagnostic blockers ──
    blockers_long  = []
    blockers_short = []
    if long_score_final < min_score_l:  blockers_long.append(f"score {long_score_final:.1f}<{min_score_l}")
    if not grade_buy_ok:                blockers_long.append(f"grade={long_grade}")
    if choppy:                          blockers_long.append("choppy")
    if not trend_1h_bull:               blockers_long.append("1H bear")
    if not trend_15m_bull:              blockers_long.append("15m bear")
    if not pa_long_ok:                  blockers_long.append("no PA")
    if not vwap_buy_ok:                 blockers_long.append("<VWAP")
    if not room_long:                   blockers_long.append("no room")
    if not btc_buy_ok:                  blockers_long.append("BTC bear")
    if long_conf < MIN_VECTOR_CONFIDENCE: blockers_long.append(f"vec={long_conf:.2f}")

    if short_score_final < min_score_s: blockers_short.append(f"score {short_score_final:.1f}<{min_score_s}")
    if not grade_sell_ok:               blockers_short.append(f"grade={short_grade}")
    if choppy:                          blockers_short.append("choppy")
    if not trend_1h_bear:               blockers_short.append("1H bull")
    if not trend_15m_bear:              blockers_short.append("15m bull")
    if not pa_short_ok:                 blockers_short.append("no PA")
    if not vwap_sell_ok:                blockers_short.append(">VWAP")
    if not room_short:                  blockers_short.append("no room")
    if not btc_sell_ok:                 blockers_short.append("BTC bull")
    if short_conf < MIN_VECTOR_CONFIDENCE: blockers_short.append(f"vec={short_conf:.2f}")

    scan_log.append({
        "symbol":       symbol,
        "long_score":   long_score_final,
        "short_score":  short_score_final,
        "blockers_long":  blockers_long,
        "blockers_short": blockers_short,
        "vector_conf":  f"L={long_conf:.2f}/S={short_conf:.2f}",
        "regime":       regime,
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
            setup_str = f"V16 long [{regime}]: {pa_str} {can_str} vec={long_conf:.2f}"
            msg = (
                f"🔥 {long_grade} {'CRYPTO' if asset_type=='crypto' else 'STOCK'} BUY\n"
                f"Symbol: {symbol}\n"
                f"Regime: {regime} | Session: {session_strength}\n"
                f"BTC: {btc_bias} | 4H: {'▲' if trend_4h_bull else '▼'}\n"
                f"Entry:  {entry:.4f}\n"
                f"Stop:   {stop}  (swing low)\n"
                f"TP1:    {tp1}  (1.5R)\n"
                f"TP2:    {tp2}  (2.5R)\n"
                f"R:R:    {rr}\n"
                f"Qty:    {qty}\n"
                f"RSI: {rsi:.1f} | VWAP: {vwap:.4f} | ATR: {atr:.4f}\n"
                f"ADX: {adx_val:.1f} | Score: {long_score_final:.1f} (rule={long_score:.1f}+vec={long_bonus})\n"
                f"Vector confidence: {long_conf:.2f}\n"
                f"PA: {pa_str}\n"
                f"Candles: {can_str}\n"
                f"Div: bull={bull_div} | POC near={near_value}"
                + (f"\n\n🚀 BYBIT:\nTRADE {symbol} BUY {qty} {stop} {tp2}" if asset_type=="crypto" else "")
            )
            send_telegram_message(msg)
            log_signal(
                timestamp, symbol, asset_type, "BUY", "5m/15m/1h/4h",
                entry, stop, tp1, tp2, round(rsi,2), round(vwap,4),
                round(atr,4), round(adx_val,1), round(long_score_final,1),
                round(long_conf,3), regime, long_grade, session_strength, rr, setup_str
            )
            print(f"✅ {long_grade} BUY: {symbol} score={long_score_final:.1f} vec={long_conf:.2f} regime={regime}")

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
            sent_alerts.add(key)
            pa_str  = f"liq={liq_short} trap={trap_short} bos={bos_short}"
            can_str = f"engulf={bear_engulf} shootstar={shoot_star} IB={ib_short}"
            setup_str = f"V16 short [{regime}]: {pa_str} {can_str} vec={short_conf:.2f}"
            msg = (
                f"🔻 {short_grade} {'CRYPTO' if asset_type=='crypto' else 'STOCK'} SELL\n"
                f"Symbol: {symbol}\n"
                f"Regime: {regime} | Session: {session_strength}\n"
                f"BTC: {btc_bias} | 4H: {'▲' if trend_4h_bull else '▼'}\n"
                f"Entry:  {entry:.4f}\n"
                f"Stop:   {stop}  (swing high)\n"
                f"TP1:    {tp1}  (1.5R)\n"
                f"TP2:    {tp2}  (2.5R)\n"
                f"R:R:    {rr}\n"
                f"Qty:    {qty}\n"
                f"RSI: {rsi:.1f} | VWAP: {vwap:.4f} | ATR: {atr:.4f}\n"
                f"ADX: {adx_val:.1f} | Score: {short_score_final:.1f} (rule={short_score:.1f}+vec={short_bonus})\n"
                f"Vector confidence: {short_conf:.2f}\n"
                f"PA: {pa_str}\n"
                f"Candles: {can_str}\n"
                f"Div: bear={bear_div} | POC near={near_value}"
                + (f"\n\n🚀 BYBIT:\nTRADE {symbol} SELL {qty} {stop} {tp2}" if asset_type=="crypto" else "")
            )
            send_telegram_message(msg)
            log_signal(
                timestamp, symbol, asset_type, "SELL", "5m/15m/1h/4h",
                entry, stop, tp1, tp2, round(rsi,2), round(vwap,4),
                round(atr,4), round(adx_val,1), round(short_score_final,1),
                round(short_conf,3), regime, short_grade, session_strength, rr, setup_str
            )
            print(f"✅ {short_grade} SELL: {symbol} score={short_score_final:.1f} vec={short_conf:.2f} regime={regime}")

    else:
        print(f"{symbol} [{regime}]: no signal | "
              f"L={long_score_final:.1f}(vec={long_conf:.2f}) "
              f"S={short_score_final:.1f}(vec={short_conf:.2f}) | "
              f"pa_l={pa_long_ok} pa_s={pa_short_ok} choppy={choppy}")

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
        "✅ Scanner bot V16 started\n"
        "Vector Engine: Regime Detection + Pattern Matching + Dynamic Weights\n"
        "Rule score + vector confidence bonus\n"
        "Hard gates: 1H+15m trend, PA, VWAP, room"
    )
    print("Bot V16 started.")

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