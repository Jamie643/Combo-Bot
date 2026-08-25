import os
import sys
import time
import threading
import traceback
import ccxt
import numpy as np
import pandas as pd
import requests
from flask import Flask

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
TARGET_PAIRS = [
    "DOGE/USDT:USDT",
    "GRAM/USDT:USDT",
    "XRP/USDT:USDT",
    "ADA/USDT:USDT",
]

TIMEFRAME = "1h"
OHLCV_LIMIT = 100
MAX_CONCURRENT_POSITIONS_BASE = 2  # Base Risk Gatekeeper
MAX_TOTAL_CRYPTO_POSITIONS = 2     # Universal Correlation Cap (Long + Short)

# Strategy Parameters
RSI_PERIOD = 14
ADX_PERIOD = 14
BB_PERIOD = 20
BB_STD = 2.0
DONCHIAN_PERIOD = 20
FIB_LEVEL = 0.618
FIB_TOLERANCE = 0.005

# Risk & Order Sizing Configuration
RISK_PER_TRADE = 0.02      # Risk 2% of account equity per trade
ATR_PERIOD = 14
ATR_SL_MULTIPLIER = 2.0    # SL = 2.0 * ATR (Wider buffer against noise)
RISK_REWARD_RATIO = 2.0    # TP = 2.0 * SL distance (1:2 Risk/Reward)
VOLUME_SPIKE_MULT = 1.5    # 150% of 20-period average volume required for breakout
MAKER_FEE_RATE = 0.0002    # 0.02% Limit/Maker Fee Rate Target
MIN_ORDER_USD = 5.05       # Bybit minimum order floor
MAX_LEVERAGE_CAP = 10     # Upper safety ceiling for dynamic leverage
MIN_LEVERAGE_CAP = 1      # Minimum leverage floor

# State Tracking
PEAK_EQUITY = 0.0
REGIME_MEMORY = {}        # Tracks consecutive regime bars per pair
COOLDOWN_TRACKER = {}     # Tracks loss cooldowns per symbol {"symbol": bars_left}

# Mainnet Bybit API Credentials
BYBIT_KEY = os.getenv("BYBIT_API_KEY") or os.getenv("BYBIT_KEY")
BYBIT_SECRET = os.getenv("BYBIT_API_SECRET") or os.getenv("BYBIT_SECRET")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

BOT_THREAD_STARTED = False

# ------------------------------------------------------------------
# TELEGRAM NOTIFIER
# ------------------------------------------------------------------
def send_telegram(message_text):
    """Sends Telegram alerts formatted in HTML."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Console Alert:\n", message_text, flush=True)
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message_text,
        "parse_mode": "HTML",
    }
    try:
        res = requests.post(url, json=payload, timeout=15)
        res.raise_for_status()
    except Exception as e:
        print(f"Failed to send Telegram message: {e}", flush=True)

def send_critical_alert(error_msg):
    alert_text = (
        "🚨 <b>CRITICAL BOT ERROR</b> 🚨\n\n"
        f"<b>Error Details:</b>\n<code>{error_msg}</code>"
    )
    send_telegram(alert_text)

# ------------------------------------------------------------------
# TECHNICAL INDICATORS & REGIME DETECTOR
# ------------------------------------------------------------------
def calculate_indicators(df):
    """Calculates technical indicators including Volume Moving Averages."""
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=RSI_PERIOD).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=RSI_PERIOD).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    df["sma20"] = df["close"].rolling(window=BB_PERIOD).mean()
    df["std20"] = df["close"].rolling(window=BB_PERIOD).std()
    df["bb_upper"] = df["sma20"] + (BB_STD * df["std20"])
    df["bb_lower"] = df["sma20"] - (BB_STD * df["std20"])

    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift(1))
    low_close = np.abs(df["low"] - df["close"].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.rolling(window=ATR_PERIOD).mean()
    df["atr_ma"] = df["atr"].rolling(window=20).mean()

    # Volume Confirmation Metric
    df["vol_ma20"] = df["volume"].rolling(window=20).mean()

    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    
    tr_smooth = tr.rolling(window=ADX_PERIOD).sum()
    plus_di = 100 * (pd.Series(plus_dm).rolling(window=ADX_PERIOD).sum() / tr_smooth)
    minus_di = 100 * (pd.Series(minus_dm).rolling(window=ADX_PERIOD).sum() / tr_smooth)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    df["adx"] = dx.rolling(window=ADX_PERIOD).mean()

    df["donchian_high"] = df["high"].shift(1).rolling(window=DONCHIAN_PERIOD).max()
    df["donchian_low"] = df["low"].shift(1).rolling(window=DONCHIAN_PERIOD).min()

    return df

def detect_market_regime(symbol, df):
    """Hysteresis filter: Requires 3 consecutive bars to switch regimes."""
    global REGIME_MEMORY

    adx = df["adx"].iloc[-1]
    atr = df["atr"].iloc[-1]
    atr_ma = df["atr_ma"].iloc[-1]

    raw_trending = (adx > 25) and (atr > atr_ma)
    raw_regime = "TREND" if raw_trending else "RANGE"

    if symbol not in REGIME_MEMORY:
        REGIME_MEMORY[symbol] = {"last_regime": raw_regime, "bars": 1}

    state = REGIME_MEMORY[symbol]

    if raw_regime == state["last_regime"]:
        state["bars"] += 1
        active_regime = raw_regime
    else:
        if state["bars"] < 3:
            active_regime = state["last_regime"]
        else:
            state["last_regime"] = raw_regime
            state["bars"] = 1
            active_regime = raw_regime

    REGIME_MEMORY[symbol] = state
    return active_regime

# ------------------------------------------------------------------
# STRATEGY ENGINE
# ------------------------------------------------------------------
def analyze_market_condition(symbol, df, balance, max_positions):
    cp = df["close"].iloc[-1]
    low_val = df["low"].iloc[-1]
    volume = df["volume"].iloc[-1]
    vol_ma = df["vol_ma20"].iloc[-1]

    rsi = df["rsi"].iloc[-1]
    adx = df["adx"].iloc[-1]
    bb_upper = df["bb_upper"].iloc[-1]
    bb_lower = df["bb_lower"].iloc[-1]
    d_high = df["donchian_high"].iloc[-1]
    d_low = df["donchian_low"].iloc[-1]
    atr = df["atr"].iloc[-1]

    regime = detect_market_regime(symbol, df)

    move_range = d_high - d_low
    fib_618_bull = d_high - (move_range * FIB_LEVEL)

    # Volume Spike Filter
    has_volume_confirmation = volume > (vol_ma * VOLUME_SPIKE_MULT)

    # Donchian Breakout + Volume Filter
    is_donchian_breakout = (cp >= d_high) and has_volume_confirmation

    # Fib Retest Candle Confirmation: Touched Fib level AND closed above it
    touched_fib = low_val <= fib_618_bull
    closed_above_fib = cp > fib_618_bull
    is_fib_confirmed = touched_fib and closed_above_fib

    is_oversold = rsi < 30 or cp <= bb_lower
    is_overbought = rsi > 70 or cp >= bb_upper

    signal = "NEUTRAL"
    bias = f"Regime: {regime} | ADX={adx:.1f}"

    if regime == "TREND":
        if is_donchian_breakout:
            signal = "🚀 LONG BREAKOUT"
            bias = f"Donchian High Break ({d_high:.4f}) + Vol Spike ({volume/vol_ma:.1f}x)"
        elif is_fib_confirmed:
            signal = "🟢 BULLISH FIB RETEST"
            bias = f"Fib 61.8% Retest Confirmed (Closed Above ${fib_618_bull:.4f})"
    elif regime == "RANGE":
        if is_oversold:
            signal = "🟢 MEAN REVERSION LONG"
            bias = f"RSI={rsi:.1f}, BB Lower Reversal"
        elif is_overbought:
            signal = "🔴 MEAN REVERSION SHORT"
            bias = f"RSI={rsi:.1f}, BB Upper Reversal"

    sl_dist = atr * ATR_SL_MULTIPLIER
    tp_dist = sl_dist * RISK_REWARD_RATIO

    if "LONG" in signal:
        sl_price = cp - sl_dist
        tp_price = cp + tp_dist
    elif "SHORT" in signal:
        sl_price = cp + sl_dist
        tp_price = cp - tp_dist
    else:
        sl_price, tp_price = cp, cp

    # Fixed Fractional Sizing
    risk_amount = balance * RISK_PER_TRADE
    units = risk_amount / sl_dist if sl_dist > 0 else 0
    notional_value = units * cp

    if notional_value > 0 and notional_value < MIN_ORDER_USD:
        notional_value = MIN_ORDER_USD
        units = notional_value / cp

    allocated_margin = min(balance * 0.40, balance / max_positions)

    if allocated_margin > 0 and notional_value > 0:
        raw_leverage = notional_value / allocated_margin
        smart_leverage = int(np.ceil(raw_leverage))
        smart_leverage = max(MIN_LEVERAGE_CAP, min(smart_leverage, MAX_LEVERAGE_CAP))
    else:
        smart_leverage = 5

    actual_margin_required = notional_value / smart_leverage

    return {
        "price": cp,
        "signal": signal,
        "bias": bias,
        "sl_price": sl_price,
        "tp_price": tp_price,
        "units": units,
        "margin": actual_margin_required,
        "leverage": smart_leverage,
    }

# ------------------------------------------------------------------
# EXECUTION & PENDING ORDER ENGINE
# ------------------------------------------------------------------
def get_active_positions(exchange):
    try:
        positions = exchange.fetch_positions(params={"category": "linear"})
        active = []
        for p in positions:
            contracts = float(p.get("contracts", 0) or p.get("size", 0))
            if contracts > 0:
                active.append(p)
        return active
    except Exception as e:
        print(f"Error fetching positions: {e}", flush=True)
        return []

def execute_trade(exchange, symbol, res):
    side = "buy" if "LONG" in res["signal"] else "sell"
    qty = res["units"]
    smart_leverage = res["leverage"]

    if qty <= 0:
        return False

    try:
        open_orders = exchange.fetch_open_orders(symbol)
        if len(open_orders) > 0:
            print(f"Skipping entry for {symbol}: Pending order exists.", flush=True)
            return False

        try:
            funding_info = exchange.fetch_funding_rate(symbol)
            funding_rate = float(funding_info.get("fundingRate", 0.0))
            if side == "buy" and funding_rate > 0.0003:
                return False
            elif side == "sell" and funding_rate < -0.0003:
                return False
        except Exception as f_err:
            print(f"Funding rate check warning ({symbol}): {f_err}", flush=True)

        exchange.set_leverage(smart_leverage, symbol)

        ticker = exchange.fetch_ticker(symbol)
        bid = ticker["bid"]
        ask = ticker["ask"]
        limit_price = bid if side == "buy" else ask

        order = exchange.create_order(
            symbol=symbol,
            type="limit",
            side=side,
            amount=qty,
            price=limit_price,
            params={
                "stopLoss": f"{res['sl_price']:.4f}",
                "takeProfit": f"{res['tp_price']:.4f}",
                "timeInForce": "PostOnly",
                "positionIdx": 0,
            }
        )
        
        exec_msg = (
            f"⚡ <b>ORDER EXECUTED</b> ⚡\n"
            f"<b>Pair:</b> {symbol} | <b>Side:</b> {side.upper()}\n"
            f"<b>Price:</b> ${limit_price:.4f} | <b>SL (2x ATR):</b> ${res['sl_price']:.4f}\n"
            f"<b>TP:</b> ${res['tp_price']:.4f} | <b>Margin:</b> ${res['margin']:.2f}"
        )
        send_telegram(exec_msg)
        return True

    except Exception as e:
        print(f"Execution Failure for {symbol}: {e}", flush=True)
        return False

# ------------------------------------------------------------------
# MARKET CHECK WORKFLOW (WITH HOURLY TELEGRAM PULSE)
# ------------------------------------------------------------------
def run_market_check(exchange):
    global PEAK_EQUITY, COOLDOWN_TRACKER

    try:
        balance_resp = exchange.fetch_balance({'accountType': 'UNIFIED'})
        free_usdt = float(balance_resp.get("USDT", {}).get("free", 10.0))
    except Exception:
        free_usdt = 10.00

    if free_usdt > PEAK_EQUITY:
        PEAK_EQUITY = free_usdt

    drawdown = (PEAK_EQUITY - free_usdt) / PEAK_EQUITY if PEAK_EQUITY > 0 else 0.0
    max_concurrent_positions = 1 if drawdown >= 0.10 else MAX_CONCURRENT_POSITIONS_BASE

    active_positions = get_active_positions(exchange)
    open_symbols = [p["symbol"] for p in active_positions]
    current_position_count = len(active_positions)

    # Decrement loss cooldown timers
    for sym in list(COOLDOWN_TRACKER.keys()):
        if COOLDOWN_TRACKER[sym] > 0:
            COOLDOWN_TRACKER[sym] -= 1

    executed_trades = 0

    for symbol in TARGET_PAIRS:
        try:
            if COOLDOWN_TRACKER.get(symbol, 0) > 0:
                continue

            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=OHLCV_LIMIT)
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df = calculate_indicators(df)
            res = analyze_market_condition(symbol, df, free_usdt, max_concurrent_positions)

            if res["signal"] != "NEUTRAL":
                if symbol in open_symbols:
                    continue
                if current_position_count >= MAX_TOTAL_CRYPTO_POSITIONS:
                    continue

                if current_position_count < max_concurrent_positions:
                    success = execute_trade(exchange, symbol, res)
                    if success:
                        current_position_count += 1
                        open_symbols.append(symbol)
                        executed_trades += 1

        except Exception as e:
            print(f"Error checking symbol {symbol}: {e}", flush=True)

    # If no trade was executed this bar, send a concise heartbeat pulse to confirm operation
    if executed_trades == 0:
        pulse_msg = (
            f"⏱️ <b>Hourly Pulse:</b> Checked {len(TARGET_PAIRS)} pairs. "
            f"No entries triggered. Balance: ${free_usdt:.2f} USDT | Active Positions: {current_position_count}"
        )
        send_telegram(pulse_msg)

# ------------------------------------------------------------------
# BACKGROUND SCHEDULER & FLASK ENGINE
# ------------------------------------------------------------------
def bot_loop():
    exchange = ccxt.bybit({
        "apiKey": BYBIT_KEY,
        "secret": BYBIT_SECRET,
        "enableRateLimit": True,
        "timeout": 20000,
        "options": {"defaultType": "linear"},
    })
    
    try:
        exchange.load_markets()
    except Exception as e:
        print(f"Market loading warning: {e}", flush=True)

    while True:
        try:
            run_market_check(exchange)
        except Exception as e:
            send_critical_alert(str(e))
        
        time.sleep(3600)

app = Flask(__name__)

@app.route("/")
def health_check():
    return "OK", 200, {"Content-Type": "text/plain"}

def init_thread():
    global BOT_THREAD_STARTED
    if not BOT_THREAD_STARTED:
        BOT_THREAD_STARTED = True
        t = threading.Thread(target=bot_loop, daemon=True)
        t.start()

init_thread()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)