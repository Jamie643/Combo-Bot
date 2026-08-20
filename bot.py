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
    "TON/USDT:USDT",
    "XRP/USDT:USDT",
    "ADA/USDT:USDT",
]

TIMEFRAME = "1h"
OHLCV_LIMIT = 100

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
ATR_SL_MULTIPLIER = 1.5    # SL = 1.5 * ATR
RISK_REWARD_RATIO = 2.0    # TP = 2.0 * SL distance (1:2 Risk/Reward)
LEVERAGE = 5               # Default Leverage

# Mainnet Bybit API Credentials
BYBIT_KEY = os.getenv("BYBIT_API_KEY") or os.getenv("BYBIT_KEY")
BYBIT_SECRET = os.getenv("BYBIT_API_SECRET") or os.getenv("BYBIT_SECRET")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# ------------------------------------------------------------------
# TELEGRAM NOTIFIER
# ------------------------------------------------------------------
def send_telegram(message_text):
    """Sends a Telegram message formatted in HTML."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram credentials missing. Printing output to console:", flush=True)
        print(message_text, flush=True)
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
    """Sends an immediate error notification if execution fails."""
    alert_text = (
        "🚨 <b>CRITICAL BOT ERROR</b> 🚨\n\n"
        f"<b>Error Details:</b>\n<code>{error_msg}</code>\n\n"
        "⚠️ <i>The Render market check encountered a failure.</i>"
    )
    send_telegram(alert_text)

# ------------------------------------------------------------------
# TECHNICAL INDICATORS
# ------------------------------------------------------------------
def calculate_indicators(df):
    """Calculates RSI, ADX, Bollinger Bands, ATR, Donchian Channels."""
    
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

# ------------------------------------------------------------------
# STRATEGY & RISK ENGINE
# ------------------------------------------------------------------
def analyze_market_condition(df, balance):
    cp = df["close"].iloc[-1]
    rsi = df["rsi"].iloc[-1]
    adx = df["adx"].iloc[-1]
    bb_upper = df["bb_upper"].iloc[-1]
    bb_lower = df["bb_lower"].iloc[-1]
    d_high = df["donchian_high"].iloc[-1]
    d_low = df["donchian_low"].iloc[-1]
    atr = df["atr"].iloc[-1]

    move_range = d_high - d_low
    fib_618_bull = d_high - (move_range * FIB_LEVEL)

    is_strong_trend = adx > 25
    is_donchian_breakout = cp >= d_high
    is_fib_retest = abs(cp - fib_618_bull) / cp <= FIB_TOLERANCE
    is_oversold = rsi < 30 or cp <= bb_lower
    is_overbought = rsi > 70 or cp >= bb_upper

    signal = "NEUTRAL"
    bias = "No specific entry setup."

    if is_donchian_breakout and is_strong_trend:
        signal = "🚀 LONG BREAKOUT"
        bias = f"Donchian High Break ({d_high:.4f}), ADX={adx:.1f}"
    elif is_fib_retest and is_strong_trend:
        signal = "🟢 BULLISH FIB RETEST"
        bias = f"Testing 61.8% Fib ({fib_618_bull:.4f})"
    elif is_oversold:
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

    risk_amount = balance * RISK_PER_TRADE
    units = risk_amount / sl_dist if sl_dist > 0 else 0
    notional_value = units * cp
    required_margin = notional_value / LEVERAGE

    return {
        "price": cp,
        "signal": signal,
        "bias": bias,
        "sl_price": sl_price,
        "tp_price": tp_price,
        "units": units,
        "margin": required_margin,
        "leverage": LEVERAGE,
    }

# ------------------------------------------------------------------
# MARKET CHECK WORKFLOW
# ------------------------------------------------------------------
def run_market_check(exchange):
    try:
        balance_resp = exchange.fetch_balance({'accountType': 'UNIFIED'})
        free_usdt = float(balance_resp.get("USDT", {}).get("free", 1000.0))
    except Exception:
        try:
            balance_resp = exchange.fetch_balance()
            free_usdt = float(balance_resp.get("USDT", {}).get("free", 1000.0))
        except Exception as e:
            print(f"Warning: Fetch balance failed ({e}). Defaulting to $1,000 baseline.", flush=True)
            free_usdt = 1000.0

    report = [
        "<b>🔄 Render Web Service Analysis Report (Mainnet)</b>",
        f"<b>Account Equity:</b> ${free_usdt:,.2f} USDT",
        "-----------------------------------------",
    ]

    trade_candidates = []

    for symbol in TARGET_PAIRS:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=OHLCV_LIMIT)
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df = calculate_indicators(df)
            res = analyze_market_condition(df, free_usdt)

            pair_name = symbol.split(":")[0]
            report.append(f"<b>{pair_name}</b> | ${res['price']:,.4f} | <code>{res['signal']}</code>")
            report.append(f"├ <i>Status: {res['bias']}</i>")

            if res["signal"] != "NEUTRAL":
                report.append(
                    f"└ <b>Trade Prep ({res['leverage']}x):</b> Margin: ${res['margin']:.2f} | "
                    f"SL: ${res['sl_price']:.4f} | TP: ${res['tp_price']:.4f}"
                )
                trade_candidates.append((pair_name, res["signal"], res["margin"]))
            else:
                report.append("└ <i>No action required</i>")
            
            report.append("")

        except Exception as e:
            report.append(f"<b>{symbol}</b>: ❌ Error reading pair: {str(e)}\n")

    report.append("<b>📋 Summary:</b>")
    if trade_candidates:
        report.append(f"Found <b>{len(trade_candidates)}</b> trade candidate(s):")
        for pair, sig, margin in trade_candidates:
            report.append(f"• <b>{pair}</b> → {sig} (Req. Margin: ${margin:.2f})")
    else:
        report.append("All pairs neutral. No trade setups met criteria.")

    send_telegram("\n".join(report))

# ------------------------------------------------------------------
# BACKGROUND SCHEDULER LOOP
# ------------------------------------------------------------------
def bot_loop():
    print("Initializing Bybit Mainnet CCXT instance...", flush=True)
    exchange = ccxt.bybit({
        "apiKey": BYBIT_KEY,
        "secret": BYBIT_SECRET,
        "enableRateLimit": True,
        "timeout": 20000,
        "options": {
            "defaultType": "linear",
        },
    })
    # Mainnet mode: sandbox setting removed

    while True:
        try:
            print("Executing hourly scan...", flush=True)
            run_market_check(exchange)
            print("Scan finished. Sleeping for 1 hour...", flush=True)
        except Exception as e:
            error_details = traceback.format_exc()
            print(f"Error in execution loop:\n{error_details}", flush=True)
            send_critical_alert(str(e))
        
        time.sleep(3600)

# ------------------------------------------------------------------
# FLASK WEB SERVER & MODULE-LEVEL THREAD INITIALIZATION
# ------------------------------------------------------------------
app = Flask(__name__)

@app.route("/")
def health_check():
    return "Bot is alive and running on Mainnet!", 200

def start_bot_thread():
    print("Starting background market scanner thread...", flush=True)
    t = threading.Thread(target=bot_loop, daemon=True)
    t.start()

# Launch thread at module import level so Gunicorn executes it on boot
start_bot_thread()

if __name__ == "__main__":
    # Fallback runner for local development
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)