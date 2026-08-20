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
# Target pairs using standard CCXT Bybit linear symbol format
TARGET_PAIRS = [
    "DOGE/USDT",
    "GRAM/USDT",
    "XRP/USDT",
    "ADA/USDT",
]

TIMEFRAME = "1h"
OHLCV_LIMIT = 100
MAX_CONCURRENT_POSITIONS = 2  # Stage 1 Risk Gatekeeper

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
# POSITION HELPERS & EXECUTION ENGINE
# ------------------------------------------------------------------
def get_active_positions(exchange):
    """Fetches currently open linear perpetual positions."""
    try:
        positions = exchange.fetch_positions(params={"category": "linear"})
        active = []
        for p in positions:
            contracts = float(p.get("contracts", 0) or p.get("size", 0))
            if contracts > 0:
                active.append(p)
        return active
    except Exception as e:
        print(f"Error fetching active positions: {e}", flush=True)
        return []

def execute_trade(exchange, symbol, res):
    """Executes market entry with attached Stop Loss & Take Profit."""
    side = "buy" if "LONG" in res["signal"] else "sell"
    qty = res["units"]

    if qty <= 0:
        print(f"Aborting execution for {symbol}: Position size units calculate to zero.", flush=True)
        return False

    try:
        # Set market leverage prior to entry
        try:
            exchange.set_leverage(LEVERAGE, symbol)
        except Exception as lev_err:
            print(f"Leverage setup notice for {symbol}: {lev_err}", flush=True)

        # Place Market Order with TP/SL attached
        order = exchange.create_order(
            symbol=symbol,
            type="market",
            side=side,
            amount=qty,
            params={
                "stopLoss": f"{res['sl_price']:.4f}",
                "takeProfit": f"{res['tp_price']:.4f}",
                "positionIdx": 0,  # One-way Mode on Bybit
            }
        )
        
        exec_msg = (
            f"⚡ <b>LIVE ORDER EXECUTED</b> ⚡\n\n"
            f"<b>Pair:</b> {symbol}\n"
            f"<b>Side:</b> {side.upper()}\n"
            f"<b>Units:</b> {qty:.2f}\n"
            f"<b>Entry Price:</b> ~${res['price']:.4f}\n"
            f"<b>Stop Loss:</b> ${res['sl_price']:.4f}\n"
            f"<b>Take Profit:</b> ${res['tp_price']:.4f}\n"
            f"<b>Leverage:</b> {LEVERAGE}x"
        )
        send_telegram(exec_msg)
        return True

    except Exception as e:
        error_msg = f"Failed to execute trade for {symbol}: {str(e)}"
        print(error_msg, flush=True)
        send_telegram(f"❌ <b>EXECUTION FAILURE ({symbol})</b>\n<code>{str(e)}</code>")
        return False

# ------------------------------------------------------------------
# MARKET CHECK WORKFLOW
# ------------------------------------------------------------------
def run_market_check(exchange):
    try:
        balance_resp = exchange.fetch_balance({'accountType': 'UNIFIED'})
        free_usdt = float(balance_resp.get("USDT", {}).get("free", 10.0))
    except Exception:
        try:
            balance_resp = exchange.fetch_balance()
            free_usdt = float(balance_resp.get("USDT", {}).get("free", 10.0))
        except Exception as e:
            print(f"Warning: Fetch balance failed ({e}). Defaulting to $10.00 baseline.", flush=True)
            free_usdt = 10.00

    active_positions = get_active_positions(exchange)
    open_symbols = [p["symbol"] for p in active_positions]
    current_position_count = len(active_positions)

    report = [
        "<b>🔄 Render Web Service Analysis Report (Mainnet)</b>",
        f"<b>Account Equity:</b> ${free_usdt:,.2f} USDT",
        f"<b>Active Positions:</b> {current_position_count}/{MAX_CONCURRENT_POSITIONS}",
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
                trade_candidates.append((symbol, res))
                
                if symbol in open_symbols:
                    report.append("└ ⚠️ <i>Signal active, but position already open for this pair.</i>")
                elif current_position_count >= MAX_CONCURRENT_POSITIONS:
                    report.append("└ 🛑 <i>Signal active, but max position cap (2) reached.</i>")
                else:
                    report.append(
                        f"└ <b>Executing ({res['leverage']}x):</b> Margin: ${res['margin']:.2f} | "
                        f"SL: ${res['sl_price']:.4f} | TP: ${res['tp_price']:.4f}"
                    )
            else:
                report.append("└ <i>No action required</i>")
            
            report.append("")

        except Exception as e:
            report.append(f"<b>{symbol}</b>: ❌ Error reading pair: {str(e)}\n")

    # Order Execution Processing Block
    report.append("<b>📋 Summary & Execution:</b>")
    if trade_candidates:
        report.append(f"Found <b>{len(trade_candidates)}</b> trade candidate(s).")
        
        for sym, res in trade_candidates:
            if sym in open_symbols:
                continue
                
            if current_position_count < MAX_CONCURRENT_POSITIONS:
                print(f"Executing trade setup for {sym}...", flush=True)
                success = execute_trade(exchange, sym, res)
                if success:
                    current_position_count += 1
                    open_symbols.append(sym)
            else:
                report.append(f"• <b>{sym}</b> skipped: Max open position limit reached ({MAX_CONCURRENT_POSITIONS}).")
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
    
    try:
        exchange.load_markets()
    except Exception as e:
        print(f"Warning: Failed to pre-load markets: {e}", flush=True)

    while True:
        try:
            print("Executing hourly scan and trade evaluation...", flush=True)
            run_market_check(exchange)
            print("Scan & execution cycle finished. Sleeping for 1 hour...", flush=True)
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

start_bot_thread()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)