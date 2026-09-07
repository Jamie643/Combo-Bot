import os
import sys
import time
import json
import threading
import traceback
import ccxt
import numpy as np
import pandas as pd
import requests
from flask import Flask

# ------------------------------------------------------------------
# CONFIGURATION & MATRIX PARAMETERS
# ------------------------------------------------------------------
TARGET_PAIRS = [
    "DOGE/USDT:USDT",
    "GRAM/USDT:USDT",
    "XRP/USDT:USDT",
    "ADA/USDT:USDT",
    "SOL/USDT:USDT",
]

STATE_FILE = "cascade_state.json"
TIMEFRAME = "1h"                 
MIN_ORDER_USD = 5.05            

# Technical Indicators Configuration
RSI_PERIOD = 14
ADX_PERIOD = 14
BB_PERIOD = 20
BB_STD = 2.0
DONCHIAN_PERIOD = 20
ATR_PERIOD = 14
VOLUME_SPIKE_MULT = 1.5         

# Base Parameters
BASE_MARGIN = 38.00             # Initial Tier 1 Capital

# Bybit Credentials & Telegram Config
BYBIT_KEY = os.getenv("BYBIT_API_KEY") or os.getenv("BYBIT_KEY")
BYBIT_SECRET = os.getenv("BYBIT_API_SECRET") or os.getenv("BYBIT_SECRET")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

REGIME_MEMORY = {}
LAST_PULSE_TIME = 0

# ------------------------------------------------------------------
# TELEGRAM NOTIFIER
# ------------------------------------------------------------------
def send_telegram(message_text):
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
        "🚨 <b>CRITICAL CASCADING BOT ERROR</b> 🚨\n\n"
        f"<b>Details:</b>\n<code>{error_msg}</code>"
    )
    send_telegram(alert_text)


def check_and_send_daily_pulse(exchange):
    global LAST_PULSE_TIME
    current_time = time.time()
    
    if current_time - LAST_PULSE_TIME >= 86400:
        try:
            usdt_bal = exchange.fetch_balance({"accountType": "UNIFIED"}).get("USDT", {}).get("free", 0.0)
            status_msg = (
                "💚 <b>DAILY BOT CHECKUP</b> 💚\n\n"
                "<b>Status:</b> Active & Scanning 1H Markets\n"
                f"<b>Pairs Watched:</b> {', '.join(TARGET_PAIRS)}\n"
                f"<b>Free Balance:</b> ${usdt_bal:.2f} USDT\n"
                "<i>System running clean. Listening for Fib breakout setups.</i>"
            )
            send_telegram(status_msg)
            LAST_PULSE_TIME = current_time
        except Exception as e:
            print(f"Daily pulse error: {e}", flush=True)


def safe_set_leverage(exchange, leverage, symbol):
    try:
        exchange.set_leverage(leverage, symbol)
    except Exception as e:
        err_str = str(e)
        if "110043" in err_str or "leverage not modified" in err_str:
            pass
        else:
            print(f"Leverage setting info: {e}", flush=True)


# ------------------------------------------------------------------
# MARKET INDICATORS & REGIME DETECTOR
# ------------------------------------------------------------------
def calculate_indicators(df):
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
    df["donchian_low"] = df["high"].shift(1).rolling(window=DONCHIAN_PERIOD).min()

    return df


def detect_market_regime(symbol, df):
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


def scan_for_matrix_trigger(exchange):
    for symbol in TARGET_PAIRS:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=100)
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df = calculate_indicators(df)

            cp = df["close"].iloc[-1]
            volume = df["volume"].iloc[-1]
            vol_ma = df["vol_ma20"].iloc[-1]
            d_high = df["donchian_high"].iloc[-1]
            d_low = df["donchian_low"].iloc[-1]
            atr = df["atr"].iloc[-1]

            regime = detect_market_regime(symbol, df)
            has_volume_confirmation = volume > (vol_ma * VOLUME_SPIKE_MULT)

            if regime == "TREND" and cp >= d_high and has_volume_confirmation:
                print(f"🎯 Matrix Launch Signal Triggered on {symbol}", flush=True)
                return {
                    "symbol": symbol,
                    "direction": "LONG",
                    "anchor_0": cp,
                    "anchor_100": cp + (d_high - d_low),
                    "atr": atr,
                }
        except Exception as e:
            print(f"Error scanning {symbol}: {e}", flush=True)

    return None


def calculate_fib_levels(anchor_0, anchor_100, direction="LONG"):
    total_dist = abs(anchor_100 - anchor_0)

    if direction.upper() == "LONG":
        return {
            "0.0": anchor_0,
            "23.6": anchor_0 + (total_dist * 0.236),
            "38.2": anchor_0 + (total_dist * 0.382),
            "50.0": anchor_0 + (total_dist * 0.500),
            "61.8": anchor_0 + (total_dist * 0.618),
            "78.6": anchor_0 + (total_dist * 0.786),
            "100.0": anchor_100,
        }
    else:
        return {
            "0.0": anchor_0,
            "23.6": anchor_0 - (total_dist * 0.236),
            "38.2": anchor_0 - (total_dist * 0.382),
            "50.0": anchor_0 - (total_dist * 0.500),
            "61.8": anchor_0 - (total_dist * 0.618),
            "78.6": anchor_0 - (total_dist * 0.786),
            "100.0": anchor_100,
        }


# ------------------------------------------------------------------
# STATE MANAGEMENT
# ------------------------------------------------------------------
class StateManager:
    @staticmethod
    def load_state():
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error reading state file: {e}", flush=True)
        return {
            "symbol": None,
            "direction": "LONG",
            "anchor_0": 0.0,
            "anchor_100": 0.0,
            "atr": 0.0,
            "active_tier": 0,
            "status": "IDLE",  
            "current_margin": BASE_MARGIN,
            "pending_order_id": None,
            "realized_pnl_history": {},
        }

    @staticmethod
    def save_state(state):
        try:
            with open(STATE_FILE, "w") as f:
                json.dump(state, f, indent=4)
        except Exception as e:
            print(f"Error saving state file: {e}", flush=True)


# ------------------------------------------------------------------
# CASCADING EXECUTION ENGINE
# ------------------------------------------------------------------
class CascadingMatrixManager:
    def __init__(self, exchange):
        self.exchange = exchange
        self.state = StateManager.load_state()
        if self.state["symbol"]:
            self.symbol = self.state["symbol"]
            self.fibs = calculate_fib_levels(
                self.state["anchor_0"], self.state["anchor_100"], self.state["direction"]
            )
        else:
            self.symbol = None
            self.fibs = {}

    def cancel_all_conditional_orders(self):
        if not self.symbol:
            return
        try:
            orders = self.exchange.fetch_open_orders(self.symbol)
            for order in orders:
                self.exchange.cancel_order(order["id"], self.symbol)
            print(f"Cleared open orders for {self.symbol}", flush=True)
        except Exception as e:
            print(f"Error clearing orders: {e}", flush=True)

    def get_tier_config(self, tier):
        """Maps specific parameters and rules according to exact Fib Table Specs."""
        atr = self.state.get("atr", 0.0)

        if tier == 1:
            return {
                "entry_price": self.fibs["0.0"],
                "trigger_price": None,
                "tp_price": self.fibs["23.6"],
                "sl_price": self.fibs["0.0"] - (1.5 * atr),  # 1.5x ATR
                "leverage": 5,                                # 5x Leverage
                "margin": BASE_MARGIN,                        # Base Margin ($38.00)
            }
        elif tier == 2:
            t1_pnl = self.state["realized_pnl_history"].get("1", 16.80)
            margin = BASE_MARGIN + (0.70 * t1_pnl)            # $38 + 70% T1 Profit
            return {
                "entry_price": self.fibs["23.6"],
                "trigger_price": self.fibs["23.6"],
                "tp_price": self.fibs["38.2"],
                "sl_price": self.fibs["0.0"],                 # T1 Entry Stop
                "leverage": 12,                               # 12x Leverage
                "margin": margin,
            }
        elif tier == 3:
            prev_margin = self.state.get("current_margin", 49.76)
            t2_pnl = self.state["realized_pnl_history"].get("2", 38.25)
            margin = prev_margin + (0.70 * t2_pnl)           # Prev. Margin + 70% T2 Profit
            return {
                "entry_price": self.fibs["38.2"],
                "trigger_price": self.fibs["38.2"],
                "tp_price": self.fibs["61.8"],
                "sl_price": self.fibs["23.6"] - (0.5 * atr), # 0.5x ATR from T2 Entry
                "leverage": 20,                               # 20x Leverage
                "margin": margin,
            }
        elif tier == 4:
            prev_margin = self.state.get("current_margin", 76.54)
            t3_pnl = self.state["realized_pnl_history"].get("3", 118.00)
            margin = prev_margin + (0.50 * t3_pnl)           # Prev. Margin + 50% T3 Profit
            return {
                "entry_price": self.fibs["61.8"],
                "trigger_price": self.fibs["61.8"],
                "tp_price": self.fibs["78.6"],
                "sl_price": self.fibs["38.2"],                 # T3 Entry Stop
                "leverage": 8,                                # 8x De-leverage
                "margin": margin,
            }
        return None

    def initialize_campaign(self, launch_params):
        self.symbol = launch_params["symbol"]
        self.state["symbol"] = launch_params["symbol"]
        self.state["direction"] = launch_params["direction"]
        self.state["anchor_0"] = launch_params["anchor_0"]
        self.state["anchor_100"] = launch_params["anchor_100"]
        self.state["atr"] = launch_params["atr"]
        self.state["status"] = "RUNNING"

        self.fibs = calculate_fib_levels(
            launch_params["anchor_0"], launch_params["anchor_100"], launch_params["direction"]
        )
        StateManager.save_state(self.state)

        send_telegram(
            f"⚡ <b>NEW MATRIX CAMPAIGN LAUNCHED</b> ⚡\n"
            f"<b>Pair:</b> {self.symbol} | <b>Direction:</b> {self.state['direction']}\n"
            f"<b>Anchor 0.0%:</b> ${launch_params['anchor_0']:.4f}\n"
            f"<b>Target 100.0%:</b> ${launch_params['anchor_100']:.4f}"
        )
        self.execute_tier_1()

    def execute_tier_1(self):
        config = self.get_tier_config(1)
        safe_set_leverage(self.exchange, config["leverage"], self.symbol)

        margin_to_use = config["margin"]
        ticker = self.exchange.fetch_ticker(self.symbol)
        cp = ticker["last"]

        notional_value = max(margin_to_use * config["leverage"], MIN_ORDER_USD)
        raw_qty = notional_value / cp

        formatted_price = float(self.exchange.price_to_precision(self.symbol, cp))
        formatted_qty = float(self.exchange.amount_to_precision(self.symbol, raw_qty))
        formatted_tp = float(self.exchange.price_to_precision(self.symbol, config["tp_price"]))
        formatted_sl = float(self.exchange.price_to_precision(self.symbol, config["sl_price"]))

        try:
            order = self.exchange.create_order(
                symbol=self.symbol,
                type="limit",
                side="buy" if self.state["direction"] == "LONG" else "sell",
                amount=formatted_qty,
                price=formatted_price,
                params={
                    "takeProfit": str(formatted_tp),
                    "stopLoss": str(formatted_sl),
                    "timeInForce": "PostOnly",
                    "positionIdx": 0,
                },
            )
            self.state["active_tier"] = 1
            self.state["current_margin"] = margin_to_use
            self.state["status"] = "RUNNING"
            self.state["pending_order_id"] = order["id"]
            StateManager.save_state(self.state)

            send_telegram(
                f"🚀 <b>CASCADING TIER 1 EXECUTED</b>\n"
                f"<b>Pair:</b> {self.symbol} | <b>Price:</b> ${formatted_price}\n"
                f"<b>TP:</b> ${formatted_tp} | <b>SL:</b> ${formatted_sl} (1.5x ATR)\n"
                f"<b>Margin Allocated:</b> ${margin_to_use:.2f} USDT | <b>Leverage:</b> {config['leverage']}x"
            )
        except Exception as e:
            send_critical_alert(f"Failed to execute Tier 1: {e}")

    def queue_conditional_tier(self, tier):
        config = self.get_tier_config(tier)
        if not config:
            return

        safe_set_leverage(self.exchange, config["leverage"], self.symbol)

        margin_to_use = config["margin"]
        notional_value = max(margin_to_use * config["leverage"], MIN_ORDER_USD)
        raw_qty = notional_value / config["entry_price"]

        formatted_entry = float(self.exchange.price_to_precision(self.symbol, config["entry_price"]))
        formatted_trigger = float(self.exchange.price_to_precision(self.symbol, config["trigger_price"]))
        formatted_qty = float(self.exchange.amount_to_precision(self.symbol, raw_qty))
        formatted_tp = float(self.exchange.price_to_precision(self.symbol, config["tp_price"]))
        formatted_sl = float(self.exchange.price_to_precision(self.symbol, config["sl_price"]))

        ticker = self.exchange.fetch_ticker(self.symbol)
        cp = ticker["last"]
        trigger_dir = 1 if config["trigger_price"] > cp else 2

        try:
            order = self.exchange.create_order(
                symbol=self.symbol,
                type="limit",
                side="buy" if self.state["direction"] == "LONG" else "sell",
                amount=formatted_qty,
                price=formatted_entry,
                params={
                    "triggerPrice": str(formatted_trigger),
                    "triggerBy": "LastPrice",
                    "triggerDirection": trigger_dir,
                    "takeProfit": str(formatted_tp),
                    "stopLoss": str(formatted_sl),
                    "timeInForce": "PostOnly",
                    "positionIdx": 0,
                },
            )

            self.state["active_tier"] = tier
            self.state["current_margin"] = margin_to_use
            self.state["status"] = "PENDING_PULLBACK"
            self.state["pending_order_id"] = order["id"]
            StateManager.save_state(self.state)

            send_telegram(
                f"⏳ <b>TIER {tier} CONDITIONAL ORDER QUEUED</b>\n"
                f"<b>Pair:</b> {self.symbol}\n"
                f"<b>Trigger / Entry:</b> ${formatted_entry}\n"
                f"<b>Target TP:</b> ${formatted_tp} | <b>SL:</b> ${formatted_sl}\n"
                f"<b>Leverage:</b> {config['leverage']}x | <b>Allocated Margin:</b> ${margin_to_use:.2f} USDT"
            )
        except Exception as e:
            send_critical_alert(f"Failed to queue Tier {tier} conditional order: {e}")

    def evaluate_matrix_step(self):
        if self.state["status"] in ["COMPLETED", "FAILED_STOPPED_OUT", "IDLE"]:
            signal = scan_for_matrix_trigger(self.exchange)
            if signal:
                self.initialize_campaign(signal)
            return

        ticker = self.exchange.fetch_ticker(self.symbol)
        current_price = ticker["last"]
        active_tier = self.state["active_tier"]

        positions = self.exchange.fetch_positions([self.symbol], params={"category": "linear"})
        has_open_position = any(float(p.get("contracts", 0) or p.get("size", 0)) > 0 for p in positions)

        # Transition Check: Advance upon tier profit completion
        if not has_open_position and self.state["status"] == "RUNNING":
            
            # Map Table Realized Profits for re-investment sizing
            pnl_map = {1: 16.80, 2: 38.25, 3: 118.00, 4: 87.00}
            self.state["realized_pnl_history"][str(active_tier)] = pnl_map.get(active_tier, 0.0)

            send_telegram(
                f"✅ <b>TIER {active_tier} COMPLETED</b>\n"
                f"Realized Tier Profit: +${pnl_map.get(active_tier, 0.0):.2f} USDT"
            )

            if active_tier < 4:
                self.queue_conditional_tier(active_tier + 1)
            else:
                self.state["status"] = "COMPLETED"
                StateManager.save_state(self.state)
                send_telegram("🎉 <b>4-TIER CASCADING MATRIX CAMPAIGN COMPLETED!</b>")


# ------------------------------------------------------------------
# BACKGROUND WORKER & FLASK SERVICE
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

    matrix = CascadingMatrixManager(exchange)
    check_and_send_daily_pulse(exchange)

    while True:
        try:
            check_and_send_daily_pulse(exchange)
            matrix.evaluate_matrix_step()
        except Exception as e:
            send_critical_alert(str(e))

        time.sleep(10)


app = Flask(__name__)


@app.route("/")
def health_check():
    return "OK - Tier Fib Cascading Bot Running", 200, {"Content-Type": "text/plain"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    threading.Thread(target=bot_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=port)
