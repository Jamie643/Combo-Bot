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
    """Computes RSI, Bollinger Bands, ATR, ADX, Donchian Channels, and Volume MA."""
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=RSI_PERIOD).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=RSI_PERIOD).mean()
    rs = gain / (loss + 1e-10)
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

    df["vol_ma20"] = df["volume"].rolling(window=20).mean()

    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    tr_smooth = tr.rolling(window=ADX_PERIOD).sum()
    plus_di = 100 * (pd.Series(plus_dm).rolling(window=ADX_PERIOD).sum() / (tr_smooth + 1e-10))
    minus_di = 100 * (pd.Series(minus_dm).rolling(window=ADX_PERIOD).sum() / (tr_smooth + 1e-10))
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    df["adx"] = dx.rolling(window=ADX_PERIOD).mean()

    df["donchian_high"] = df["high"].shift(1).rolling(window=DONCHIAN_PERIOD).max()
    df["donchian_low"] = df["low"].shift(1).rolling(window=DONCHIAN_PERIOD).min()

    return df


def detect_market_regime(symbol, df):
    """Requires 3 consecutive bars with ADX > 22 and expanding ATR."""
    global REGIME_MEMORY

    if len(df) < 20 or "adx" not in df.columns:
        return "RANGE"

    adx_cond = df["adx"] > 22.0
    atr_expanding = df["atr"] > df["atr"].shift(1)
    raw_trending = adx_cond & atr_expanding

    recent_regime = raw_trending.tail(3)
    active_regime = "TREND" if recent_regime.all() else "RANGE"

    REGIME_MEMORY[symbol] = {"last_regime": active_regime}
    return active_regime


def scan_for_matrix_trigger(exchange):
    """Scans target pairs for LONG/SHORT Donchian breaches with 1.5x volume confirmation."""
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

            if regime == "TREND" and has_volume_confirmation:
                # LONG Breakout
                if cp >= d_high:
                    print(f"🎯 Matrix Launch Signal Triggered (LONG) on {symbol}", flush=True)
                    return {
                        "symbol": symbol,
                        "direction": "LONG",
                        "anchor_0": cp,
                        "anchor_100": cp + (d_high - d_low),
                        "atr": atr,
                    }
                # SHORT Breakout
                elif cp <= d_low:
                    print(f"🎯 Matrix Launch Signal Triggered (SHORT) on {symbol}", flush=True)
                    return {
                        "symbol": symbol,
                        "direction": "SHORT",
                        "anchor_0": cp,
                        "anchor_100": cp - (d_high - d_low),
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
            "pending_order_id": None,
            "orders_placed": {},
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
    TIER_CONFIG = {
        1: {"name": "Base Core", "leverage": 5, "atr_mult": 1.75},
        2: {"name": "Acceleration Rail", "leverage": 12, "atr_mult": 1.0},
        3: {"name": "Velocity Maximum", "leverage": 20, "atr_mult": 0.5},
        4: {"name": "De-escalation Bracket", "leverage": 10, "atr_mult": 1.0},
        5: {"name": "Terminal Run", "leverage": 5, "atr_mult": 1.5},
    }

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

    def fetch_available_usdt(self):
        try:
            balance = self.exchange.fetch_balance({"accountType": "UNIFIED"})
            return float(balance.get("USDT", {}).get("free", 0.0))
        except Exception as e:
            print(f"Failed to fetch balance: {e}", flush=True)
            return 0.0

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

    def calculate_atr_stop_loss(self, entry_price, current_atr, tier, direction):
        mult = self.TIER_CONFIG[tier]["atr_mult"]
        offset = current_atr * mult
        if direction.upper() == "SHORT":
            return round(entry_price + offset, 4)
        return round(entry_price - offset, 4)

    def get_tier_config(self, tier):
        """Maps parameters dynamically using 35% Smart Margin Model."""
        atr = self.state.get("atr", 0.0)
        direction = self.state.get("direction", "LONG")
        free_usdt = self.fetch_available_usdt()
        margin_allocated = free_usdt * 0.35

        fib_mapping = {
            1: ("0.0", "23.6"),
            2: ("23.6", "38.2"),
            3: ("38.2", "50.0"),
            4: ("50.0", "61.8"),
            5: ("61.8", "78.6"),
        }

        if tier not in fib_mapping:
            return None

        entry_key, tp_key = fib_mapping[tier]
        entry_price = self.fibs[entry_key]
        tp_price = self.fibs[tp_key]

        # Stop Loss Logic
        if tier == 2:
            sl_price = self.fibs["0.0"]  # Tier 1 Entry Trail-Lock
        elif tier == 4:
            sl_price = self.fibs["38.2"] # Tier 3 Entry Lock
        else:
            sl_price = self.calculate_atr_stop_loss(entry_price, atr, tier, direction)

        return {
            "entry_price": entry_price,
            "trigger_price": entry_price,
            "tp_price": tp_price,
            "sl_price": sl_price,
            "leverage": self.TIER_CONFIG[tier]["leverage"],
            "margin": margin_allocated,
        }

    def initialize_campaign(self, launch_params):
        self.symbol = launch_params["symbol"]
        self.state["symbol"] = launch_params["symbol"]
        self.state["direction"] = launch_params["direction"]
        self.state["anchor_0"] = launch_params["anchor_0"]
        self.state["anchor_100"] = launch_params["anchor_100"]
        self.state["atr"] = launch_params["atr"]
        self.state["status"] = "RUNNING"
        self.state["orders_placed"] = {}

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
            self.state["status"] = "RUNNING"
            self.state["pending_order_id"] = order["id"]
            self.state["orders_placed"]["1"] = True
            StateManager.save_state(self.state)

            send_telegram(
                f"🚀 <b>CASCADING TIER 1 EXECUTED</b>\n"
                f"<b>Pair:</b> {self.symbol} | <b>Price:</b> ${formatted_price}\n"
                f"<b>TP:</b> ${formatted_tp} | <b>SL:</b> ${formatted_sl}\n"
                f"<b>Smart Margin:</b> ${margin_to_use:.2f} USDT | <b>Leverage:</b> {config['leverage']}x"
            )
        except Exception as e:
            send_critical_alert(f"Failed to execute Tier 1: {e}")

    def queue_conditional_tier(self, tier):
        str_tier = str(tier)
        if self.state.get("orders_placed", {}).get(str_tier):
            return

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
            self.state["status"] = "PENDING_PULLBACK"
            self.state["pending_order_id"] = order["id"]
            self.state["orders_placed"][str_tier] = True
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
        next_tier = active_tier + 1
        direction = self.state["direction"]

        has_pending_order = len(self.exchange.fetch_open_orders(self.symbol)) > 0

        next_config = self.get_tier_config(next_tier)
        if next_config:
            next_trigger = next_config["entry_price"]
            surged_past = (current_price >= next_trigger) if direction == "LONG" else (current_price <= next_trigger)

            # --- GUARDRAIL A: Stale Order Omission Threshold ---
            if has_pending_order and surged_past:
                self.cancel_all_conditional_orders()
                send_telegram(
                    f"⚡ <b>Guardrail A Activated</b>: Price surged past Tier {next_tier} target (${next_trigger}). "
                    f"Skipping stale Tier {active_tier} order."
                )
                self.state["active_tier"] = next_tier
                StateManager.save_state(self.state)
                self.queue_conditional_tier(next_tier)
                return

            # --- TIER ADVANCE LOGIC ---
            if surged_past and not self.state.get("orders_placed", {}).get(str(next_tier)):
                if next_tier == 2:
                    t1_entry = self.fibs["0.0"]
                    self.exchange.edit_position_trading_stop(self.symbol, stopLoss=t1_entry)
                    send_telegram(f"🔒 <b>Tier 2 Activated</b>: Position Stop Loss trail-locked to Tier 1 Entry (${t1_entry}).")

                self.state["active_tier"] = next_tier
                StateManager.save_state(self.state)
                self.queue_conditional_tier(next_tier)

        # --- TIER 3 HYPER-TIGHT 0.5x ATR TRAILING STOP ---
        if active_tier == 3:
            try:
                ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe=TIMEFRAME, limit=30)
                df = calculate_indicators(pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]))
                current_atr = df["atr"].iloc[-1]
                tight_sl = self.calculate_atr_stop_loss(current_price, current_atr, tier=3, direction=direction)
                self.exchange.edit_position_trading_stop(self.symbol, stopLoss=tight_sl)
            except Exception as e:
                print(f"Tier 3 Trailing Stop update failed: {e}", flush=True)

        # Check for campaign completion
        positions = self.exchange.fetch_positions([self.symbol], params={"category": "linear"})
        has_open_position = any(float(p.get("contracts", 0) or p.get("size", 0)) > 0 for p in positions)

        if not has_open_position and not has_pending_order and active_tier >= 5:
            self.state["status"] = "COMPLETED"
            StateManager.save_state(self.state)
            send_telegram("🎉 <b>5-TIER CASCADING MATRIX CAMPAIGN COMPLETED!</b>")


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
