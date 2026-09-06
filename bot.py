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
# CONFIGURATION & HYBRID PARAMETERS
# ------------------------------------------------------------------
TARGET_PAIRS = [
    "DOGE/USDT:USDT",
    "GRAM/USDT:USDT",
    "XRP/USDT:USDT",
    "ADA/USDT:USDT",
    "SOL/USDT:USDT",
]

STATE_FILE = "cascade_state.json"
TIMEFRAME = "1h"                 # Timeframe for market regime scanning
MAKER_FEE_RATE = 0.0002         # Limit fee target
MIN_ORDER_USD = 5.05            # Bybit minimum order floor

# Market Analysis Technical Indicators Configuration
RSI_PERIOD = 14
ADX_PERIOD = 14
BB_PERIOD = 20
BB_STD = 2.0
DONCHIAN_PERIOD = 20
ATR_PERIOD = 14
VOLUME_SPIKE_MULT = 1.5         # 150% average volume required for breakout confirmation

# Risk & Smart Leverage Configuration
INITIAL_CORE_MARGIN = 5.00      # USDT allocation for Tier 1
LEVERAGE_BASELINE = 15          # 15x Baseline Leverage
LEVERAGE_HYPER = 30             # 30x Hyper Leverage

# Bybit Credentials & Telegram Config
BYBIT_KEY = os.getenv("BYBIT_API_KEY") or os.getenv("BYBIT_KEY")
BYBIT_SECRET = os.getenv("BYBIT_API_SECRET") or os.getenv("BYBIT_SECRET")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

BOT_THREAD_STARTED = False
REGIME_MEMORY = {}

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
        "🚨 <b>CRITICAL CASCADING BOT ERROR</b> 🚨\n\n"
        f"<b>Details:</b>\n<code>{error_msg}</code>"
    )
    send_telegram(alert_text)


# ------------------------------------------------------------------
# LEVERAGE HELPER (SAFELY CATCHES BYBIT CODE 110043)
# ------------------------------------------------------------------
def safe_set_leverage(exchange, leverage, symbol):
    """Sets leverage on Bybit while ignoring retCode 110043 (leverage not modified)."""
    try:
        exchange.set_leverage(leverage, symbol)
    except Exception as e:
        err_str = str(e)
        if "110043" in err_str or "leverage not modified" in err_str:
            pass
        else:
            print(f"Leverage setting info: {e}", flush=True)


# ------------------------------------------------------------------
# CORE MARKET ANALYSIS & REGIME DETECTOR ENGINE
# ------------------------------------------------------------------
def calculate_indicators(df):
    """Calculates RSI, BB, ATR, ADX, Donchian Channels, and Volume Averages."""
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
    df["donchian_low"] = df["low"].shift(1).rolling(window=DONCHIAN_PERIOD).min()

    return df


def detect_market_regime(symbol, df):
    """Requires 3 consecutive bars to confirm regime shift."""
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
    """Scans watched pairs for confirmed trend/volume breakouts to auto-anchor matrix campaigns."""
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

            regime = detect_market_regime(symbol, df)
            has_volume_ confirmation = volume > (vol_ma * VOLUME_SPIKE_MULT)

            if regime == "TREND" and cp >= d_high and has_volume_confirmation:
                print(f"🎯 Matrix Launch Signal Triggered on {symbol}", flush=True)
                return {
                    "symbol": symbol,
                    "direction": "LONG",
                    "anchor_0": cp,
                    "anchor_100": cp + (d_high - d_low),
                }
        except Exception as e:
            print(f"Error scanning {symbol}: {e}", flush=True)

    return None


# ------------------------------------------------------------------
# MATHEMATICAL CORE ENGINE (FIBONACCI MATRIX)
# ------------------------------------------------------------------
def calculate_fib_levels(anchor_0, anchor_100, direction="LONG"):
    """Calculates standard 5-tier Fibonacci extension/retracement levels."""
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
# STATE PERSISTENCE MANAGER
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
            "active_tier": 0,
            "status": "IDLE",  # IDLE, RUNNING, COMPLETED, FAILED_STOPPED_OUT
            "current_balance": INITIAL_CORE_MARGIN,
            "pending_order_id": None,
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

    def fetch_available_usdt(self):
        """Fetches free available USDT balance to avoid code 110007."""
        try:
            bal_resp = self.exchange.fetch_balance({"accountType": "UNIFIED"})
            return float(bal_resp.get("USDT", {}).get("free", 0.0))
        except Exception as e:
            print(f"Error fetching USDT balance: {e}", flush=True)
            return float(self.state.get("current_balance", INITIAL_CORE_MARGIN))

    def cancel_all_conditional_orders(self):
        """Cancels all pending orders for active asset."""
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
        """Maps specific parameters for each Tier using market anchor Fib levels."""
        if tier == 1:
            return {
                "entry_price": self.fibs["0.0"],
                "trigger_price": None,
                "tp_price": self.fibs["23.6"],
                "sl_price": self.fibs["0.0"] * 0.99,
                "leverage": LEVERAGE_BASELINE,
            }
        elif tier == 2:
            return {
                "entry_price": self.fibs["23.6"],
                "trigger_price": self.fibs["38.2"],
                "tp_price": self.fibs["38.2"],
                "sl_price": self.fibs["0.0"],
                "leverage": LEVERAGE_HYPER,
            }
        elif tier == 3:
            return {
                "entry_price": self.fibs["38.2"],
                "trigger_price": self.fibs["50.0"],
                "tp_price": self.fibs["50.0"],
                "sl_price": self.fibs["23.6"],
                "leverage": LEVERAGE_BASELINE,
            }
        elif tier == 4:
            return {
                "entry_price": self.fibs["50.0"],
                "trigger_price": self.fibs["61.8"],
                "tp_price": self.fibs["61.8"],
                "sl_price": self.fibs["38.2"],
                "leverage": LEVERAGE_BASELINE,
            }
        elif tier == 5:
            return {
                "entry_price": self.fibs["61.8"],
                "trigger_price": self.fibs["78.6"],
                "tp_price": self.fibs["78.6"],
                "sl_price": self.fibs["50.0"],
                "leverage": LEVERAGE_HYPER,
            }
        return None

    def initialize_campaign(self, launch_params):
        """Loads new market breakout signal and sets anchor geometry."""
        self.symbol = launch_params["symbol"]
        self.state["symbol"] = launch_params["symbol"]
        self.state["direction"] = launch_params["direction"]
        self.state["anchor_0"] = launch_params["anchor_0"]
        self.state["anchor_100"] = launch_params["anchor_100"]
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
        """Launches Tier 1 dynamically aligned to live market price."""
        config = self.get_tier_config(1)
        safe_set_leverage(self.exchange, config["leverage"], self.symbol)

        avail_usdt = self.fetch_available_usdt()
        margin_to_use = min(INITIAL_CORE_MARGIN, avail_usdt)

        if margin_to_use <= 0:
            send_critical_alert("Cannot execute Tier 1: Insufficient available USDT balance.")
            return

        ticker = self.exchange.fetch_ticker(self.symbol)
        cp = ticker["last"]

        notional_value = max(margin_to_use * config["leverage"], MIN_ORDER_USD)
        qty = notional_value / cp

        try:
            order = self.exchange.create_order(
                symbol=self.symbol,
                type="limit",
                side="buy" if self.state["direction"] == "LONG" else "sell",
                amount=qty,
                price=cp,
                params={
                    "stopLoss": f"{config['sl_price']:.4f}",
                    "takeProfit": f"{config['tp_price']:.4f}",
                    "timeInForce": "PostOnly",
                    "positionIdx": 0,
                },
            )
            self.state["active_tier"] = 1
            self.state["status"] = "RUNNING"
            self.state["pending_order_id"] = order["id"]
            StateManager.save_state(self.state)

            send_telegram(
                f"🚀 <b>CASCADING TIER 1 EXECUTED</b>\n"
                f"<b>Pair:</b> {self.symbol} | <b>Price:</b> ${cp:.4f}\n"
                f"<b>TP:</b> ${config['tp_price']:.4f} | <b>SL:</b> ${config['sl_price']:.4f}\n"
                f"<b>Leverage:</b> {config['leverage']}x Baseline"
            )
        except Exception as e:
            send_critical_alert(f"Failed to execute Tier 1: {e}")

    def queue_conditional_tier(self, tier):
        """Queues Bybit Conditional Limit Order for Tiers 2 through 5."""
        config = self.get_tier_config(tier)
        if not config:
            return

        safe_set_leverage(self.exchange, config["leverage"], self.symbol)

        avail_usdt = self.fetch_available_usdt()
        target_margin = self.state.get("current_balance", INITIAL_CORE_MARGIN)
        margin_to_use = min(target_margin, avail_usdt)

        if margin_to_use <= 0:
            send_critical_alert(f"Cannot queue Tier {tier}: Insufficient USDT balance.")
            return

        notional_value = max(margin_to_use * config["leverage"], MIN_ORDER_USD)
        qty = notional_value / config["entry_price"]

        ticker = self.exchange.fetch_ticker(self.symbol)
        cp = ticker["last"]
        trigger_dir = 1 if config["trigger_price"] > cp else 2

        try:
            order = self.exchange.create_order(
                symbol=self.symbol,
                type="limit",
                side="buy" if self.state["direction"] == "LONG" else "sell",
                amount=qty,
                price=config["entry_price"],
                params={
                    "triggerPrice": f"{config['trigger_price']:.4f}",
                    "triggerBy": "LastPrice",
                    "triggerDirection": trigger_dir,
                    "stopLoss": f"{config['sl_price']:.4f}",
                    "takeProfit": f"{config['tp_price']:.4f}",
                    "timeInForce": "PostOnly",
                    "positionIdx": 0,
                },
            )

            self.state["active_tier"] = tier
            self.state["status"] = "PENDING_PULLBACK"
            self.state["pending_order_id"] = order["id"]
            StateManager.save_state(self.state)

            send_telegram(
                f"⏳ <b>TIER {tier} CONDITIONAL ORDER QUEUED</b>\n"
                f"<b>Pair:</b> {self.symbol}\n"
                f"<b>Trigger:</b> ${config['trigger_price']:.4f} | <b>Limit Entry:</b> ${config['entry_price']:.4f}\n"
                f"<b>Target TP:</b> ${config['tp_price']:.4f} | <b>Leverage:</b> {config['leverage']}x"
            )
        except Exception as e:
            send_critical_alert(f"Failed to queue Tier {tier} conditional order: {e}")

    def evaluate_matrix_step(self):
        """Lifecycle evaluation loop running continuous regime scan + matrix transitions."""
        if self.state["status"] in ["COMPLETED", "FAILED_STOPPED_OUT", "IDLE"]:
            # Scan pairs to find market breakout setup
            signal = scan_for_matrix_trigger(self.exchange)
            if signal:
                self.initialize_campaign(signal)
            return

        ticker = self.exchange.fetch_ticker(self.symbol)
        current_price = ticker["last"]
        active_tier = self.state["active_tier"]

        positions = self.exchange.fetch_positions([self.symbol], params={"category": "linear"})
        has_open_position = any(float(p.get("contracts", 0) or p.get("size", 0)) > 0 for p in positions)

        # Behavioral Safeguard: "No-Fill" Omission Rule
        next_tier_config = self.get_tier_config(active_tier + 1)
        if next_tier_config and not has_open_position:
            next_target = next_tier_config["trigger_price"]
            is_surged = (
                (current_price >= next_target)
                if self.state["direction"] == "LONG"
                else (current_price <= next_target)
            )

            if is_surged and self.state["pending_order_id"]:
                print(f"⚠️ Market surged to next target ${next_target:.4f}. Canceling stale entry...", flush=True)
                self.cancel_all_conditional_orders()

                send_telegram(
                    f"⏩ <b>GUARDRAIL A: TIER {active_tier} SKIPPED</b>\n"
                    f"Price surged past ${next_target:.4f} without filling pullback. Advancing sequence..."
                )

                if active_tier + 1 <= 5:
                    self.queue_conditional_tier(active_tier + 1)
                else:
                    self.state["status"] = "COMPLETED"
                    StateManager.save_state(self.state)
                return

        # Transition Check: Advance upon tier profit completion
        if not has_open_position and self.state["status"] == "RUNNING":
            usdt_bal = self.fetch_available_usdt()
            self.state["current_balance"] = usdt_bal

            send_telegram(
                f"✅ <b>TIER {active_tier} COMPLETED</b>\n"
                f"Compounded Balance Pool: ${usdt_bal:.2f} USDT"
            )

            if active_tier < 5:
                self.queue_conditional_tier(active_tier + 1)
            else:
                self.state["status"] = "COMPLETED"
                StateManager.save_state(self.state)
                send_telegram("🎉 <b>CASCADING MATRIX CAMPAIGN COMPLETED!</b>")


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

    while True:
        try:
            matrix.evaluate_matrix_step()
        except Exception as e:
            send_critical_alert(str(e))

        time.sleep(10)


app = Flask(__name__)


@app.route("/")
def health_check():
    return "OK - Hybrid Cascading Bot Running", 200, {"Content-Type": "text/plain"}


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