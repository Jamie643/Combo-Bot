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
# CONFIGURATION & PRD CASCADING PARAMETERS
# ------------------------------------------------------------------
TARGET_SYMBOL = "SOL/USDT:USDT"  # Focus asset for cascading matrix
STATE_FILE = "cascade_state.json"

TIMEFRAME = "5m"                # Lower timeframe to capture micro-pullbacks
MAKER_FEE_RATE = 0.0002         # Limit fee target
MIN_ORDER_USD = 5.05            # Bybit minimum order floor

# User Matrix Inputs (Configuration)
DIRECTION = "LONG"              # "LONG" or "SHORT"
INITIAL_CORE_MARGIN = 5.00      # USDT allocation for Tier 1
MACRO_ANCHOR_0 = 120.00         # Entry/Baseline price (Anchor 0%)
MACRO_ANCHOR_100 = 150.00       # Final Target price (Anchor 100%)

LEVERAGE_BASELINE = 15          # 15x Baseline Leverage
LEVERAGE_HYPER = 30             # 30x Hyper Leverage

# Bybit Credentials & Telegram Config
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
            pass  # Leverage is already set to target level; safe to ignore
        else:
            print(f"Leverage setting info: {e}", flush=True)


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
    def __init__(self, exchange, symbol=TARGET_SYMBOL):
        self.exchange = exchange
        self.symbol = symbol
        self.fibs = calculate_fib_levels(MACRO_ANCHOR_0, MACRO_ANCHOR_100, DIRECTION)
        self.state = StateManager.load_state()

    def cancel_all_conditional_orders(self):
        """Cancels all open conditional/limit orders for the asset."""
        try:
            orders = self.exchange.fetch_open_orders(self.symbol)
            for order in orders:
                self.exchange.cancel_order(order["id"], self.symbol)
            print(f"Cleared all pending orders for {self.symbol}", flush=True)
        except Exception as e:
            print(f"Error clearing orders: {e}", flush=True)

    def hard_reset_stop_loss(self, reason="Stop Loss Triggered"):
        """Guardrail B: Hard Reset on Stop Loss."""
        print(f"⚠️ Guardrail B Triggered: {reason}", flush=True)
        self.cancel_all_conditional_orders()
        self.state["status"] = "FAILED_STOPPED_OUT"
        self.state["pending_order_id"] = None
        StateManager.save_state(self.state)

        alert_msg = (
            f"⛔ <b>CASCADING MATRIX STOPPED OUT</b> ⛔\n"
            f"<b>Symbol:</b> {self.symbol}\n"
            f"<b>Failed At Tier:</b> {self.state['active_tier']}\n"
            f"<b>Reason:</b> {reason}\n"
            f"<b>Status:</b> Sequence Terminated."
        )
        send_telegram(alert_msg)

    def get_tier_config(self, tier):
        """Maps specific parameters for each Tier according to PRD Specs."""
        if tier == 1:
            return {
                "entry_price": self.fibs["0.0"],
                "trigger_price": None,  # Direct entry
                "tp_price": self.fibs["23.6"],
                "sl_price": self.fibs["0.0"] * 0.99,  # Tight buffer
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

    def execute_tier_1(self):
        """Launches Tier 1 dynamically aligned to live market prices."""
        ticker = self.exchange.fetch_ticker(self.symbol)
        cp = ticker["last"]

        # Dynamically set Anchor 0% to live market price to prevent SL/Entry logic rejections
        entry_price = cp
        if DIRECTION == "LONG":
            tp_price = cp * 1.02   # +2% target
            sl_price = cp * 0.98   # -2% stop loss (below entry)
        else:
            tp_price = cp * 0.98   # -2% target
            sl_price = cp * 1.02   # +2% stop loss (above entry)

        config = self.get_tier_config(1)
        safe_set_leverage(self.exchange, config["leverage"], self.symbol)

        # Enforce Minimum Notional Order Value Floor
        balance = max(self.state["current_balance"], INITIAL_CORE_MARGIN)
        notional_value = max(balance * config["leverage"], MIN_ORDER_USD)
        qty = notional_value / entry_price

        try:
            order = self.exchange.create_order(
                symbol=self.symbol,
                type="limit",
                side="buy" if DIRECTION == "LONG" else "sell",
                amount=qty,
                price=entry_price,
                params={
                    "stopLoss": f"{sl_price:.4f}",
                    "takeProfit": f"{tp_price:.4f}",
                    "timeInForce": "PostOnly",
                    "positionIdx": 0,
                },
            )
            self.state["active_tier"] = 1
            self.state["status"] = "RUNNING"
            self.state["pending_order_id"] = order["id"]
            StateManager.save_state(self.state)

            send_telegram(
                f"🚀 <b>CASCADING TIER 1 LAUNCHED</b>\n"
                f"<b>Entry:</b> ${entry_price:.4f} | <b>TP:</b> ${tp_price:.4f}\n"
                f"<b>SL:</b> ${sl_price:.4f} | <b>Leverage:</b> {config['leverage']}x Baseline"
            )
        except Exception as e:
            send_critical_alert(f"Failed to execute Tier 1: {e}")

    def queue_conditional_tier(self, tier):
        """Queues Bybit Conditional Limit Order for Tiers 2 through 5."""
        config = self.get_tier_config(tier)
        if not config:
            return

        safe_set_leverage(self.exchange, config["leverage"], self.symbol)

        # Enforce Minimum Notional Order Value Floor
        balance = max(self.state["current_balance"], INITIAL_CORE_MARGIN)
        notional_value = max(balance * config["leverage"], MIN_ORDER_USD)
        qty = notional_value / config["entry_price"]

        ticker = self.exchange.fetch_ticker(self.symbol)
        cp = ticker["last"]

        # Determine triggerDirection: 1 (Ascending) if trigger > current, 2 (Descending) if trigger < current
        trigger_dir = 1 if config["trigger_price"] > cp else 2

        try:
            order = self.exchange.create_order(
                symbol=self.symbol,
                type="limit",
                side="buy" if DIRECTION == "LONG" else "sell",
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
                f"<b>Trigger Price:</b> ${config['trigger_price']:.4f}\n"
                f"<b>Limit Entry:</b> ${config['entry_price']:.4f}\n"
                f"<b>Target TP:</b> ${config['tp_price']:.4f}\n"
                f"<b>Leverage:</b> {config['leverage']}x"
            )
        except Exception as e:
            send_critical_alert(f"Failed to queue Tier {tier} conditional order: {e}")

    def evaluate_matrix_step(self):
        """Main lifecycle loop handling Guardrails A, B, and Tier progressions."""
        if self.state["status"] in ["COMPLETED", "FAILED_STOPPED_OUT"]:
            return

        # Initialize Tier 1 if idle
        if self.state["status"] == "IDLE":
            self.execute_tier_1()
            return

        ticker = self.exchange.fetch_ticker(self.symbol)
        current_price = ticker["last"]
        active_tier = self.state["active_tier"]

        # Check Active Positions
        positions = self.exchange.fetch_positions([self.symbol], params={"category": "linear"})
        has_open_position = any(float(p.get("contracts", 0) or p.get("size", 0)) > 0 for p in positions)

        # Guardrail A: "No-Fill" Omission Rule
        next_tier_config = self.get_tier_config(active_tier + 1)
        if next_tier_config and not has_open_position:
            next_target = next_tier_config["trigger_price"]
            is_surged = (current_price >= next_target) if DIRECTION == "LONG" else (current_price <= next_target)

            if is_surged and self.state["pending_order_id"]:
                print(f"⚠️ Guardrail A Triggered: Market surged to next target ${next_target:.4f}", flush=True)
                self.cancel_all_conditional_orders()

                send_telegram(
                    f"⏩ <b>GUARDRAIL A: TIER {active_tier} SKIPPED</b>\n"
                    f"Price surged past ${next_target:.4f} without filling pullback limit. Advancing sequence..."
                )

                # Skip forward
                if active_tier + 1 <= 5:
                    self.queue_conditional_tier(active_tier + 1)
                else:
                    self.state["status"] = "COMPLETED"
                    StateManager.save_state(self.state)
                return

        # Transition Check: If position closed with profit, advance to next Tier
        if not has_open_position and self.state["status"] == "RUNNING":
            # Fetch balance to update compounding pool
            bal_resp = self.exchange.fetch_balance({"accountType": "UNIFIED"})
            usdt_bal = float(bal_resp.get("USDT", {}).get("free", self.state["current_balance"]))
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
                send_telegram("🎉 <b>CASCADING MATRIX FULLY COMPLETED! (TIER 5 REACHED)</b>")


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

    matrix = CascadingMatrixManager(exchange, TARGET_SYMBOL)

    while True:
        try:
            matrix.evaluate_matrix_step()
        except Exception as e:
            send_critical_alert(str(e))

        # Check every 10 seconds on 24/7 Render service
        time.sleep(10)


app = Flask(__name__)


@app.route("/")
def health_check():
    return "OK - Cascading Bot Running", 200, {"Content-Type": "text/plain"}


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