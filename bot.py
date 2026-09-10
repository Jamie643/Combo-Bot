import json
import logging
import os
import threading
import time
import requests
import pandas as pd
import numpy as np
from flask import Flask

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Flask Web Health Check Setup
app = Flask(__name__)

@app.route("/")
def health_check():
    return "Trading engine is running!", 200


class StateManager:
    """Handles persistence of campaign state to guard against container restarts."""
    def __init__(self, filepath="cascade_state.json"):
        self.filepath = filepath
        self.state = self.load_state()

    def load_state(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r") as f:
                    return json.load(f)
            except Exception as e:
                logging.error(f"Error loading state: {e}")
        return {
            "active_campaign": False, 
            "symbol": None, 
            "direction": None, 
            "active_tier": 0, 
            "tier_targets": {}, 
            "orders_placed": {}  # Tracks executed tiers to prevent order spam loops
        }

    def save_state(self):
        try:
            with open(self.filepath, "w") as f:
                json.dump(self.state, f, indent=4)
        except Exception as e:
            logging.error(f"Error saving state: {e}")

    def reset_state(self):
        """Clears active campaign state after trade completion or stop out."""
        self.state = {
            "active_campaign": False,
            "symbol": None,
            "direction": None,
            "active_tier": 0,
            "tier_targets": {},
            "orders_placed": {}
        }
        self.save_state()


class CascadingMatrixManager:
    """Core Execution Engine managing Indicators, Sizing, and Dynamic Hybrid Leverage Rules."""
    
    TIER_CONFIG = {
        1: {"name": "Base Core", "leverage": 5, "atr_mult": 1.75},
        2: {"name": "Acceleration Rail", "leverage": 12, "atr_mult": 1.0},
        3: {"name": "Velocity Maximum", "leverage": 20, "atr_mult": 0.5},
        4: {"name": "De-escalation Bracket", "leverage": 10, "atr_mult": 1.0},
        5: {"name": "Terminal Run", "leverage": 5, "atr_mult": 1.5}
    }

    def __init__(self, exchange_client, state_manager, telegram_config=None):
        self.exchange = exchange_client
        self.state_mgr = state_manager
        self.telegram_config = telegram_config or {}

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes ATR, ADX, Donchian Channels, and Volume Moving Averages."""
        df['tr0'] = abs(df['high'] - df['low'])
        df['tr1'] = abs(df['high'] - df['close'].shift(1))
        df['tr2'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
        df['atr'] = df['tr'].rolling(window=14).mean()

        # Donchian Channels (20-period)
        df['donchian_high'] = df['high'].rolling(window=20).max()
        df['donchian_low'] = df['low'].rolling(window=20).min()

        # Volume Confirmation
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        
        return df

    def detect_market_regime(self, df: pd.DataFrame) -> str:
        """Requires 3 consecutive bars confirming volatility/trend parameters."""
        if len(df) < 20:
            return "RANGE"
        
        breakout_condition = df['close'] > df['donchian_high'].shift(1)
        volume_condition = df['volume'] > (1.5 * df['volume_ma'])
        
        recent_trends = (breakout_condition & volume_condition).tail(3)
        if recent_trends.all():
            return "TREND"
        return "RANGE"

    def calculate_fib_levels(self, anchor_0: float, anchor_100: float, direction: str) -> dict:
        """Calculates key Fibonacci expansion/retracement coordinates."""
        diff = abs(anchor_100 - anchor_0)
        
        if direction.upper() == "SHORT":
            return {
                "0.0": anchor_0,
                "23.6": anchor_0 - (diff * 0.236),
                "38.2": anchor_0 - (diff * 0.382),
                "50.0": anchor_0 - (diff * 0.500),
                "61.8": anchor_0 - (diff * 0.618),
                "78.6": anchor_0 - (diff * 0.786),
                "100.0": anchor_100
            }
        else: # LONG
            return {
                "0.0": anchor_0,
                "23.6": anchor_0 + (diff * 0.236),
                "38.2": anchor_0 + (diff * 0.382),
                "50.0": anchor_0 + (diff * 0.500),
                "61.8": anchor_0 + (diff * 0.618),
                "78.6": anchor_0 + (diff * 0.786),
                "100.0": anchor_100
            }

    def fetch_available_usdt(self) -> float:
        """Retrieves available wallet balance."""
        try:
            balance = self.exchange.get_wallet_balance()
            return float(balance.get("USDT", 0.0))
        except Exception as e:
            logging.error(f"Failed to fetch balance: {e}")
            return 0.0

    def calculate_position_size(self, available_usdt: float, tier: int) -> tuple:
        """35% Dynamic Smart Margin Calculation with Tier Leverage."""
        margin_allocated = available_usdt * 0.35
        leverage = self.TIER_CONFIG[tier]["leverage"]
        position_notional = margin_allocated * leverage
        return margin_allocated, position_notional, leverage

    def calculate_atr_stop_loss(self, entry_price: float, current_atr: float, tier: int, direction: str) -> float:
        """Applies dynamic ATR multiplier according to the tier safety profile."""
        mult = self.TIER_CONFIG[tier]["atr_mult"]
        offset = current_atr * mult
        
        if direction.upper() == "SHORT":
            return round(entry_price + offset, 4)
        else:
            return round(entry_price - offset, 4)

    def queue_conditional_tier(self, symbol: str, tier: int, config: dict):
        """
        Executes order for specified tier with strict Pre-Flight Leverage updates 
        and State Lock checks.
        """
        str_tier = str(tier)
        orders_placed = self.state_mgr.state.get("orders_placed", {})
        
        if orders_placed.get(str_tier):
            logging.debug(f"Tier {tier} order already submitted. Skipping execution loop.")
            return None

        leverage = self.TIER_CONFIG[tier]["leverage"]
        try:
            logging.info(f"Setting account leverage to {leverage}x for {symbol} prior to order execution.")
            self.exchange.set_leverage(symbol=symbol, leverage=leverage)
        except Exception as e:
            logging.error(f"Pre-flight leverage configuration failed: {e}")
            return None

        usdt_balance = self.fetch_available_usdt()
        margin, notional, leverage = self.calculate_position_size(usdt_balance, tier)
        
        entry_price = config["entry"]
        target_tp = config["target_tp"]
        stop_loss = config["stop_loss"]

        qty = notional / entry_price
        
        logging.info(f"Executing Tier {tier} ({self.TIER_CONFIG[tier]['name']}): Margin=${margin:.2f}, Leverage={leverage}x, Notional=${notional:.2f}")
        
        try:
            order_res = self.exchange.place_order(
                symbol=symbol,
                side="SELL",
                qty=qty,
                price=entry_price,
                stop_loss=stop_loss,
                take_profit=target_tp
            )
            
            self.state_mgr.state["orders_placed"][str_tier] = True
            self.state_mgr.save_state()
            
            return order_res
        except Exception as e:
            logging.error(f"Failed to place order for Tier {tier}: {e}")
            return None

    def evaluate_matrix_step(self, symbol: str, current_price: float, active_tier: int, tier_targets: dict):
        """Main execution step evaluating Campaign Status, Guardrail A, and Tier Advances."""
        try:
            has_open_position = self.exchange.has_active_position(symbol)
            has_pending_order = self.exchange.has_pending_order(symbol)

            # Automated Campaign Reset if trade was closed on exchange
            if not has_open_position and not has_pending_order:
                logging.info(f"No active positions or pending orders found for {symbol}. Campaign finished.")
                self.send_telegram_alert(f"🏁 *Campaign Terminated*: Position for {symbol} is closed. Resetting bot state.")
                self.state_mgr.reset_state()
                return

            next_tier = active_tier + 1
            str_next_tier = str(next_tier)
            
            if str_next_tier in tier_targets or next_tier in tier_targets:
                target_key = next_tier if next_tier in tier_targets else str_next_tier
                next_trigger_price = tier_targets[target_key]["entry"]

                # Guardrail A requires active position context + order pending state
                if has_open_position and has_pending_order:
                    # SHORT Direction Check: Current price surged past lower tier trigger
                    if current_price <= next_trigger_price:
                        logging.warning(f"Guardrail A Triggered: Price ({current_price}) surged past Tier {next_tier} target ({next_trigger_price}).")
                        self.exchange.cancel_all_conditional_orders(symbol)
                        self.send_telegram_alert(f"⚡ *Guardrail A Activated*: Skipped stale order for Tier {active_tier}. Advancing to Tier {next_tier}.")
                        
                        self.state_mgr.state["active_tier"] = next_tier
                        self.state_mgr.save_state()
                        
                        self.queue_conditional_tier(symbol, next_tier, tier_targets[target_key])
                        return

                # Tier Advance & Trail-Lock Check
                if has_open_position and current_price <= next_trigger_price:
                    orders_placed = self.state_mgr.state.get("orders_placed", {})
                    
                    if not orders_placed.get(str_next_tier):
                        logging.info(f"Advancing from Tier {active_tier} to Tier {next_tier}")
                        
                        if next_tier == 2:
                            tier_1_key = 1 if 1 in tier_targets else "1"
                            tier_1_entry = tier_targets[tier_1_key]["entry"]
                            try:
                                self.exchange.set_position_stop_loss(symbol, stop_loss_price=tier_1_entry)
                                logging.info(f"Tier 2 Activated: Trail-locked Stop Loss to Tier 1 Entry ({tier_1_entry})")
                            except Exception as e:
                                logging.error(f"Failed setting Trail-Lock SL: {e}")

                        self.state_mgr.state["active_tier"] = next_tier
                        self.state_mgr.save_state()
                        self.queue_conditional_tier(symbol, next_tier, tier_targets[target_key])

        except Exception as e:
            logging.error(f"Error during matrix step evaluation: {e}")

    def send_telegram_alert(self, message: str):
        """Dispatches notification via Telegram API."""
        bot_token = self.telegram_config.get("bot_token")
        chat_id = self.telegram_config.get("chat_id")
        if bot_token and chat_id:
            try:
                url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
                payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
                requests.post(url, json=payload, timeout=5)
            except Exception as e:
                logging.error(f"Failed to send Telegram alert: {e}")


def get_optimal_execution_pairs(exchange_client) -> list:
    """
    Scans Bybit market state to extract high-volume linear USDT perpetual pairs.
    Filters out meme coins and low-liquidity ranges dynamically.
    """
    try:
        tickers = exchange_client.get_tickers(category="linear")
        valid_pairs = []
        blacklisted_tokens = ["DOGE", "SHIB", "PEPE", "WIF", "BONK", "FLOKI", "1000SATS", "ADA"]
        
        for ticker in tickers.get('result', {}).get('list', []):
            symbol = ticker.get('symbol', '')
            turnover_24h = float(ticker.get('turnover24h', 0))
            
            if symbol.endswith("USDT") and turnover_24h > 50_000_000:
                base_asset = symbol.replace("USDT", "")
                if not any(meme == base_asset for meme in blacklisted_tokens):
                    valid_pairs.append(symbol)
                    
        logging.info(f"Market Scanner selected {len(valid_pairs)} optimal execution pairs: {valid_pairs}")
        return valid_pairs
    except Exception as e:
        logging.error(f"Error scanning market pairs: {e}")
        # Fallback to high-beta default pairs if scanner fails
        return ["SOLUSDT", "ETHUSDT", "AVAXUSDT", "LINKUSDT", "GRAMUSDT"]


def bot_loop(manager: CascadingMatrixManager):
    """Infinite loop execution cycle running every 10 seconds."""
    logging.info("Starting Cascading Execution Engine Loop...")
    last_pulse = time.time()
    
    while True:
        try:
            # Query optimal execution pairs dynamically on each iteration
            target_symbols = get_optimal_execution_pairs(manager.exchange)

            for symbol in target_symbols:
                df = manager.exchange.get_ohlcv(symbol, timeframe="1h", limit=50)
                df = manager.calculate_indicators(df)
                regime = manager.detect_market_regime(df)
                
                state = manager.state_mgr.load_state()
                
                # Campaign Initializer
                if not state.get("active_campaign") and regime == "TREND":
                    current_price = df['close'].iloc[-1]
                    low_anchor = df['donchian_low'].iloc[-1]
                    
                    fibs = manager.calculate_fib_levels(current_price, low_anchor, direction="SHORT")
                    current_atr = df['atr'].iloc[-1]
                    
                    tier_targets = {
                        "1": {"entry": fibs["0.0"], "target_tp": fibs["23.6"], "stop_loss": manager.calculate_atr_stop_loss(fibs["0.0"], current_atr, 1, "SHORT")},
                        "2": {"entry": fibs["23.6"], "target_tp": fibs["38.2"], "stop_loss": manager.calculate_atr_stop_loss(fibs["23.6"], current_atr, 2, "SHORT")},
                        "3": {"entry": fibs["38.2"], "target_tp": fibs["50.0"], "stop_loss": manager.calculate_atr_stop_loss(fibs["38.2"], current_atr, 3, "SHORT")},
                        "4": {"entry": fibs["50.0"], "target_tp": fibs["61.8"], "stop_loss": manager.calculate_atr_stop_loss(fibs["50.0"], current_atr, 4, "SHORT")},
                        "5": {"entry": fibs["61.8"], "target_tp": fibs["78.6"], "stop_loss": manager.calculate_atr_stop_loss(fibs["61.8"], current_atr, 5, "SHORT")},
                    }
                    
                    manager.state_mgr.state = {
                        "active_campaign": True,
                        "symbol": symbol,
                        "direction": "SHORT",
                        "active_tier": 1,
                        "tier_targets": tier_targets,
                        "orders_placed": {}
                    }
                    manager.state_mgr.save_state()
                    
                    manager.queue_conditional_tier(symbol, 1, tier_targets["1"])
                    manager.send_telegram_alert(f"🚀 *Campaign Launched*: {symbol} Short Breakout detected. Executing Tier 1.")

                # Active Campaign Monitor
                elif state.get("active_campaign") and state.get("symbol") == symbol:
                    current_price = df['close'].iloc[-1]
                    active_tier = state.get("active_tier", 1)
                    tier_targets = state.get("tier_targets", {})
                    
                    manager.evaluate_matrix_step(symbol, current_price, active_tier, tier_targets)

            # Operational Heartbeat (24-Hour)
            if time.time() - last_pulse > 86400:
                manager.send_telegram_alert("💓 *Daily Pulse*: Execution engine is operational.")
                last_pulse = time.time()

        except Exception as e:
            logging.error(f"Error in main bot loop cycle: {e}")
            manager.send_telegram_alert(f"⚠️ *Critical Engine Alert*: {str(e)}")

        time.sleep(10)


def start_bot_thread():
    """Initializes execution dependencies and launches the bot_loop thread."""
    # Place exchange client instantiation here (e.g., Pybit or CCXT wrapper)
    # exchange_client = ...
    # state_manager = StateManager()
    # telegram_config = {"bot_token": os.environ.get("TELEGRAM_BOT_TOKEN"), "chat_id": os.environ.get("TELEGRAM_CHAT_ID")}
    # manager = CascadingMatrixManager(exchange_client, state_manager, telegram_config)
    # bot_loop(manager)
    pass


if __name__ == "__main__":
    # Launch trading loop in daemon thread
    bot_thread = threading.Thread(target=start_bot_thread, daemon=True)
    bot_thread.start()

    # Serve Flask health check endpoint on port configured by Render
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)