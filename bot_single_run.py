import os
import sys
import asyncio
import logging
from decimal import Decimal, ROUND_DOWN
import ccxt.async_support as ccxt
import pandas as pd
import numpy as np
import requests

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# ---------------------------------------------------------------------------
# Environment Variables & Config
# ---------------------------------------------------------------------------
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Updated Telegram Chat ID
TELEGRAM_CHAT_ID = "-1003952503235"

SYMBOLS = ["DOGE/USDT:USDT", "TON/USDT:USDT", "XRP/USDT:USDT", "ADA/USDT:USDT"]
TIMEFRAME = "1h"
LEVERAGE = 5
RISK_PER_TRADE = 0.02  # 2% equity risk per trade

# ---------------------------------------------------------------------------
# Telegram Helper
# ---------------------------------------------------------------------------
def send_telegram_message(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logging.warning("Telegram credentials missing. Skipping notification.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        response = requests.post(url, json=payload, timeout=10)
        res_data = response.json()
        if not res_data.get("ok"):
            logging.error(f"Telegram API Error: {res_data}")
        else:
            logging.info("Telegram notification sent successfully.")
    except Exception as e:
        logging.error(f"Failed to send Telegram message: {e}")

# ---------------------------------------------------------------------------
# Indicator Calculations
# ---------------------------------------------------------------------------
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # 20 SMA & Bollinger Bands
    df["sma20"] = df["close"].rolling(window=20).mean()
    df["std20"] = df["close"].rolling(window=20).std()
    df["upper_bb"] = df["sma20"] + (2 * df["std20"])
    df["lower_bb"] = df["sma20"] - (2 * df["std20"])

    # 14 RSI
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # ATR (14)
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df["atr"] = true_range.rolling(14).mean()

    return df

# ---------------------------------------------------------------------------
# Exchange Initialization (Proxy Bypassing for GitHub Actions)
# ---------------------------------------------------------------------------
async def init_exchange():
    exchange = ccxt.bybit({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "timeout": 30000,
        "options": {
            "defaultType": "future",
            "adjustForTimeDifference": True,
        },
        "headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
    })
    
    # Configure testnet endpoints
    exchange.set_sandbox_mode(True)
    
    # Proxy configuration to handle region-based 403 Forbidden restrictions
    # Redirects API traffic via HTTPS proxy endpoint
    proxy_url = os.getenv("HTTP_PROXY") or os.getenv("HTTPS_PROXY")
    if proxy_url:
        exchange.aiohttp_proxy = proxy_url
        logging.info(f"Using environment proxy: {proxy_url}")
    
    return exchange

# ---------------------------------------------------------------------------
# Core Execution Strategy Loop
# ---------------------------------------------------------------------------
async def run_bot():
    exchange = await init_exchange()
    summary_logs = []
    
    try:
        logging.info("Connecting to Bybit Testnet...")
        await exchange.load_markets()
        
        balance_info = await exchange.fetch_balance()
        usdt_free = balance_info.get("USDT", {}).get("free", 0.0)
        summary_logs.append(f"🤖 *Bot Execution Status*\n• Available Balance: `${usdt_free:.2f} USDT`")

        for symbol in SYMBOLS:
            logging.info(f"Processing symbol: {symbol}")
            try:
                # Set Leverage
                try:
                    await exchange.set_leverage(LEVERAGE, symbol)
                except Exception as lev_err:
                    logging.debug(f"Leverage set notice for {symbol}: {lev_err}")

                # Fetch OHLCV
                ohlcv = await exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=50)
                df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df = calculate_indicators(df)

                last_row = df.iloc[-1]
                close_price = last_row["close"]
                rsi = last_row["rsi"]
                lower_bb = last_row["lower_bb"]
                upper_bb = last_row["upper_bb"]

                summary_logs.append(
                    f"\n📊 *{symbol}*\n"
                    f"• Close: `{close_price:.4f}` | RSI: `{rsi:.1f}`\n"
                    f"• Bands: Low `{lower_bb:.4f}` | High `{upper_bb:.4f}`"
                )

                # Check position
                positions = await exchange.fetch_positions([symbol])
                open_pos = [p for p in positions if float(p.get("contracts", 0)) > 0]

                if not open_pos:
                    # Entry logic: Mean reversion (Oversold near lower BB)
                    if close_price <= lower_bb and rsi < 35:
                        atr = last_row["atr"]
                        stop_loss = close_price - (1.5 * atr)
                        risk_per_unit = close_price - stop_loss
                        
                        position_size = (usdt_free * RISK_PER_TRADE) / risk_per_unit
                        
                        # Market precision adjustment
                        market = exchange.market(symbol)
                        amount = float(exchange.amount_to_precision(symbol, position_size))

                        if amount > 0:
                            order = await exchange.create_order(
                                symbol=symbol,
                                type="market",
                                side="buy",
                                amount=amount,
                                params={"stopLoss": exchange.price_to_precision(symbol, stop_loss)}
                            )
                            alert_msg = f"🚀 *BUY Signal Executed*\nSymbol: `{symbol}`\nAmount: `{amount}`\nPrice: `{close_price}`"
                            logging.info(alert_msg)
                            send_telegram_message(alert_msg)

            except Exception as e:
                logging.error(f"Error executing logic for {symbol}: {e}")
                summary_logs.append(f"\n⚠️ *{symbol} Error*: `{str(e)}`")

        # Send final execution log summary
        send_telegram_message("\n".join(summary_logs))

    except Exception as e:
        logging.critical(f"Critical Bot Failure: {e}", exc_info=True)
        send_telegram_message(f"🚨 *Critical Bot Error*: `{str(e)}`")
    finally:
        await exchange.close()
        logging.info("Exchange connection closed.")

if __name__ == "__main__":
    asyncio.run(run_bot())