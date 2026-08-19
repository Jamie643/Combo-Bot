import os
import asyncio
import pandas as pd
import numpy as np
import ccxt.async_support as ccxt
import telegram
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
SYMBOLS = ["DOGE/USDT:USDT", "TON/USDT:USDT", "XRP/USDT:USDT", "ADA/USDT:USDT"]
TIMEFRAME = "1h"
LEVERAGE = 2
MIN_ORDER_USD = 5.0

# Retrieve credentials from GitHub Repository Secrets
API_KEY = os.getenv("BYBIT_TESTNET_KEY")
API_SECRET = os.getenv("BYBIT_TESTNET_SECRET")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

tg_bot = telegram.Bot(token=TELEGRAM_TOKEN) if TELEGRAM_TOKEN else None

async def send_alert(message: str):
    logging.info(message)
    if tg_bot and TELEGRAM_CHAT_ID:
        try:
            await tg_bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode="Markdown")
        except Exception as e:
            logging.error(f"Failed to send Telegram alert: {e}")

def calculate_adx(df, period=14):
    data = df.copy()
    data['up'] = data['high'] - data['high'].shift(1)
    data['down'] = data['low'].shift(1) - data['low']
    data['plus_dm'] = np.where((data['up'] > data['down']) & (data['up'] > 0), data['up'], 0.0)
    data['minus_dm'] = np.where((data['down'] > data['up']) & (data['down'] > 0), data['down'], 0.0)
    
    tr1 = data['high'] - data['low']
    tr2 = (data['high'] - data['close'].shift(1)).abs()
    tr3 = (data['low'] - data['close'].shift(1)).abs()
    data['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    data['atr'] = data['tr'].ewm(alpha=1/period, adjust=False).mean()
    data['plus_di'] = 100 * (data['plus_dm'].ewm(alpha=1/period, adjust=False).mean() / data['atr'])
    data['minus_di'] = 100 * (data['minus_dm'].ewm(alpha=1/period, adjust=False).mean() / data['atr'])
    
    dx = 100 * (data['plus_di'] - data['minus_di']).abs() / (data['plus_di'] + data['minus_di'])
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx, data['atr']

async def fetch_signals(exchange, symbol):
    ohlcv = await exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=100)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['std_dev'] = df['close'].rolling(window=20).std()
    df['upper_bb'] = df['sma_20'] + (2.0 * df['std_dev'])
    df['lower_bb'] = df['sma_20'] - (2.0 * df['std_dev'])
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    df['adx'], df['atr'] = calculate_adx(df, period=14)
    
    return df.iloc[-2]  # Last closed candle

async def evaluate_symbol(exchange, symbol):
    try:
        latest = await fetch_signals(exchange, symbol)
        price = latest['close']
        rsi = latest['rsi']
        adx = latest['adx']
        upper_bb = latest['upper_bb']
        lower_bb = latest['lower_bb']
        
        positions = await exchange.fetch_positions([symbol])
        active_pos = next((p for p in positions if float(p['contracts']) > 0), None)
        
        balance = await exchange.fetch_balance()
        usdt_free = balance['USDT']['free']
        
        # Build market description string
        clean_ticker = symbol.split('/')[0]
        market_desc = []
        if price <= lower_bb:
            market_desc.append("At/Below Lower Bollinger Band")
        elif price >= upper_bb:
            market_desc.append("At/Above Upper Bollinger Band")
        else:
            market_desc.append("Inside Bollinger Bands")

        if rsi < 30:
            market_desc.append("Oversold (RSI < 30)")
        elif rsi > 70:
            market_desc.append("Overbought (RSI > 70)")
        else:
            market_desc.append(f"Neutral RSI ({rsi:.1f})")

        if adx < 25:
            market_desc.append("Ranging/Low ADX (< 25)")
        else:
            market_desc.append(f"Trending (ADX {adx:.1f})")

        market_summary_str = ", ".join(market_desc)
        status_str = "❌ *Requirements Not Met*"

        if active_pos:
            side = active_pos['side']
            contracts = float(active_pos['contracts'])
            entry_price = float(active_pos['entryPrice'])
            
            if side == 'long' and price >= latest['sma_20']:
                await exchange.create_market_sell_order(symbol, contracts, params={'reduceOnly': True})
                pnl = (price - entry_price) * contracts
                status_str = f"🔴 *CLOSED LONG* (Est. PnL: `${pnl:.2f}`)"
                await send_alert(f"🔴 *CLOSE LONG* ({symbol})\nPrice: `{price}` | Est. PnL: `${pnl:.2f}`")
            elif side == 'short' and price <= latest['sma_20']:
                await exchange.create_market_buy_order(symbol, contracts, params={'reduceOnly': True})
                pnl = (entry_price - price) * contracts
                status_str = f"🔴 *CLOSED SHORT* (Est. PnL: `${pnl:.2f}`)"
                await send_alert(f"🔴 *CLOSE SHORT* ({symbol})\nPrice: `{price}` | Est. PnL: `${pnl:.2f}`")
            else:
                status_str = f"⏳ *HOLDING {side.upper()} POSITION*"
        else:
            notional = usdt_free * LEVERAGE
            if notional >= MIN_ORDER_USD:
                amount = notional / price
                atr = latest['atr']
                
                if (price <= lower_bb) and (rsi < 30) and (adx < 25):
                    sl = price - (atr * 1.5)
                    await exchange.create_market_buy_order(symbol, amount, params={'stopLoss': str(sl)})
                    status_str = "🟢 *ENTRY LONG TRIGGERED*"
                    await send_alert(f"🟢 *ENTRY LONG* ({symbol})\nEntry: `{price}`\nTarget: `{latest['sma_20']:.4f}`\nSL: `{sl:.4f}`")
                
                elif (price >= upper_bb) and (rsi > 70) and (adx < 25):
                    sl = price + (atr * 1.5)
                    await exchange.create_market_sell_order(symbol, amount, params={'stopLoss': str(sl)})
                    status_str = "🟢 *ENTRY SHORT TRIGGERED*"
                    await send_alert(f"🟢 *ENTRY SHORT* ({symbol})\nEntry: `{price}`\nTarget: `{latest['sma_20']:.4f}`\nSL: `{sl:.4f}`")

        # Return formatted block for the summary scan
        return (
            f"🔹 *{clean_ticker}* @ `${price:.4f}`\n"
            f"• *Market*: {market_summary_str}\n"
            f"• *RSI*: `{rsi:.1f}` | *ADX*: `{adx:.1f}`\n"
            f"• *Status*: {status_str}\n"
        )

    except Exception as e:
        clean_ticker = symbol.split('/')[0]
        return f"⚠️ *{clean_ticker}*: Error evaluating pair (`{str(e)}`)\n"

async def main():
    exchange = ccxt.bybit({
        'apiKey': API_KEY,
        'secret': API_SECRET,
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}
    })
    
    exchange.set_sandbox_mode(True)

    for symbol in SYMBOLS:
        try:
            await exchange.set_leverage(LEVERAGE, symbol)
        except Exception as e:
            logging.warning(f"Leverage note for {symbol}: {e}")

    report_lines = ["🤖 *Combo-Bot Hourly Scan Report*\n"]

    try:
        for symbol in SYMBOLS:
            res = await evaluate_symbol(exchange, symbol)
            report_lines.append(res)
    finally:
        await exchange.close()

    # Send consolidated hourly summary message to Telegram
    final_report = "\n".join(report_lines)
    await send_alert(final_report)

if __name__ == "__main__":
    asyncio.run(main())