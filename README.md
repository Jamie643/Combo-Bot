# Multi-Regime Dynamic Leverage Trading Bot

An automated Python cryptocurrency trading bot hosted on **Render** utilizing CCXT to execute linear perpetual trades on Bybit Mainnet.

## 🌟 Key Features

* **Multi-Regime Strategy Engine:** Combines Donchian Breakouts (Trend), 61.8% Fibonacci Retests (Retracement), and Bollinger Bands + RSI (Mean Reversion).
* **Smart Dynamic Leverage:** Dynamically calculates required leverage (1x–10x) based on ATR volatility and equity margin limits while adhering to Bybit's $5.05 USDT order minimum.
* **Risk & Capacity Gatekeeper:** Hard limit of maximum 2 concurrent open positions and 40% margin allocation cap per position to prevent over-leveraging.
* **Telegram Integration:** Sends real-time trade alerts, execution details, and diagnostic error reports.

## 🛠️ Architecture & Stack

* **Language:** Python 3.10+
* **Exchange Integration:** `ccxt` (Bybit Linear Perpetual API)
* **Web Service:** Flask + Gunicorn (Render Web Service deployment)
* **Notifications:** Telegram Bot API

## ⚙️ Environment Variables Required

Configure these key-value pairs in your Render Dashboard under **Environment Settings**:

| Variable Name | Description |
| :--- | :--- |
| `BYBIT_API_KEY` | Bybit Mainnet API Key |
| `BYBIT_API_SECRET` | Bybit Mainnet API Secret |
| `TELEGRAM_BOT_TOKEN` | Telegram Bot API Token |
| `TELEGRAM_CHAT_ID` | Telegram Chat ID for alerts |

---
*Deployed & maintained on Render.*
