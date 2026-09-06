# Multi-Regime Dynamic Leverage & Cascading Trading Bot

An automated Python cryptocurrency trading bot hosted on **Render** utilizing CCXT to execute linear perpetual trades on Bybit Mainnet.

## 🌟 Key Features

* **Multi-Regime Strategy Engine:** Combines Donchian Breakouts (Trend), 61.8% Fibonacci Retests (Retracement), and Bollinger Bands + RSI (Mean Reversion).
* **Conditional Fibonacci Cascading Matrix:** Executes sequential 5-tier Fibonacci pullback trades using Bybit V5 Conditional Limit Orders (`triggerPrice` + `orderPrice`).
* **Smart Dynamic Leverage:** Dynamically scales between **15x Baseline** and **30x Hyper** leverage while maintaining ATR volatility protection and adhering to Bybit's $5.05 USDT order floor.
* **Execution Guardrails:** 
  * **Guardrail A (No-Fill Rule):** Auto-cancels stale conditional orders if market prices surge directly to the next tier's target.
  * **Guardrail B (Hard Reset):** Instantly cancels pending matrix orders and halts the sequence upon hitting a stop-loss.
* **Risk & Capacity Gatekeeper:** Enforces position limits, universal correlation caps, and dynamic equity drawdown protections (capping margin allocations at 40% per position).
* **State Persistence & Alerts:** Utilizes a lightweight local JSON state engine (`cascade_state.json`) to persist sequence progress across Render web service restarts, paired with real-time Telegram event notifications.

## 🛠️ Architecture & Stack

* **Language:** Python 3.10+
* **Exchange Integration:** `ccxt` (Bybit V5 Linear Perpetual API)
* **Web Service:** Flask + Gunicorn (Render 24/7 Web Service Deployment)
* **Notifications:** Telegram Bot API

## ⚙️ Environment Variables Required

Configure these key-value pairs in your Render Dashboard under **Environment Settings**:

| Variable Name | Description |
| :--- | :--- |
| `BYBIT_API_KEY` | Bybit Mainnet API Key |
| `BYBIT_API_SECRET` | Bybit Mainnet API Secret |
| `TELEGRAM_BOT_TOKEN` | Telegram Bot API Token |
| `TELEGRAM_CHAT_ID` | Telegram Chat ID for real-time alerts |

---
*Deployed & maintained on Render.*
