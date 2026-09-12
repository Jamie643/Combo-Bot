"""
bot.py — CascadeBot: Bybit USDT-perpetual cascading Fibonacci trading bot.

Implements the four-stage architecture end-to-end in a single file:

  Stage 1 — Market Scanner        (universe -> ranked, de-correlated candidates)
  Stage 2 — Regime Detection 1H   (candidates -> TREND/RANGE/AMBIGUOUS + confidence)
  Stage 3 — Cascading Execution   (TREND -> Fib-mapped, conviction-scaled campaign)
  Stage 4 — State & Operations    (persistence, execution, reconciliation, alerts)

Symbol convention:
  - Internal identity: raw Bybit form (BTCUSDT) — state file, config, logs.
  - ccxt boundary:    unified form (BTC/USDT:USDT) — used only for API calls.
  - Conversion:       BybitClient.raw_symbol() / BybitClient.ccxt_symbol().

Deployment: Render (Flask health server on port 10000, SIGTERM graceful shutdown).
"""

import asyncio
import atexit
import fcntl
import json
import logging
import os
import signal
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import wraps
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# =====================================================================
# SECTION 1 — IMPORTS AND CONFIGURATION
# =====================================================================

import ccxt.async_support as ccxt  # noqa: E402
from aiohttp import ClientSession, ClientTimeout, WSMsgType  # noqa: E402
from flask import Flask, jsonify, request  # noqa: E402

CONFIG: Dict[str, Any] = {
    # ---- Stage 1: Universe / hard filters ----
    "MIN_TURNOVER_24H": 5_000_000,
    "MAX_SPREAD_BPS": 10.0,
    "MIN_DEPTH_1PCT_USD": 100_000,
    "MIN_LISTING_AGE_DAYS": 30,
    "MAX_FUNDING_ABS": 0.001,
    "MIN_ATR_DAILY_PCT": 0.8,
    "MAX_ATR_DAILY_PCT": 12.0,
    "STABLECOIN_SYMBOLS": ["USDCUSDT", "USDEUSDT", "DAIUSDT", "FDUSDUSDT", "TUSDUSDT",
                           "EURUSDT", "BUSDUSDT", "USD1USDT", "WUSDUSDT"],
    "KLINES_LIMIT": 200,
    "ATR_PERIOD": 14,
    "CORRELATION_LOOKBACK_DAYS": 30,
    "CORRELATION_THRESHOLD": 0.85,
    "CORRELATION_REFRESH_HOURS": 4,
    "CORRELATION_MAX_AGE_HOURS": 8,
    "SCORE_WEIGHTS": {"liquidity": 0.20, "spread": 0.15, "depth": 0.15,
                      "volatility": 0.15, "funding": 0.10, "trend": 0.15,
                      "correlation": 0.10},

    # ---- Stage 2: Regime detection ----
    "DONCHIAN_PERIOD": 20,
    "VOLUME_MA_PERIOD": 20,
    "VOLUME_SPIKE_MULT": 1.5,
    "ADX_PERIOD": 14,
    "ADX_MIN_STRONG": 25.0,
    "ADX_MIN_WEAK": 20.0,
    "MIN_CLOSES_OUTSIDE": 2,
    "MAX_BREAKOUT_AGE": 10,
    "MIN_CANDLES_REQUIRED": 100,
    "MAX_KLINE_GAP_CANDLES": 2,
    "MAX_KLINE_AGE_SECONDS": 120,
    "HTF_INTERVAL": "240",
    "HTF_CONFIDENCE_PENALTY": 0.20,
    "HTF_CONFIDENCE_CAP_NO_DATA": 0.65,
    "CONF_TREND_MIN": 0.70,
    "CONF_AMBIG_MIN": 0.40,
    "CONF_WEIGHTS": {"donchian": 0.30, "volume": 0.20, "adx": 0.20,
                     "persistence": 0.15, "htf": 0.15},

    # ---- Stage 3: Campaign ----
    "MAX_CONCURRENT_CAMPAIGNS": 2,
    "MAX_CAMPAIGN_DURATION_HOURS": 48,
    "FIB_ANCHOR_LOOKBACK": 50,
    "FIB_MIN_RANGE_ATR_MULT": 2.0,
    "FIB_MAX_RANGE_PCT": 0.20,
    "FIB_RATIOS_3TIER": [0.0, 0.236, 0.382],
    "FIB_RATIOS_4TIER": [0.0, 0.236, 0.382, 0.500],
    "FIB_RATIOS_5TIER": [0.0, 0.236, 0.382, 0.500, 0.618],
    "FIB_TP_EXTENSION": 1.000,
    "LEVERAGE_LADDER": [5, 20, 12, 8, 5],
    "CONVICTION_LEVERAGE_MULT": {"STANDARD": 1.0, "STRONG": 1.25, "ELITE": 1.5},
    "TIER_WEIGHTS_3": [0.5, 1.0, 0.8],
    "TIER_WEIGHTS_5": [0.5, 1.0, 0.8, 0.6, 0.4],
    "CAMPAIGN_MULTIPLIERS": {"STANDARD": 1.0, "STRONG": 1.5, "ELITE": 2.5},
    "CONFIDENCE_CLASS_BANDS": {"STANDARD": (0.70, 0.79),
                               "STRONG": (0.80, 0.89),
                               "ELITE": (0.90, 1.00)},
    "AGGRESSION_FACTOR_MICRO": 0.20,
    "AGGRESSION_FACTOR_MID": 0.15,
    "AGGRESSION_FACTOR_MATURE": 0.10,
    "MIN_NOTIONAL_FLOOR": 5.00,
    "MAX_MARGIN_DEPLOYED_PCT": 0.45,
    "SL_BUFFER_ATR_MULT": 0.25,
    "SL_CAP_STANDARD_PCT": 0.03,
    "SL_CAP_ELITE_PCT": 0.02,
    "SL_TIME_DECAY_CANDLES": 10,
    "SL_VOLATILITY_WIDEN_MULT": 2.0,
    "SL_TRAIL_ATR_MULT": 0.5,
    "ZONE_HALF_WIDTH_ATR_MULT": 0.5,
    "VOLUME_CONFIRMATION_MULT": 1.0,
    "TRIGGER_ORDER_TYPE": "PostOnly",
    "TRIGGER_SLIPPAGE_TOLERANCE": 0.001,
    "GAP_FILL_TIMEOUT_CANDLES": 3,
    "DRAWDOWN_HALT_24H_PCT": 0.25,
    "DRAWDOWN_HALT_MANUAL_PCT": 0.40,
    "COMPOUNDING_MIN_CAMPAIGNS": 20,
    "COMPOUNDING_MIN_WINRATE": 0.45,
    "COMPOUNDING_MAX_DD_PCT": 0.15,
    "MODE_MICRO_MAX": 100,
    "MODE_MID_MAX": 400,
    "MODE_MATURE_MIN": 1000,

    # ---- Stage 4: Operations ----
    "STATE_FILE": "cascade_state.json",
    "STATE_BACKUP_COUNT": 5,
    "LOCK_FILE": "cascade.lock",
    "ORDER_RETRY_MAX": 3,
    "ORDER_RETRY_BACKOFF_S": [1, 3, 9],
    "POST_ONLY_FALLBACK": True,
    "SL_PLACE_BEFORE_CANCEL": True,
    "CLIENT_ORDER_ID_FORMAT": "{campaign_id}-T{tier}-{attempt}",
    "RECONCILE_INTERVAL_S": 30,
    "ORPHAN_POSITION_ACTION": "alert_only",
    "TOPUP_QUEUE_ENABLED": True,
    "DEPLOY_ON_ALL_CLOSED": True,
    "TELEGRAM_ENABLED": True,
    "ALERT_THROTTLE_PER_MIN": 1,
    "HEARTBEAT_INTERVAL_HOURS": 4,
    "HEARTBEAT_UTC_HOURS": [0, 4, 8, 12, 16, 20],
    "HEARTBEAT_ALWAYS_SEND": True,
    "HEARTBEAT_RETRY_MAX": 3,
    "DAILY_PULSE_UTC_HOUR": 0,
    "SILENCE_COMMAND_ENABLED": True,
    "FLASK_PORT": 10000,
    "FLASK_BIND": "0.0.0.0",
    "SCAN_INTERVAL_MIN": 15,
    "INVALIDATION_INTERVAL_S": 60,
    "GRACEFUL_SHUTDOWN_TIMEOUT_S": 30,
    "CIRCUIT_BREAKER_FAILURES": 3,
    "CIRCUIT_BREAKER_COOLDOWN_S": 300,
    "API_TIMEOUT_S": 5,
    "WS_PING_INTERVAL_S": 20,
    "LOG_FILE": "bot.log",
    "LOG_MAX_BYTES": 10 * 1024 * 1024,
    "LOG_BACKUP_COUNT": 5,
}

# ---- Secrets from environment ----
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "")
BYBIT_TESTNET = os.getenv("BYBIT_TESTNET", "true").lower() == "true"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
HEALTH_API_TOKEN = os.getenv("HEALTH_API_TOKEN", "")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# =====================================================================
# SECTION 2 — UTILITIES
# =====================================================================


def setup_logging() -> logging.Logger:
    """Configure root logger: stdout (Render) + rotating file bot.log."""
    logger = logging.getLogger("cascade")
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = RotatingFileHandler(CONFIG["LOG_FILE"], maxBytes=CONFIG["LOG_MAX_BYTES"],
                             backupCount=CONFIG["LOG_BACKUP_COUNT"])
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


log = setup_logging()


def now_utc() -> datetime:
    """Return current UTC time as an aware datetime."""
    return datetime.now(timezone.utc)


def iso_now() -> str:
    """Return current UTC time as ISO-8601 string."""
    return now_utc().isoformat()


def retry_async(max_attempts: int = 3, backoff: Tuple[int, ...] = (1, 3, 9)):
    """Async retry decorator: exponential backoff; 429 rate limits sleep 60s."""
    def deco(fn):
        @wraps(fn)
        async def wrapper(*args, **kwargs):
            last_exc: Optional[Exception] = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return await fn(*args, **kwargs)
                except ccxt.RateLimitExceeded as exc:
                    last_exc = exc
                    log.warning("rate limited on %s; sleeping 60s", fn.__name__)
                    await asyncio.sleep(60)
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    if attempt >= max_attempts:
                        break
                    delay = backoff[min(attempt - 1, len(backoff) - 1)]
                    log.warning("%s attempt %d failed (%s); retry in %ds",
                                fn.__name__, attempt, exc, delay)
                    await asyncio.sleep(delay)
            raise last_exc  # type: ignore[misc]
        return wrapper
    return deco


def atomic_write_json(path: str, data: Dict[str, Any], backup_count: int = 0) -> None:
    """Atomically persist JSON: write .tmp, fsync, rename, fsync directory."""
    tmp_path = path + ".tmp"
    directory = os.path.dirname(os.path.abspath(path)) or "."
    if backup_count > 0 and os.path.exists(path):
        for i in range(backup_count - 1, 0, -1):
            src, dst = f"{path}.bak.{i}", f"{path}.bak.{i + 1}"
            if os.path.exists(src):
                os.replace(src, dst)
        os.replace(path, f"{path}.bak.1")
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_path, path)
    dir_fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


class FileLock:
    """POSIX advisory file lock; exits if another instance holds it."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.fh: Optional[Any] = None

    def acquire(self) -> None:
        """Acquire the lock or raise RuntimeError if already held."""
        self.fh = open(self.path, "w")
        try:
            fcntl.flock(self.fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("another bot instance holds the lock; exiting") from exc
        self.fh.write(str(os.getpid()))
        self.fh.flush()

    def release(self) -> None:
        """Release the lock and remove the lock file."""
        if self.fh:
            try:
                fcntl.flock(self.fh, fcntl.LOCK_UN)
                self.fh.close()
            finally:
                if os.path.exists(self.path):
                    os.remove(self.path)
                self.fh = None


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range over `period` bars (simple mean of TR)."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index (Wilder smoothing)."""
    high, low, close = df["high"], df["low"], df["close"]
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr = pd.concat([(high - low), (high - close.shift(1)).abs(),
                    (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, min_periods=period).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, min_periods=period).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, min_periods=period).mean()


def donchian(df: pd.DataFrame, period: int = 20) -> Tuple[float, float]:
    """Donchian channel excluding the current (forming) candle."""
    window = df.iloc[-period - 1:-1]
    return float(window["high"].max()), float(window["low"].min())


def md_escape(text: str) -> str:
    """Escape a string for Telegram MarkdownV2 parse mode."""
    for ch in "_*[]()~`>#+-=|{}.!":
        text = text.replace(ch, f"\\{ch}")
    return text


# =====================================================================
# SECTION 3 — BYBIT API CLIENT WRAPPER
# =====================================================================


class BybitClient:
    """Thin async wrapper over ccxt's Bybit V5 client with circuit breaking.

    Symbol convention:
      - ccxt unified:  BTC/USDT:USDT  (used ONLY for ccxt API calls)
      - raw Bybit:     BTCUSDT        (used for all internal identity)
      - Convert via raw_symbol() / ccxt_symbol() at the boundary.
    """

    def __init__(self) -> None:
        config: Dict[str, Any] = {
            "apiKey": BYBIT_API_KEY,
            "secret": BYBIT_API_SECRET,
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},
        }
        self.exchange = ccxt.bybit(config)
        if BYBIT_TESTNET:
            self.exchange.set_sandbox_mode(True)
        self.testnet = BYBIT_TESTNET
        self._consecutive_failures = 0
        self._breaker_open_until = 0.0
        self.last_latency_ms: float = 0.0
        self.markets_loaded = False

    async def close(self) -> None:
        """Close the underlying ccxt session."""
        await self.exchange.close()

    @staticmethod
    def raw_symbol(symbol: str) -> str:
        """Convert ccxt unified (BTC/USDT:USDT) to raw Bybit (BTCUSDT)."""
        if ":" in symbol:
            base_quote, _ = symbol.split(":")
            base, quote = base_quote.split("/")
            return f"{base}{quote}"
        return symbol.replace("/", "")

    @staticmethod
    def ccxt_symbol(symbol: str, quote: str = "USDT") -> str:
        """Convert raw Bybit (BTCUSDT) to ccxt unified (BTC/USDT:USDT)."""
        if ":" in symbol:
            return symbol
        if symbol.endswith(quote):
            base = symbol[: -len(quote)]
            return f"{base}/{quote}:{quote}"
        return symbol

    async def _guard(self, coro: Any, use_cache: Optional[Any] = None) -> Any:
        """Run a coroutine under the circuit breaker; measure latency."""
        if time.time() < self._breaker_open_until:
            if use_cache is not None:
                return use_cache
            raise RuntimeError("circuit breaker OPEN")
        t0 = time.monotonic()
        try:
            result = await coro
            self._consecutive_failures = 0
            self.last_latency_ms = (time.monotonic() - t0) * 1000
            return result
        except Exception:
            self._consecutive_failures += 1
            if self._consecutive_failures >= CONFIG["CIRCUIT_BREAKER_FAILURES"]:
                self._breaker_open_until = time.time() + CONFIG["CIRCUIT_BREAKER_COOLDOWN_S"]
                log.critical("circuit breaker OPEN for %ds",
                             CONFIG["CIRCUIT_BREAKER_COOLDOWN_S"])
            raise

    async def load_markets(self) -> Dict[str, Any]:
        """Load and cache ccxt markets (keyed by ccxt unified symbols)."""
        if not self.markets_loaded:
            await self.exchange.load_markets()
            self.markets_loaded = True
        return self.exchange.markets

    @retry_async()
    async def fetch_tickers(self) -> Dict[str, Any]:
        """Fetch linear USDT-perp tickers, KEYED BY RAW BYBIT SYMBOL (BTCUSDT)."""
        raw = await self._guard(self.exchange.fetch_tickers())
        normalized: Dict[str, Any] = {}
        for sym, ticker in raw.items():
            if not sym.endswith(":USDT"):
                continue
            raw_sym = self.raw_symbol(sym)
            t = dict(ticker)
            t["_ccxt_symbol"] = sym
            normalized[raw_sym] = t
        return normalized

    @retry_async()
    async def fetch_markets_meta(self) -> Dict[str, Any]:
        """Fetch instrument metadata, KEYED BY RAW BYBIT SYMBOL (BTCUSDT)."""
        markets = await self.load_markets()
        remapped: Dict[str, Any] = {}
        for sym, m in markets.items():
            if not sym.endswith(":USDT"):
                continue
            remapped[self.raw_symbol(sym)] = m
        return remapped

    @retry_async()
    async def fetch_order_book(self, symbol: str, limit: int = 50) -> Dict[str, Any]:
        """Fetch order book snapshot. `symbol` may be raw or ccxt form."""
        return await self._guard(
            self.exchange.fetch_order_book(self.ccxt_symbol(symbol), limit))

    @retry_async()
    async def fetch_klines(self, symbol: str, timeframe: str = "1h",
                           limit: int = 200) -> pd.DataFrame:
        """Fetch OHLCV. `symbol` may be raw or ccxt form."""
        raw = await self._guard(
            self.exchange.fetch_ohlcv(self.ccxt_symbol(symbol),
                                      timeframe=timeframe, limit=limit))
        df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        return df

    @retry_async(max_attempts=CONFIG["ORDER_RETRY_MAX"],
                 backoff=tuple(CONFIG["ORDER_RETRY_BACKOFF_S"]))
    async def place_order(self, symbol: str, side: str, order_type: str, amount: float,
                          price: Optional[float] = None,
                          params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Place an order. `symbol` may be raw or ccxt form."""
        merged = params or {}
        return await self._guard(
            self.exchange.create_order(self.ccxt_symbol(symbol), order_type, side,
                                       amount, price, merged))

    @retry_async(max_attempts=CONFIG["ORDER_RETRY_MAX"],
                 backoff=tuple(CONFIG["ORDER_RETRY_BACKOFF_S"]))
    async def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """Cancel an order. `symbol` may be raw or ccxt form."""
        return await self._guard(
            self.exchange.cancel_order(order_id, self.ccxt_symbol(symbol)))

    @retry_async()
    async def fetch_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch open orders. `symbol=None` fetches all; otherwise raw or ccxt form."""
        coro = (self.exchange.fetch_open_orders(self.ccxt_symbol(symbol))
                if symbol else self.exchange.fetch_open_orders())
        return await self._guard(coro)

    @retry_async()
    async def fetch_positions(self) -> List[Dict[str, Any]]:
        """Fetch all non-zero linear positions, each augmented with 'symbol_raw'."""
        positions = await self._guard(self.exchange.fetch_positions())
        out: List[Dict[str, Any]] = []
        for p in positions:
            contracts = abs(float(p.get("contracts") or 0))
            info_size = abs(float(p.get("info", {}).get("size") or 0))
            if contracts <= 0 and info_size <= 0:
                continue
            ccxt_sym = p.get("symbol", "")
            p["symbol_raw"] = self.raw_symbol(ccxt_sym)
            out.append(p)
        return out

    @retry_async()
    async def fetch_balance(self) -> Dict[str, Any]:
        """Fetch wallet balance (USDT)."""
        return await self._guard(self.exchange.fetch_balance())

    async def fetch_free_usdt(self) -> float:
        """Return free USDT margin, 0.0 on failure."""
        try:
            bal = await self.fetch_balance()
            return float(bal.get("free", {}).get("USDT") or 0.0)
        except Exception as exc:  # noqa: BLE001
            log.error("fetch_free_usdt failed: %s", exc)
            return 0.0

    async def fetch_total_usdt(self) -> float:
        """Return total USDT equity, 0.0 on failure."""
        try:
            bal = await self.fetch_balance()
            usdt = bal.get("USDT", {})
            return float(usdt.get("total") or usdt.get("free") or 0.0)
        except Exception as exc:  # noqa: BLE001
            log.error("fetch_total_usdt failed: %s", exc)
            return 0.0

    def _market(self, symbol: str) -> Dict[str, Any]:
        """Look up a market by raw or ccxt symbol."""
        if ":" in symbol:
            return self.exchange.markets.get(symbol, {})
        return self.exchange.markets.get(self.ccxt_symbol(symbol), {})

    def round_qty(self, symbol: str, qty: float) -> float:
        """Round quantity to the instrument's quantity step."""
        market = self._market(symbol)
        step = float(market.get("precision", {}).get("amount") or 0) or None
        if step and step > 0:
            return float(self.exchange.amount_to_precision(
                self.ccxt_symbol(symbol), qty))
        return qty

    def round_price(self, symbol: str, price: float) -> float:
        """Round price to the instrument's tick size."""
        return float(self.exchange.price_to_precision(
            self.ccxt_symbol(symbol), price))

    def max_leverage(self, symbol: str) -> float:
        """Return the instrument's maximum leverage."""
        market = self._market(symbol)
        limits = market.get("limits", {}).get("leverage", {})
        mx = limits.get("max")
        return float(mx) if mx else 100.0

    def min_notional(self, symbol: str) -> float:
        """Return the instrument's minimum notional (cost)."""
        market = self._market(symbol)
        mn = market.get("limits", {}).get("cost", {}).get("min")
        return float(mn) if mn else 5.0


class BybitWebSocket:
    """Private WebSocket (aiohttp) subscribing to `order` and `position` streams."""

    def __init__(self, client: BybitClient) -> None:
        self.client = client
        self._handlers: List[Any] = []
        self._running = False
        self._task: Optional[asyncio.Task] = None

    def register(self, handler: Any) -> None:
        """Register an async callable(event: dict) for order/position events."""
        self._handlers.append(handler)

    async def _dispatch(self, event: Dict[str, Any]) -> None:
        for handler in self._handlers:
            try:
                await handler(event)
            except Exception as exc:  # noqa: BLE001
                log.error("WS handler error: %s", exc)

    def start(self) -> None:
        """Start the listener as a background task."""
        self._running = True
        self._task = asyncio.create_task(self._run(), name="bybit-ws")

    async def stop(self) -> None:
        """Stop the listener."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run(self) -> None:
        backoff = 1
        base = ("wss://stream-testnet.bybit.com/v5/private" if self.client.testnet
                else "wss://stream.bybit.com/v5/private")
        while self._running:
            try:
                await self._connect_and_listen(base)
                backoff = 1
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.warning("WS disconnected (%s); reconnect in %ds", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)

    async def _connect_and_listen(self, url: str) -> None:
        timeout = ClientTimeout(total=CONFIG["API_TIMEOUT_S"] * 4)
        async with ClientSession(timeout=timeout) as session:
            async with session.ws_connect(url, heartbeat=CONFIG["WS_PING_INTERVAL_S"]) as ws:
                import hashlib
                import hmac as hmac_mod
                expires = int(time.time() * 1000) + 5000
                signature = hmac_mod.new(
                    BYBIT_API_SECRET.encode(), f"GET/realtime{expires}".encode(),
                    hashlib.sha256).hexdigest()
                await ws.send_json({"op": "auth",
                                    "args": [BYBIT_API_KEY, expires, signature]})
                auth_resp = await ws.receive_json()
                if auth_resp.get("success") is not True:
                    raise RuntimeError(f"WS auth failed: {auth_resp}")
                await ws.send_json({"op": "subscribe", "args": ["order", "position"]})
                log.info("Bybit WS authenticated; subscribed to order+position")
                async for msg in ws:
                    if msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSED,
                                    WSMsgType.ERROR, WSMsgType.CLOSING):
                        break
                    if msg.type == WSMsgType.TEXT:
                        try:
                            data = json.loads(msg.data)
                        except json.JSONDecodeError:
                            continue
                        if data.get("op") == "ping":
                            await ws.send_json({"op": "pong"})
                        topic = data.get("topic", "")
                        for key in ("order", "position"):
                            if not topic.startswith(key):
                                continue
                            for event in data.get("data", []):
                                if "symbol" in event:
                                    event["symbol_raw"] = self.client.raw_symbol(
                                        event["symbol"])
                                await self._dispatch({"topic": key, "data": event})


# =====================================================================
# SECTION 4 — TELEGRAM NOTIFIER
# =====================================================================


class Notifier:
    """Telegram alert dispatcher: MarkdownV2 messages, throttling, silence mode."""

    def __init__(self) -> None:
        self.enabled = bool(CONFIG["TELEGRAM_ENABLED"] and TELEGRAM_BOT_TOKEN
                            and TELEGRAM_CHAT_ID)
        self._throttle: Dict[Tuple[str, str], float] = {}
        self._session: Optional[ClientSession] = None
        self._silenced_until = 0.0
        self._update_offset = 0

    async def _get_session(self) -> ClientSession:
        if self._session is None or self._session.closed:
            self._session = ClientSession(timeout=ClientTimeout(total=10))
        return self._session

    async def close(self) -> None:
        """Close the underlying HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    def _throttled(self, event_type: str, campaign_id: str) -> bool:
        key = (event_type, campaign_id)
        now = time.time()
        window = 60.0 / max(CONFIG["ALERT_THROTTLE_PER_MIN"], 1)
        if now - self._throttle.get(key, 0.0) < window:
            return True
        self._throttle[key] = now
        return False

    async def send(self, message: str, event_type: str = "generic",
                   campaign_id: str = "-", critical: bool = False,
                   force: bool = False) -> bool:
        """Send a MarkdownV2 message; retry 3x; log to disk on total failure."""
        if not self.enabled:
            log.info("[telegram disabled] %s: %s", event_type, message)
            return True
        if not (critical or force):
            if time.time() < self._silenced_until:
                log.info("[silenced] %s", message)
                return True
            if self._throttled(event_type, campaign_id):
                return True
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message,
                   "parse_mode": "MarkdownV2"}
        for attempt in range(1, CONFIG["HEARTBEAT_RETRY_MAX"] + 1):
            try:
                session = await self._get_session()
                async with session.post(url, json=payload) as resp:
                    if resp.status == 200:
                        return True
                    body = await resp.text()
                    log.warning("telegram send attempt %d: HTTP %s %s",
                                attempt, resp.status, body[:200])
            except Exception as exc:  # noqa: BLE001
                log.warning("telegram send attempt %d failed: %s", attempt, exc)
            await asyncio.sleep(1.5 * attempt)
        log.error("telegram undeliverable after retries; logging locally: %s", message)
        return False

    async def poll_commands(self, bot_ref: Any) -> None:
        """Long-poll Telegram for operator commands (/silence, /status, /halt)."""
        if not self.enabled or not CONFIG["SILENCE_COMMAND_ENABLED"]:
            return
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
        while True:
            try:
                session = await self._get_session()
                async with session.get(
                        url, params={"offset": self._update_offset,
                                     "timeout": 25}) as resp:
                    if resp.status != 200:
                        await asyncio.sleep(5)
                        continue
                    data = await resp.json()
                for update in data.get("result", []):
                    self._update_offset = update["update_id"] + 1
                    message = update.get("message", {})
                    text = (message.get("text") or "").strip()
                    chat_id = str(message.get("chat", {}).get("id", ""))
                    if chat_id != TELEGRAM_CHAT_ID:
                        continue
                    await self._handle_command(text.lower(), bot_ref)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.warning("telegram poll error: %s", exc)
                await asyncio.sleep(5)

    async def _handle_command(self, text: str, bot_ref: Any) -> None:
        if text.startswith("/silence"):
            parts = text.split()
            hours = float(parts[1]) if len(parts) > 1 else 4.0
            self._silenced_until = time.time() + hours * 3600
            await self.send(f"🔇 *Silence mode* enabled for {hours:g}h",
                            "silence", force=True)
        elif text.startswith("/status"):
            await self.send(bot_ref.status_summary(), "status", force=True)
        elif text.startswith("/halt"):
            await self.send("🛑 *HALT requested* — no new campaigns will open.",
                            "halt", critical=True)
            bot_ref.halt_new_campaigns = True

    @staticmethod
    def fmt_campaign_launch(c: Dict[str, Any]) -> str:
        t = c["tiers"]
        return (
            f"🚀 *Campaign Launched*\n"
            f"Symbol: `{c['symbol']}` | Direction: *{c['direction']}*\n"
            f"Class: *{c['conviction_class']}* \\(conf {c['stage2_confidence']:.2f}\\)\n"
            f"Range: `{c['fib_anchor']['low']}` → `{c['fib_anchor']['high']}`\n"
            f"Tiers: `{t[0]['trigger_price']}` / `{t[1]['trigger_price']}`"
            + (f" / `{t[2]['trigger_price']}`" if len(t) > 2 else "") + "\n"
            f"TP: `{c['take_profit']['price']}` \\({int(CONFIG['FIB_TP_EXTENSION'] * 100)}% ext\\)\n"
            f"SL: `{c['stop_loss']['current']}`\n"
            f"Margin: `${sum(x['margin_usd'] for x in t):.2f}`")

    @staticmethod
    def fmt_tier_fill(c: Dict[str, Any], tier: Dict[str, Any]) -> str:
        return (
            f"📈 *Tier {tier['tier']} Filled — {c['symbol']}*\n"
            f"Price: `{tier.get('fill_price', tier['trigger_price'])}`"
            f" | Lev: `{tier['leverage']}x`\n"
            f"Notional: `${tier['notional_usd']:.2f}`\n"
            f"SL → `{c['stop_loss']['current']}`")

    @staticmethod
    def fmt_sl_ratchet(c: Dict[str, Any], old_sl: float, new_sl: float) -> str:
        return (
            f"🔒 *SL Ratcheted — {c['symbol']}*\n"
            f"From `{old_sl}` → `{new_sl}`\n"
            f"State: `{c['stop_loss']['state']}`")

    @staticmethod
    def fmt_campaign_close(c: Dict[str, Any], net: float, wallet: float,
                           dd: float, reason: str) -> str:
        sign = "+" if net >= 0 else ""
        pct = net / max(c.get("deployed_margin", 1.0), 0.01) * 100
        icon = "✅" if net >= 0 else "🛑"
        word = "Closed" if net >= 0 else "Aborted"
        return (
            f"{icon} *Campaign {word} — {c['symbol']}*\n"
            f"Exit: {md_escape(reason)}\n"
            f"Net: `{sign}{net:.2f} USDT` \\({sign}{pct:.1f}%\\)\n"
            f"Wallet: `${wallet:.2f}` | DD: `{dd * 100:.1f}%`")

    @staticmethod
    def fmt_guardrail(name: str, detail: str) -> str:
        return f"⚠️ *Guardrail {name}*\n{md_escape(detail)}"

    @staticmethod
    def fmt_drawdown_halt(dd: float, wallet: float, peak: float) -> str:
        return (
            f"🚨 *DRAWDOWN HALT*\n"
            f"Drawdown: `{dd * 100:.1f}%` \\(threshold "
            f"{CONFIG['DRAWDOWN_HALT_24H_PCT'] * 100:.0f}%\\)\n"
            f"New campaigns paused 24h\\.\n"
            f"Wallet: `${wallet:.2f}` | Peak: `${peak:.2f}`")

    @staticmethod
    def fmt_critical(module: str, detail: str, action: str) -> str:
        return (f"🔥 *CRITICAL ERROR*\nModule: `{module}`\n"
                f"Detail: {md_escape(detail)}\nAction: {md_escape(action)}")


# =====================================================================
# SECTION 5 — STATEMANAGER (ATOMIC PERSISTENCE)
# =====================================================================

SCHEMA_VERSION = 1


def fresh_state() -> Dict[str, Any]:
    """Return the canonical empty state object."""
    return {
        "schema_version": SCHEMA_VERSION,
        "last_updated": iso_now(),
        "wallet": {"balance_usdt": 0.0, "reserved_capital": 0.0,
                   "peak_balance": 0.0, "drawdown_pct": 0.0},
        "active_campaigns": [],
        "pending_tier_orders": [],
        "closed_campaigns_recent": [],
        "performance": {"completed_campaigns": 0, "wins": 0, "losses": 0,
                        "win_rate": 0.0, "max_drawdown_pct": 0.0,
                        "avg_win_pct": 0.0, "avg_loss_pct": 0.0,
                        "realized_pnl_today": 0.0, "fees_today": 0.0},
        "meta": {"base_unit": 2.00, "mode": "MICRO",
                 "compounding_gate_passed": False,
                 "last_scan_ts": None, "last_reconcile_ts": None,
                 "drawdown_halt_until": None},
    }


class StateManager:
    """In-memory state + atomic JSON persistence with rotating backups."""

    def __init__(self, path: str = CONFIG["STATE_FILE"]) -> None:
        self.path = path
        self.lock = FileLock(CONFIG["LOCK_FILE"])
        self.state: Dict[str, Any] = fresh_state()
        self._persist_lock = asyncio.Lock()

    def acquire_lock(self) -> None:
        """Acquire the instance lock; raises if another process holds it."""
        self.lock.acquire()

    def release_lock(self) -> None:
        """Release the instance lock."""
        self.lock.release()

    def _walk_backups(self) -> Optional[Dict[str, Any]]:
        for i in range(1, CONFIG["STATE_BACKUP_COUNT"] + 1):
            candidate = f"{self.path}.bak.{i}"
            if not os.path.exists(candidate):
                continue
            try:
                with open(candidate, encoding="utf-8") as fh:
                    data = json.load(fh)
                log.warning("recovered state from %s", candidate)
                return data
            except (json.JSONDecodeError, OSError):
                continue
        return None

    def load(self) -> Dict[str, Any]:
        """Load state from disk; fall back to backups, then fresh state."""
        if os.path.exists(self.path):
            try:
                with open(self.path, encoding="utf-8") as fh:
                    data = json.load(fh)
                if data.get("schema_version") != SCHEMA_VERSION:
                    raise ValueError("schema_version mismatch")
                self.state = data
                log.info("state loaded from %s", self.path)
                return self.state
            except (json.JSONDecodeError, ValueError, OSError) as exc:
                log.critical("state file corrupt (%s); walking backups", exc)
                recovered = self._walk_backups()
                self.state = recovered if recovered else fresh_state()
                if recovered is None:
                    log.critical("no usable backup; starting FRESH state")
                return self.state
        log.info("no state file; starting fresh")
        self.state = fresh_state()
        return self.state

    async def persist(self) -> None:
        """Atomically persist current state (serialized through a lock)."""
        async with self._persist_lock:
            self.state["last_updated"] = iso_now()
            try:
                atomic_write_json(self.path, self.state,
                                  backup_count=CONFIG["STATE_BACKUP_COUNT"])
            except OSError as exc:
                log.critical("state persist failed (disk full?): %s", exc)
                raise

    def campaigns(self) -> List[Dict[str, Any]]:
        return self.state["active_campaigns"]

    def get_campaign(self, campaign_id: str) -> Optional[Dict[str, Any]]:
        for c in self.state["active_campaigns"]:
            if c["campaign_id"] == campaign_id:
                return c
        return None

    def get_campaign_by_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        for c in self.state["active_campaigns"]:
            if c["symbol"] == symbol:
                return c
        return None

    def add_campaign(self, campaign: Dict[str, Any]) -> None:
        self.state["active_campaigns"].append(campaign)

    def archive_campaign(self, campaign: Dict[str, Any]) -> None:
        """Move a campaign from active to the recent-closed audit ring."""
        self.state["active_campaigns"] = [
            c for c in self.state["active_campaigns"]
            if c["campaign_id"] != campaign["campaign_id"]]
        self.state["closed_campaigns_recent"].append(campaign)
        self.state["closed_campaigns_recent"] = \
            self.state["closed_campaigns_recent"][-50:]

    def pending_orders(self) -> List[Dict[str, Any]]:
        return self.state["pending_tier_orders"]

    def get_pending_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        for o in self.state["pending_tier_orders"]:
            if o["order_id"] == order_id:
                return o
        return None

    def remove_pending_order(self, order_id: str) -> None:
        self.state["pending_tier_orders"] = [
            o for o in self.state["pending_tier_orders"] if o["order_id"] != order_id]


# =====================================================================
# SECTION 6 — STAGE 1: MARKET SCANNER
# =====================================================================


class MarketScanner:
    """Stage 1: universe fetch -> hard filters -> correlation -> scoring -> rank."""

    def __init__(self, client: BybitClient) -> None:
        self.client = client
        self._correlation_matrix: Dict[Tuple[str, str], float] = {}
        self._correlation_ts: float = 0.0
        self._cached_universe: Optional[Dict[str, Any]] = None
        self._cached_universe_ts: float = 0.0

    async def _layer1_universe(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Fetch tickers + market metadata; fall back to cache on failure."""
        try:
            tickers, markets = await asyncio.gather(
                self.client.fetch_tickers(), self.client.fetch_markets_meta())
            self._cached_universe = tickers
            self._cached_universe_ts = time.time()
            return tickers, markets
        except Exception as exc:  # noqa: BLE001
            age_h = (time.time() - self._cached_universe_ts) / 3600
            if self._cached_universe is not None and age_h <= 4:
                log.warning("universe fetch failed; using cache (%.1fh old): %s",
                            age_h, exc)
                markets = await self.client.fetch_markets_meta()
                return self._cached_universe, markets
            log.critical("universe fetch failed with no usable cache: %s", exc)
            raise

    def _layer2_hard_filters(self, tickers: Dict[str, Any],
                             markets: Dict[str, Any]) -> Tuple[List[Dict[str, Any]],
                                                               Dict[str, int]]:
        """Apply the nine binary gates; return survivors + rejection counts.

        Symbols are already normalized to raw Bybit form (BTCUSDT) by
        BybitClient.fetch_tickers(), so no ':' check is needed here.
        """
        rejected: Dict[str, int] = {k: 0 for k in (
            "status_not_trading", "listing_too_new", "turnover_below_floor",
            "spread_too_wide", "funding_extreme", "volatility_out_of_band",
            "stablecoin_or_wrapped", "unknown_symbol", "not_usdt")}
        survivors: List[Dict[str, Any]] = []
        stables = set(CONFIG["STABLECOIN_SYMBOLS"])
        now_ms = int(time.time() * 1000)
        for symbol, t in tickers.items():
            if not symbol.endswith("USDT"):
                rejected["not_usdt"] += 1
                continue
            market = markets.get(symbol)
            if not market:
                rejected["unknown_symbol"] += 1
                continue
            info = market.get("info", {})
            if (info.get("status") or "Trading") != "Trading":
                rejected["status_not_trading"] += 1
                continue
            launch = int(info.get("launchTime") or 0)
            if launch and (now_ms - launch) / 86400000 < CONFIG["MIN_LISTING_AGE_DAYS"]:
                rejected["listing_too_new"] += 1
                continue
            if symbol in stables or market.get("base") in ("USDC", "USDE", "DAI",
                                                           "FDUSD", "TUSD", "EUR"):
                rejected["stablecoin_or_wrapped"] += 1
                continue
            turnover = float(t.get("quoteVolume") or 0.0)
            if turnover < CONFIG["MIN_TURNOVER_24H"]:
                rejected["turnover_below_floor"] += 1
                continue
            bid, ask = float(t.get("bid") or 0), float(t.get("ask") or 0)
            mid = (bid + ask) / 2 if bid and ask else float(t.get("last") or 0)
            spread_bps = ((ask - bid) / mid * 1e4) if mid and bid and ask else 1e9
            if spread_bps > CONFIG["MAX_SPREAD_BPS"]:
                rejected["spread_too_wide"] += 1
                continue
            funding = abs(float(t.get("info", {}).get("fundingRate") or 0.0))
            if funding > CONFIG["MAX_FUNDING_ABS"]:
                rejected["funding_extreme"] += 1
                continue
            survivors.append({
                "symbol": symbol,
                "ccxt_symbol": t.get("_ccxt_symbol",
                                     self.client.ccxt_symbol(symbol)),
                "turnover_24h": turnover,
                "spread_bps": spread_bps,
                "funding_rate_8h": funding,
                "last_price": float(t.get("last") or mid),
                "high_24h": float(t.get("high") or mid),
                "low_24h": float(t.get("low") or mid),
            })
        return survivors, rejected

    async def _layer3_correlation(self, survivors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Refresh correlation matrix if stale; keep top-ranked per cluster."""
        age_h = (time.time() - self._correlation_ts) / 3600
        if age_h > CONFIG["CORRELATION_REFRESH_HOURS"] or not self._correlation_matrix:
            await self._rebuild_correlation(survivors)
        for s in survivors:
            s["cluster_id"] = 0
        cluster_next = 1
        kept: List[Dict[str, Any]] = []
        for s in survivors:
            s["_pre_rank"] = s["turnover_24h"]
        ordered = sorted(survivors, key=lambda x: -x["_pre_rank"])
        for s in ordered:
            placed = False
            for k in kept:
                corr = self._correlation_matrix.get(
                    tuple(sorted((s["symbol"], k["symbol"]))), 0.0)
                if corr > CONFIG["CORRELATION_THRESHOLD"]:
                    s["cluster_id"] = k["cluster_id"]
                    placed = True
                    break
            if not placed:
                s["cluster_id"] = cluster_next
                cluster_next += 1
                kept.append(s)
        for s in survivors:
            s.pop("_pre_rank", None)
        kept_symbols = {s["symbol"] for s in kept}
        return [s for s in survivors if s["symbol"] in kept_symbols]

    async def _rebuild_correlation(self, survivors: List[Dict[str, Any]]) -> None:
        """Fetch daily klines and rebuild the pairwise correlation matrix."""
        sem = asyncio.Semaphore(8)

        async def _returns(symbol: str) -> Optional[pd.Series]:
            async with sem:
                try:
                    df = await self.client.fetch_klines(
                        symbol, "1d", CONFIG["CORRELATION_LOOKBACK_DAYS"] + 1)
                    if len(df) < CONFIG["CORRELATION_LOOKBACK_DAYS"] // 2:
                        return None
                    return df["close"].pct_change().dropna()
                except Exception as exc:  # noqa: BLE001
                    log.warning("correlation klines failed for %s: %s", symbol, exc)
                    return None

        symbols = [s["symbol"] for s in survivors[:60]]
        series_map = dict(zip(symbols, await asyncio.gather(*[_returns(s) for s in symbols])))
        matrix: Dict[Tuple[str, str], float] = {}
        for i, a in enumerate(symbols):
            sa = series_map.get(a)
            if sa is None or len(sa) < 5:
                continue
            for b in symbols[i + 1:]:
                sb = series_map.get(b)
                if sb is None or len(sb) < 5:
                    continue
                joined = pd.concat([sa, sb], axis=1, keys=["a", "b"]).dropna()
                if len(joined) < 5:
                    continue
                corr = float(joined["a"].corr(joined["b"]))
                if not np.isnan(corr):
                    matrix[tuple(sorted((a, b)))] = corr
        self._correlation_matrix = matrix
        self._correlation_ts = time.time()
        log.info("correlation matrix rebuilt: %d pairs", len(matrix))

    def _layer4_score(self, survivors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score survivors on seven weighted components; attach metrics; rank."""
        if not survivors:
            return []
        turn = np.array([s["turnover_24h"] for s in survivors])
        sprd = np.array([s["spread_bps"] for s in survivors])
        fund = np.array([s["funding_rate_8h"] for s in survivors])
        atrp = np.array([s.get("atr_daily_pct", 1.0) for s in survivors])
        tq = np.array([s.get("trend_quality", 0.5) for s in survivors])
        depth = np.array([s.get("depth_1pct_usd", 0.0) for s in survivors])

        def norm(x: np.ndarray) -> np.ndarray:
            lo, hi = float(np.min(x)), float(np.max(x))
            return (x - lo) / (hi - lo + 1e-12)

        w = CONFIG["SCORE_WEIGHTS"]
        vol_center = (CONFIG["MIN_ATR_DAILY_PCT"] + CONFIG["MAX_ATR_DAILY_PCT"]) / 2
        vol_fit = 1.0 - np.minimum(np.abs(atrp - vol_center) / vol_center, 1.0)
        corr_pen = np.ones(len(survivors))
        score = (w["liquidity"] * norm(np.log1p(turn))
                 + w["spread"] * (1.0 - norm(sprd))
                 + w["depth"] * norm(np.log1p(depth))
                 + w["volatility"] * vol_fit
                 + w["funding"] * (1.0 - norm(fund))
                 + w["trend"] * tq
                 + w["correlation"] * corr_pen)
        for s, sc in zip(survivors, score):
            s["composite_score"] = round(float(sc), 4)
        ranked = sorted(survivors, key=lambda x: -x["composite_score"])
        for i, s in enumerate(ranked, start=1):
            s["rank"] = i
            s["metrics"] = {
                "turnover_24h": s["turnover_24h"],
                "spread_bps": round(s["spread_bps"], 2),
                "depth_1pct_usd": s.get("depth_1pct_usd", 0.0),
                "atr14_daily_pct": round(float(s.get("atr_daily_pct", 0.0)), 2),
                "funding_rate_8h": s["funding_rate_8h"],
                "trend_quality": round(float(s.get("trend_quality", 0.5)), 3),
                "cluster_id": s["cluster_id"],
            }
        return ranked

    async def scan(self) -> Dict[str, Any]:
        """Run the full Stage 1 pipeline and return the scan result object."""
        tickers, markets = await self._layer1_universe()
        survivors, rejected = self._layer2_hard_filters(tickers, markets)
        log.info("Stage1: %d raw -> %d after hard filters",
                 len(tickers), len(survivors))

        sem = asyncio.Semaphore(8)

        async def _enrich(s: Dict[str, Any]) -> None:
            async with sem:
                try:
                    df = await self.client.fetch_klines(
                        s["symbol"], "1d", CONFIG["ATR_PERIOD"] + 5)
                    atr = compute_atr(df, CONFIG["ATR_PERIOD"]).iloc[-1]
                    s["atr_daily_pct"] = float(atr / s["last_price"] * 100)
                    closes = df["close"]
                    mom = (closes.iloc[-1] - closes.iloc[-6]) / closes.iloc[-6]
                    s["trend_quality"] = float(np.clip(0.5 + mom * 5, 0.0, 1.0))
                except Exception:  # noqa: BLE001
                    s["atr_daily_pct"] = 0.0
                    s["trend_quality"] = 0.5
                if not (CONFIG["MIN_ATR_DAILY_PCT"] <= s["atr_daily_pct"]
                        <= CONFIG["MAX_ATR_DAILY_PCT"]):
                    s["_vol_reject"] = True
                try:
                    ob = await self.client.fetch_order_book(s["symbol"], 50)
                    depth = 0.0
                    for side_key, px_key in (("bids", 0), ("asks", 0)):
                        for level in ob.get(side_key, []):
                            px = float(level[px_key])
                            if abs(px - s["last_price"]) / s["last_price"] <= 0.01:
                                depth += float(level[1]) * px
                    s["depth_1pct_usd"] = depth
                    if depth < CONFIG["MIN_DEPTH_1PCT_USD"]:
                        s["_depth_reject"] = True
                except Exception:  # noqa: BLE001
                    s["depth_1pct_usd"] = s["turnover_24h"] * 0.001
                    s["_depth_reject"] = False

        await asyncio.gather(*[_enrich(s) for s in survivors])
        vol_rej = sum(1 for s in survivors if s.pop("_vol_reject", False))
        dep_rej = sum(1 for s in survivors if s.pop("_depth_reject", False))
        rejected["volatility_out_of_band"] = vol_rej
        rejected["depth_too_thin"] = dep_rej
        survivors = [s for s in survivors
                     if CONFIG["MIN_ATR_DAILY_PCT"] <= s["atr_daily_pct"]
                     <= CONFIG["MAX_ATR_DAILY_PCT"]
                     and s["depth_1pct_usd"] >= CONFIG["MIN_DEPTH_1PCT_USD"]]

        decorrelated = await self._layer3_correlation(survivors)
        ranked = self._layer4_score(decorrelated)
        result = {
            "scan_timestamp": iso_now(),
            "universe_size_raw": len(tickers),
            "universe_size_after_hard_filters": len(survivors),
            "universe_size_after_correlation": len(ranked),
            "candidates": ranked[:25],
            "rejected_summary": {k: v for k, v in rejected.items() if v},
            "health": {
                "api_latency_ms": round(self.client.last_latency_ms, 1),
                "correlation_matrix_age_min": round(
                    (time.time() - self._correlation_ts) / 60, 1),
            },
        }
        top = ", ".join(f"{c['symbol']}({c['composite_score']:.2f})"
                        for c in ranked[:5])
        log.info("Stage1 top-5: %s", top or "none")
        log.info("Stage1 rejection breakdown: %s", result["rejected_summary"])
        return result


# =====================================================================
# SECTION 7 — STAGE 2: REGIME DETECTION (1H)
# =====================================================================


class RegimeDetector:
    """Stage 2: five-layer gate classifying candidates TREND/RANGE/AMBIGUOUS."""

    def __init__(self, client: BybitClient) -> None:
        self.client = client

    async def _fetch_validated(self, symbol: str,
                               timeframe: str) -> Optional[pd.DataFrame]:
        """Fetch klines and validate freshness/integrity/gaps."""
        try:
            df = await self.client.fetch_klines(symbol, timeframe,
                                                CONFIG["KLINES_LIMIT"])
        except Exception as exc:  # noqa: BLE001
            log.warning("Stage2 kline fetch failed %s %s: %s", symbol, timeframe, exc)
            return None
        if len(df) < CONFIG["MIN_CANDLES_REQUIRED"]:
            log.info("%s: insufficient_history (%d candles)", symbol, len(df))
            return None
        age_s = (now_utc() - df["ts"].iloc[-1]).total_seconds()
        if age_s > CONFIG["MAX_KLINE_AGE_SECONDS"] + 3600:
            log.info("%s: stale klines (%.0fs old)", symbol, age_s)
            return None
        if not ((df["high"] >= np.maximum(df["open"], df["close"])) &
                (df["low"] <= np.minimum(df["open"], df["close"]))).all():
            log.info("%s: OHLC integrity violation", symbol)
            return None
        gap = df["ts"].diff().dt.total_seconds()
        expected = 3600 if timeframe == "1h" else 14400
        if (gap > expected * (CONFIG["MAX_KLINE_GAP_CANDLES"] + 1)).any():
            log.info("%s: data_gap", symbol)
            return None
        return df

    def _classify(self, symbol: str, df: pd.DataFrame,
                  htf_aligned: Optional[bool]) -> Dict[str, Any]:
        """Core classification logic applying Donchian, Volume, ADX, Persistence, HTF."""
        invalidators: List[str] = []
        upper, lower = donchian(df, CONFIG["DONCHIAN_PERIOD"])
        close = float(df["close"].iloc[-1])
        vol_ma = float(df["volume"].iloc[-CONFIG["VOLUME_MA_PERIOD"] - 1:-1].mean())
        curr_vol = float(df["volume"].iloc[-1])
        vol_ratio = curr_vol / (vol_ma + 1e-12)
        adx_series = compute_adx(df, CONFIG["ADX_PERIOD"])
        adx = float(adx_series.iloc[-1]) if not adx_series.empty else 0.0
        atr = float(compute_atr(df, CONFIG["ATR_PERIOD"]).iloc[-1])

        break_up = close > upper
        break_dn = close < lower
        rng = max(upper - lower, 1e-12)

        evidence = {
            "last_close": close,
            "donchian_upper": upper,
            "donchian_lower": lower,
            "donchian_high": upper,
            "donchian_low": lower,
            "volume_ratio": round(vol_ratio, 2),
            "adx_14": round(adx, 2),
            "atr_abs": round(atr, 6),
            "atr_pct": round(atr / max(close, 1e-12) * 100, 3),
        }

        if not (break_up or break_dn):
            return self._finish(symbol, "RANGE", None, 0.0, evidence,
                                ["no_breakout"], htf_aligned, df, atr=atr)

        direction = "LONG" if break_up else "SHORT"
        outside = sum(1 for i in range(1, 6)
                      if (float(df["close"].iloc[-i]) > upper if break_up
                          else float(df["close"].iloc[-i]) < lower))
        evidence["closes_outside"] = outside

        breakout_idx = None
        for i in range(1, min(len(df), CONFIG["MAX_BREAKOUT_AGE"] + 5)):
            c = float(df["close"].iloc[-i])
            if (break_up and c <= upper) or (break_dn and c >= lower):
                breakout_idx = i - 1
                break
        age = breakout_idx if breakout_idx is not None else 0
        evidence["breakout_age_candles"] = age

        vol_spike = vol_ratio >= CONFIG["VOLUME_SPIKE_MULT"]
        age_ok = age <= CONFIG["MAX_BREAKOUT_AGE"]

        if not vol_spike:
            invalidators.append("no_volume_spike")
        if outside < CONFIG["MIN_CLOSES_OUTSIDE"]:
            invalidators.append("insufficient_persistence")
        if not age_ok:
            invalidators.append("breakout_stale")
        if adx < CONFIG["ADX_MIN_WEAK"]:
            invalidators.append("low_adx")

        w = CONFIG["CONF_WEIGHTS"]
        break_strength = float(np.clip(
            abs(close - (upper if break_up else lower)) / rng + 0.5, 0.0, 1.0))
        vol_score = float(np.clip(evidence["volume_ratio"] /
                                  (CONFIG["VOLUME_SPIKE_MULT"] * 2), 0.0, 1.0))
        adx_score = float(np.clip(adx / 40.0, 0.0, 1.0))
        persist_score = float(np.clip(outside / CONFIG["MIN_CLOSES_OUTSIDE"],
                                      0.0, 1.0))
        if htf_aligned is True:
            htf_score, evidence["higher_tf_alignment"] = 1.0, True
        elif htf_aligned is False:
            htf_score = 0.0
            evidence["higher_tf_alignment"] = False
            invalidators.append("htf_misaligned")
        else:
            htf_score, evidence["higher_tf_alignment"] = 0.3, False

        confidence = (w["donchian"] * break_strength + w["volume"] * vol_score
                      + w["adx"] * adx_score + w["persistence"] * persist_score
                      + w["htf"] * htf_score)
        if htf_aligned is None:
            confidence = min(confidence, CONFIG["HTF_CONFIDENCE_CAP_NO_DATA"])
        elif not htf_aligned:
            confidence -= CONFIG["HTF_CONFIDENCE_PENALTY"]
        confidence = round(float(np.clip(confidence, 0.0, 1.0)), 4)

        quality_hits = int(adx >= CONFIG["ADX_MIN_STRONG"]) + \
            int(outside >= CONFIG["MIN_CLOSES_OUTSIDE"]) + int(age_ok)
        if quality_hits == 3 and confidence >= CONFIG["CONF_TREND_MIN"]:
            regime = "TREND"
        elif confidence >= CONFIG["CONF_TREND_MIN"] and quality_hits == 2:
            regime = "TREND"
        elif confidence >= CONFIG["CONF_AMBIG_MIN"]:
            regime = "AMBIGUOUS"
        else:
            regime = "RANGE"
        if not vol_spike and regime == "TREND":
            regime = "AMBIGUOUS"
            invalidators.append("no_volume_spike")
        return self._finish(symbol, regime, direction if regime != "RANGE" else None,
                            confidence, evidence, invalidators, htf_aligned, df,
                            atr=atr)

    def _finish(self, symbol: str, regime: str, direction: Optional[str],
                confidence: float, evidence: Dict[str, Any],
                invalidators: List[str], htf_aligned: Optional[bool],
                df: pd.DataFrame, atr: Optional[float] = None) -> Dict[str, Any]:
        advance = regime == "TREND" and confidence >= CONFIG["CONF_TREND_MIN"]
        if atr is None:
            atr = float(compute_atr(df, CONFIG["ATR_PERIOD"]).iloc[-1])
            evidence.setdefault("atr_pct", round(atr / evidence["last_close"] * 100, 3))
            evidence.setdefault("atr_abs", round(atr, 6))
        return {
            "symbol": symbol, "regime": regime, "direction": direction,
            "confidence": confidence, "evidence": evidence,
            "invalidators": invalidators,
            "advance_to_stage3": advance,
            "watch": regime == "AMBIGUOUS",
        }

    async def _htf_alignment(self, symbol: str, direction: str) -> Optional[bool]:
        """Layer 4: 4H Donchian alignment. None if data unavailable."""
        try:
            df = await self.client.fetch_klines(
                symbol, CONFIG["HTF_INTERVAL"], CONFIG["DONCHIAN_PERIOD"] + 5)
            if len(df) < CONFIG["DONCHIAN_PERIOD"]:
                return None
            upper, lower = donchian(df, CONFIG["DONCHIAN_PERIOD"])
            close = float(df["close"].iloc[-1])
            if direction == "LONG":
                return close > upper
            return close < lower
        except Exception as exc:  # noqa: BLE001
            log.warning("HTF fetch failed %s: %s", symbol, exc)
            return None

    async def evaluate(self, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run Stage 2 over all Stage 1 candidates."""
        regimes: List[Dict[str, Any]] = []
        sem = asyncio.Semaphore(6)

        async def _one(cand: Dict[str, Any]) -> Dict[str, Any]:
            symbol = cand["symbol"]
            async with sem:
                df = await self._fetch_validated(symbol, "1h")
                if df is None:
                    return {"symbol": symbol, "regime": "RANGE", "direction": None,
                            "confidence": 0.0, "evidence": {},
                            "invalidators": ["data_invalid"],
                            "advance_to_stage3": False, "watch": False}
                upper, lower = donchian(df, CONFIG["DONCHIAN_PERIOD"])
                close = float(df["close"].iloc[-1])
                if close > upper:
                    direction = "LONG"
                elif close < lower:
                    direction = "SHORT"
                else:
                    direction = "LONG"
                htf = await self._htf_alignment(symbol, direction) \
                    if close > upper or close < lower else None
                return self._classify(symbol, df, htf)

        regimes = list(await asyncio.gather(*[_one(c) for c in candidates]))
        for r in regimes:
            for i, c in enumerate(candidates, start=1):
                if c["symbol"] == r["symbol"]:
                    r["rank_from_stage1"] = i
                    break
        summary = {
            "scan_timestamp": iso_now(),
            "candidates_in": len(candidates),
            "trend": sum(1 for r in regimes if r["regime"] == "TREND"),
            "range": sum(1 for r in regimes if r["regime"] == "RANGE"),
            "ambiguous": sum(1 for r in regimes if r["regime"] == "AMBIGUOUS"),
        }
        advanced = [r for r in regimes if r.get("advance_to_stage3")]
        log.info("Stage2: %d in / %d TREND / %d RANGE / %d AMBIGUOUS",
                 summary["candidates_in"], summary["trend"], summary["range"],
                 summary["ambiguous"])
        return {"summary": summary, "regimes": regimes, "advanced": advanced}


# =====================================================================
# SECTION 8 — STAGE 3: CASCADING EXECUTION MATRIX
# =====================================================================


def mode_from_balance(balance: float) -> str:
    """Map wallet balance to campaign mode (MICRO/MID/MATURE)."""
    if balance >= CONFIG["MODE_MATURE_MIN"]:
        return "MATURE"
    if balance > CONFIG["MODE_MID_MAX"]:
        return "MATURE"
    if balance >= CONFIG["MODE_MICRO_MAX"]:
        return "MID"
    return "MICRO"


def aggression_for_mode(mode: str) -> float:
    """Return the sizing aggression factor for the mode."""
    return {"MICRO": CONFIG["AGGRESSION_FACTOR_MICRO"],
            "MID": CONFIG["AGGRESSION_FACTOR_MID"],
            "MATURE": CONFIG["AGGRESSION_FACTOR_MATURE"]}[mode]


def fib_ratios_for_mode(mode: str) -> List[float]:
    """Return the Fibonacci tier ratios for the mode."""
    if mode == "MICRO":
        return list(CONFIG["FIB_RATIOS_3TIER"])
    if mode == "MID":
        return list(CONFIG["FIB_RATIOS_4TIER"])
    return list(CONFIG["FIB_RATIOS_5TIER"])


def tier_weights_for_count(n: int) -> List[float]:
    """Return tier weights; first 3 shared between 3/5-tier configs."""
    if n <= 3:
        return list(CONFIG["TIER_WEIGHTS_3"])
    return list(CONFIG["TIER_WEIGHTS_5"])


def conviction_class(confidence: float) -> str:
    """Classify a Stage 2 confidence into STANDARD/STRONG/ELITE."""
    for cls, (lo, hi) in CONFIG["CONFIDENCE_CLASS_BANDS"].items():
        if lo <= confidence <= hi:
            return cls
    return "STANDARD" if confidence < 0.80 else "STRONG"


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value into [lo, hi]."""
    return max(lo, min(hi, value))


class CascadePlanner:
    """Stage 3: converts a TREND candidate into a full cascade campaign plan."""

    def __init__(self, client: BybitClient, ledger: Any = None) -> None:
        self.client = client
        self.ledger = ledger

    def _anchors(self, direction: str, df: pd.DataFrame,
                 evidence: Dict[str, Any]) -> Optional[Tuple[float, float]]:
        """Pick swing low/high anchors; validate compression bounds.

        For LONG, anchor_high is ALWAYS the Donchian high (the breakout
        level price just broke). For SHORT, anchor_low is ALWAYS the
        Donchian low. The opposite anchor comes from the 50-candle swing.
        """
        lookback = df.iloc[-CONFIG["FIB_ANCHOR_LOOKBACK"]:]
        swing_low = float(lookback["low"].min())
        swing_high = float(lookback["high"].max())
        dc_low = float(evidence.get("donchian_low") or swing_low)
        dc_high = float(evidence.get("donchian_high") or swing_high)
        close = float(evidence["last_close"])
        atr_abs = float(evidence["atr_abs"])
        if direction == "LONG":
            anchor_low = swing_low
            anchor_high = dc_high
        else:
            anchor_low = dc_low
            anchor_high = swing_high
        if anchor_high <= anchor_low:
            log.info("anchor ordering invalid: low=%.6f high=%.6f",
                     anchor_low, anchor_high)
            return None
        rng = anchor_high - anchor_low
        base = max(close, 1e-12)
        if atr_abs > 0 and rng < CONFIG["FIB_MIN_RANGE_ATR_MULT"] * atr_abs:
            log.info("fib_range_too_small: %.4f < %.4f", rng,
                     CONFIG["FIB_MIN_RANGE_ATR_MULT"] * atr_abs)
            return None
        if rng / base > CONFIG["FIB_MAX_RANGE_PCT"]:
            log.info("fib_range_too_wide: %.2f%%", rng / base * 100)
            return None
        return anchor_low, anchor_high

    def _brackets(self, direction: str, anchor_low: float,
                  anchor_high: float, mode: str) -> Tuple[List[float], float]:
        """Compute tier trigger prices + TP from the anchor range."""
        rng = anchor_high - anchor_low
        if direction == "LONG":
            triggers = [anchor_high + r * rng for r in fib_ratios_for_mode(mode)]
            tp = anchor_high + CONFIG["FIB_TP_EXTENSION"] * rng
        else:
            triggers = [anchor_low - r * rng for r in fib_ratios_for_mode(mode)]
            tp = anchor_low - CONFIG["FIB_TP_EXTENSION"] * rng
        return triggers, tp

    def _size(self, direction: str, confidence: float, triggers: List[float],
              wallet: float, mode: str, max_lev: float,
              atr_abs: float, last_close: float) -> List[Dict[str, Any]]:
        """Size each tier: conviction-weighted notional, leverage peaks at T2."""
        cls = conviction_class(confidence)
        camp_mult = CONFIG["CAMPAIGN_MULTIPLIERS"][cls]
        lev_mult = CONFIG["CONVICTION_LEVERAGE_MULT"][cls]
        weights = tier_weights_for_count(len(triggers))
        aggression = aggression_for_mode(mode)
        zone = CONFIG["ZONE_HALF_WIDTH_ATR_MULT"] * atr_abs
        tiers: List[Dict[str, Any]] = []
        for i, (trig, w) in enumerate(zip(triggers, weights)):
            base_lev = CONFIG["LEVERAGE_LADDER"][i] * lev_mult
            lev = min(base_lev, max_lev)
            notional = CONFIG["MIN_NOTIONAL_FLOOR"] + \
                wallet * aggression * w * camp_mult
            margin = notional / lev
            tiers.append({
                "tier": i + 1,
                "fib_ratio": fib_ratios_for_mode(mode)[i],
                "trigger_price": round(trig, 8),
                "zone_width": round(zone, 8),
                "leverage": int(round(lev)),
                "notional_usd": round(notional, 4),
                "margin_usd": round(margin, 4),
                "qty": 0.0, "status": "PENDING",
            })
        return [t for t in tiers
                if t["notional_usd"] >= CONFIG["MIN_NOTIONAL_FLOOR"] * 0.99]

    def _sl_plan(self, direction: str, anchor_low: float, anchor_high: float,
                 triggers: List[float], atr_abs: float, cls: str,
                 last_close: float) -> Dict[str, Any]:
        """Build the initial SL + ratchet table."""
        buffer = CONFIG["SL_BUFFER_ATR_MULT"] * atr_abs
        cap = (CONFIG["SL_CAP_ELITE_PCT"] if cls == "ELITE"
               else CONFIG["SL_CAP_STANDARD_PCT"])
        if direction == "LONG":
            initial = anchor_low - buffer
            max_sl = last_close * (1 - cap)
            initial = max(initial, max_sl)
            table = {0: initial, 1: anchor_low}
            for i, trig in enumerate(triggers[:-1], start=2):
                table[i] = max(trig - buffer, max_sl)
        else:
            initial = anchor_high + buffer
            max_sl = last_close * (1 + cap)
            initial = min(initial, max_sl)
            table = {0: initial, 1: anchor_high}
            for i, trig in enumerate(triggers[:-1], start=2):
                table[i] = min(trig + buffer, max_sl)
        return {"current": round(initial, 8), "state": "INITIAL",
                "ratchet_table": {str(k): round(v, 8) for k, v in table.items()},
                "cap_pct": cap,
                "buffer": round(buffer, 8),
                "next_ratchet_on": "Tier 1 fill"}

    def _guardrail_prechecks(self, regime: Dict[str, Any],
                             active_campaigns: List[Dict[str, Any]],
                             wallet: float, deployed_margin: float,
                             total_tier_margin: float) -> Tuple[bool, Optional[str]]:
        """Portfolio-level gates before a campaign may be created."""
        if len(active_campaigns) >= CONFIG["MAX_CONCURRENT_CAMPAIGNS"]:
            return False, "max_concurrent_campaigns"
        if deployed_margin + total_tier_margin > wallet * CONFIG["MAX_MARGIN_DEPLOYED_PCT"]:
            return False, "margin_deployed_pct"
        peak = max(wallet, 1e-9)
        dd = 0.0
        if self.ledger is not None:
            peak = max(self.ledger.state.state["wallet"].get("peak_balance", wallet), 1e-9)
            dd = (peak - wallet) / peak
        if dd > CONFIG["DRAWDOWN_HALT_24H_PCT"]:
            return False, "drawdown_halt"
        sym = regime["symbol"]
        if self.ledger is not None and hasattr(self.ledger, "scanner"):
            for c in active_campaigns:
                corr = self.ledger.scanner_corr(sym, c["symbol"])  # type: ignore[attr-defined]
                if corr is not None and corr > CONFIG["CORRELATION_THRESHOLD"]:
                    return False, "correlation_conflict"
        return True, None

    async def build_campaign(self, regime: Dict[str, Any], wallet: float,
                             active_campaigns: List[Dict[str, Any]],
                             deployed_margin: float) -> Optional[Dict[str, Any]]:
        """Build a full campaign object from a TREND regime verdict."""
        symbol = regime["symbol"]
        direction = regime["direction"] or "LONG"
        evidence = regime["evidence"]
        confidence = float(regime["confidence"])
        mode = mode_from_balance(wallet)
        try:
            df = await self.client.fetch_klines(
                symbol, "1h", CONFIG["FIB_ANCHOR_LOOKBACK"] + 10)
        except Exception as exc:  # noqa: BLE001
            log.warning("campaign build: klines failed %s: %s", symbol, exc)
            return None
        anchors = self._anchors(direction, df, evidence)
        if anchors is None:
            return None
        anchor_low, anchor_high = anchors
        triggers, tp = self._brackets(direction, anchor_low, anchor_high, mode)
        atr_abs = float(evidence["atr_abs"])
        last_close = float(evidence["last_close"])
        max_lev = self.client.max_leverage(symbol)
        tiers = self._size(direction, confidence, triggers, wallet, mode,
                           max_lev, atr_abs, last_close)
        if not tiers:
            log.info("%s: all tiers below minimum notional", symbol)
            return None
        total_margin = sum(t["margin_usd"] for t in tiers)
        allowed, reason = self._guardrail_prechecks(
            regime, active_campaigns, wallet, deployed_margin, total_margin)
        if not allowed:
            log.info("%s: campaign rejected (%s)", symbol, reason)
            return None
        cls = conviction_class(confidence)
        campaign_id = (f"{symbol.replace('USDT', '')}-"
                       f"{now_utc():%Y-%m-%d-%H%M}-{uuid.uuid4().hex[:4]}")
        sl = self._sl_plan(direction, anchor_low, anchor_high, triggers,
                           atr_abs, cls, last_close)
        campaign = {
            "campaign_id": campaign_id,
            "symbol": symbol,
            "direction": direction,
            "conviction_class": cls,
            "stage2_confidence": confidence,
            "mode": mode,
            "fib_anchor": {"low": round(anchor_low, 8),
                           "high": round(anchor_high, 8),
                           "range": round(anchor_high - anchor_low, 8)},
            "tiers": tiers,
            "stop_loss": sl,
            "take_profit": {"price": round(tp, 8),
                            "fib_ratio": CONFIG["FIB_TP_EXTENSION"],
                            "type": "FIB_EXTENSION",
                            "order_id": None},
            "state": "ARMED",
            "frozen": False,
            "deployed_margin": round(total_margin, 4),
            "highest_filled_tier": 0,
            "entry_time": iso_now(),
            "last_updated": iso_now(),
            "sl_order_id": None,
            "tp_order_id": None,
            "bracket_entry_candle": 0,
            "partial_exits": [],
        }
        log.info("Stage3: campaign %s %s %s (%s, conf %.2f, %d tiers, margin $%.2f)",
                 campaign_id, symbol, direction, cls, confidence, len(tiers),
                 total_margin)
        return campaign

    @staticmethod
    def ratchet_sl(campaign: Dict[str, Any], filled_tier: int) -> Optional[float]:
        """Return the new SL after `filled_tier` fills, or None if unchanged."""
        table = campaign["stop_loss"]["ratchet_table"]
        key = str(filled_tier)
        if key not in table:
            return None
        return float(table[key])


# =====================================================================
# SECTION 9 — STAGE 4: PORTFOLIO LEDGER & CAPITAL QUEUE
# =====================================================================


class PortfolioLedger:
    """Stage 4.4: margin, P&L, capital deployment, and portfolio risk gates."""

    def __init__(self, state: StateManager, client: BybitClient) -> None:
        self.state = state
        self.client = client
        self.scanner: Optional[MarketScanner] = None

    def scanner_corr(self, a: str, b: str) -> Optional[float]:
        """Return the cached correlation between two symbols, if known."""
        if self.scanner is None:
            return None
        return self.scanner._correlation_matrix.get(tuple(sorted((a, b))))  # noqa: SLF001

    async def refresh_wallet(self) -> Dict[str, Any]:
        """Pull balance from Bybit, update wallet block + drawdown tracking."""
        w = self.state.state["wallet"]
        balance = await self.client.fetch_total_usdt()
        if balance <= 0:
            return w
        used = 0.0
        unreal = 0.0
        try:
            for p in await self.client.fetch_positions():
                unreal += float(p.get("unrealizedPnl") or 0)
                used += float(p.get("initialMargin") or 0)
        except Exception as exc:  # noqa: BLE001
            log.warning("position refresh in ledger failed: %s", exc)
        w["balance_usdt"] = round(balance, 4)
        w["used_margin"] = round(used, 4)
        w["unrealized_pnl"] = round(unreal, 4)
        w["free_margin"] = round(max(balance - used, 0.0), 4)
        peak = max(float(w.get("peak_balance") or 0), balance)
        w["peak_balance"] = round(peak, 4)
        w["drawdown_pct"] = round((peak - balance) / peak if peak else 0.0, 6)
        perf = self.state.state["performance"]
        if w["drawdown_pct"] > float(perf.get("max_drawdown_pct") or 0):
            perf["max_drawdown_pct"] = w["drawdown_pct"]
        meta = self.state.state["meta"]
        meta["mode"] = mode_from_balance(balance)
        self._evaluate_compounding_gate()
        self._update_base_unit()
        return w

    def _evaluate_compounding_gate(self) -> None:
        """Guardrail J: raise base unit only after 20 campaigns, 45% WR, DD<15%."""
        perf = self.state.state["performance"]
        completed = int(perf.get("completed_campaigns") or 0)
        wr = float(perf.get("win_rate") or 0)
        dd = float(perf.get("max_drawdown_pct") or 0)
        passed = (completed >= CONFIG["COMPOUNDING_MIN_CAMPAIGNS"]
                  and wr >= CONFIG["COMPOUNDING_MIN_WINRATE"]
                  and dd < CONFIG["COMPOUNDING_MAX_DD_PCT"])
        self.state.state["meta"]["compounding_gate_passed"] = passed

    def _update_base_unit(self) -> None:
        """Recompute base_unit from mode/balance, respecting the compounding gate."""
        meta = self.state.state["meta"]
        balance = float(self.state.state["wallet"].get("balance_usdt") or 0)
        mode = meta["mode"]
        if mode == "MICRO":
            new_unit = 2.00 if balance < 100 else 3.00
        elif mode == "MID":
            new_unit = 4.00 if balance < 250 else 5.00
        else:
            new_unit = round(balance * 0.015, 2)
        if not meta.get("compounding_gate_passed"):
            new_unit = min(new_unit, float(meta.get("base_unit") or new_unit))
        meta["base_unit"] = new_unit

    def deployed_margin(self) -> float:
        """Sum of margins across active campaigns."""
        return round(sum(float(c.get("deployed_margin") or 0)
                         for c in self.state.campaigns()), 4)

    async def check_portfolio_gates(self, new_margin: float) -> Tuple[bool, Optional[str]]:
        """Pre-campaign guardrail gate (drawdown halt, concurrency, margin)."""
        meta = self.state.state["meta"]
        halt_until = meta.get("drawdown_halt_until")
        if halt_until and now_utc().isoformat() < halt_until:
            return False, "drawdown_halt_active"
        w = self.state.state["wallet"]
        balance = float(w.get("balance_usdt") or 0)
        if len(self.state.campaigns()) >= CONFIG["MAX_CONCURRENT_CAMPAIGNS"]:
            return False, "max_concurrent_campaigns"
        if (self.deployed_margin() + new_margin
                > balance * CONFIG["MAX_MARGIN_DEPLOYED_PCT"]):
            return False, "margin_deployed_pct"
        peak = max(float(w.get("peak_balance") or balance), 1e-9)
        dd = (peak - balance) / peak
        if dd > CONFIG["DRAWDOWN_HALT_24H_PCT"]:
            from datetime import timedelta
            meta["drawdown_halt_until"] = (
                now_utc() + timedelta(hours=24)).isoformat()
            return False, "drawdown_halt"
        return True, None

    def attribute_close(self, campaign: Dict[str, Any], net_pnl: float,
                        fees: float, exit_reason: str) -> Dict[str, float]:
        """Record a closed campaign's P&L into performance stats."""
        perf = self.state.state["performance"]
        perf["completed_campaigns"] = int(perf.get("completed_campaigns") or 0) + 1
        margin = max(float(campaign.get("deployed_margin") or 1), 1e-9)
        pct = net_pnl / margin
        if net_pnl >= 0:
            perf["wins"] = int(perf.get("wins") or 0) + 1
            wins, losses = int(perf["wins"]), int(perf.get("losses") or 0)
            avg_win = float(perf.get("avg_win_pct") or 0)
            perf["avg_win_pct"] = round((avg_win * (wins - 1) + pct) / wins, 6)
        else:
            perf["losses"] = int(perf.get("losses") or 0) + 1
            wins, losses = int(perf["wins"] or 0), int(perf["losses"])
            avg_loss = float(perf.get("avg_loss_pct") or 0)
            perf["avg_loss_pct"] = round(
                (avg_loss * (losses - 1) + pct) / losses, 6)
        total = wins + losses
        perf["win_rate"] = round(wins / total, 4) if total else 0.0
        perf["realized_pnl_today"] = round(
            float(perf.get("realized_pnl_today") or 0) + net_pnl, 4)
        perf["fees_today"] = round(
            float(perf.get("fees_today") or 0) + fees, 4)
        self._evaluate_compounding_gate()
        self._update_base_unit()
        return self.state.state["wallet"]

    def daily_reset(self) -> None:
        """Reset per-day counters at 00:00 UTC."""
        perf = self.state.state["performance"]
        perf["realized_pnl_today"] = 0.0
        perf["fees_today"] = 0.0
        log.info("daily P&L counters reset")


class CapitalQueue:
    """Stage 4.5: queue salary top-ups; deploy when all campaigns close."""

    def __init__(self, state: StateManager, ledger: PortfolioLedger,
                 notifier: Notifier) -> None:
        self.state = state
        self.ledger = ledger
        self.notifier = notifier
        self._last_seen_balance = 0.0

    async def detect_topups(self) -> None:
        """Compare exchange balance to last seen; queue unexplained increases."""
        if not CONFIG["TOPUP_QUEUE_ENABLED"]:
            return
        balance = await self.client_balance()
        if balance <= 0:
            return
        wallet = self.state.state["wallet"]
        if self._last_seen_balance > 0:
            expected = self._last_seen_balance + \
                float(self.state.state["performance"].get("realized_pnl_today") or 0)
            delta = balance - expected
            if delta > 1.0:
                await self._on_topup(delta, balance)
        self._last_seen_balance = balance

    async def client_balance(self) -> float:
        """Current exchange equity (via ledger's client)."""
        return await self.ledger.client.fetch_total_usdt()

    async def _on_topup(self, amount: float, balance: float) -> None:
        wallet = self.state.state["wallet"]
        if len(self.state.campaigns()) == 0:
            wallet["reserved_capital"] = round(
                float(wallet.get("reserved_capital") or 0), 4)
            wallet["balance_usdt"] = round(balance, 4)
            self.ledger._update_base_unit()  # noqa: SLF001
            await self.state.persist()
            await self.notifier.send(
                f"💵 *Top-up deployed* immediately\nAmount: `${amount:.2f}`\n"
                f"New balance: `${balance:.2f}`", "topup", force=False)
        else:
            wallet["reserved_capital"] = round(
                float(wallet.get("reserved_capital") or 0) + amount, 4)
            await self.state.persist()
            await self.notifier.send(
                f"💵 *Top-up queued*\nAmount: `${amount:.2f}` queued while "
                f"{len(self.state.campaigns())} campaign(s) active\\.\n"
                f"Will deploy after all campaigns close\\.", "topup", force=False)
        self._last_seen_balance = balance

    async def deploy_if_idle(self) -> None:
        """Deploy queued capital once no campaigns are active."""
        if not CONFIG["DEPLOY_ON_ALL_CLOSED"]:
            return
        wallet = self.state.state["wallet"]
        reserved = float(wallet.get("reserved_capital") or 0)
        if reserved <= 0 or len(self.state.campaigns()) > 0:
            return
        balance = await self.client_balance()
        wallet["reserved_capital"] = 0.0
        wallet["balance_usdt"] = round(balance, 4)
        self.ledger._update_base_unit()  # noqa: SLF001
        await self.state.persist()
        meta = self.state.state["meta"]
        await self.notifier.send(
            f"💵 *Queued capital deployed*\nNew balance: `${balance:.2f}`\n"
            f"Base unit: `${meta['base_unit']:.2f}`", "topup_deploy", force=False)


# =====================================================================
# SECTION 10 — STAGE 4: EXECUTION ENGINE
# =====================================================================


class ExecutionEngine:
    """Stage 4.2: order placement & lifecycle for cascade campaigns."""

    def __init__(self, client: BybitClient, state: StateManager,
                 notifier: Notifier, ledger: PortfolioLedger) -> None:
        self.client = client
        self.state = state
        self.notifier = notifier
        self.ledger = ledger
        self._order_attempts: Dict[str, int] = {}

    def _cid(self, campaign_id: str, tier: int, kind: str = "entry") -> str:
        key = f"{campaign_id}-{kind}-{tier}"
        attempt = self._order_attempts.get(key, 0) + 1
        self._order_attempts[key] = attempt
        return CONFIG["CLIENT_ORDER_ID_FORMAT"].format(
            campaign_id=campaign_id, tier=tier, attempt=attempt)

    async def arm_campaign(self, campaign: Dict[str, Any]) -> bool:
        """Place Tier 1 order + register campaign as active."""
        tier1 = campaign["tiers"][0]
        side = "buy" if campaign["direction"] == "LONG" else "sell"
        ok, order = await self._place_entry(campaign, tier1, side)
        if not ok:
            return False
        campaign["state"] = "ACTIVE"
        campaign["tiers"][0]["status"] = "PLACED"
        campaign["tiers"][0]["order_id"] = order["id"]
        campaign["tiers"][0]["client_order_id"] = order.get("clientOrderId")
        self.state.add_campaign(campaign)
        self.state.pending_orders().append({
            "order_id": order["id"],
            "client_order_id": campaign["tiers"][0].get("client_order_id"),
            "campaign_id": campaign["campaign_id"], "tier": 1,
            "type": "PostOnlyLimit", "price": tier1["trigger_price"],
            "qty": tier1["qty"], "status": "OPEN", "side": side,
        })
        await self.state.persist()
        await self.notifier.send(
            self.notifier.fmt_campaign_launch(campaign),
            "campaign_launch", campaign["campaign_id"])
        return True

    async def _place_entry(self, campaign: Dict[str, Any], tier: Dict[str, Any],
                           side: str) -> Tuple[bool, Dict[str, Any]]:
        symbol = campaign["symbol"]
        price = self.client.round_price(symbol, tier["trigger_price"])
        raw_qty = tier["notional_usd"] / max(price, 1e-12)
        qty = self.client.round_qty(symbol, raw_qty)
        tier["qty"] = qty
        cid = self._cid(campaign["campaign_id"], tier["tier"], "entry")
        params = {"timeInForce": "PostOnly", "orderLinkId": cid}
        try:
            order = await self.client.place_order(
                symbol, side, "limit", qty, price, params)
            return True, order
        except Exception as exc:  # noqa: BLE001
            log.error("place_entry tier %d failed %s: %s", tier["tier"], symbol, exc)
            return False, {}