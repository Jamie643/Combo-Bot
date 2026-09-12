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
# SECTION 2 — UTILITIES: LOGGING, TIME, RETRY, ATOMIC IO, INDICATORS
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
    high, low = df["high"], df["low"]
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

    # ---- Symbol conversion helpers ----

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

    # ---- Fetch methods: ccxt in, internal-friendly out ----

    @retry_async()
    async def fetch_tickers(self) -> Dict[str, Any]:
        """Fetch linear USDT-perp tickers, KEYED BY RAW BYBIT SYMBOL (BTCUSDT).

        Each ticker carries '_ccxt_symbol' so downstream code can call
        the exchange without re-deriving it.
        """
        raw = await self._guard(self.exchange.fetch_tickers())
        normalized: Dict[str, Any] = {}
        for sym, ticker in raw.items():
            if not sym.endswith(":USDT"):
                continue  # only linear USDT perps
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
        """Fetch all non-zero linear positions, KEYED in each dict by raw symbol.

        Injects 'symbol_raw' (BTCUSDT) alongside ccxt's 'symbol' (BTC/USDT:USDT)
        so callers can match against internal state regardless of form.
        """
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

    # ---- Precision / limits (accept raw or ccxt symbol) ----

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
    """Private WebSocket (aiohttp) subscribing to `order` and `position` streams.

    Normalizes each event's 'symbol' to raw Bybit form so the orchestrator
    can match against campaign state without extra conversion.
    """

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
                async with session.post(url,