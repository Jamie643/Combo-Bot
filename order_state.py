"""Durable order state + SQLite journal. Never guesses on unknown outcomes.

All writes are wrapped so a journal failure never interrupts trading. If the
journal is disabled or the DB is unwritable, the bot keeps running and only
loses post-hoc analysis capability.
"""
import logging
import os
import time
import uuid

try:
    import aiosqlite
    AIOSQLITE_AVAILABLE = True
except ImportError:
    AIOSQLITE_AVAILABLE = False

log = logging.getLogger("cascade.journal")

DB_PATH = "trade_journal.db"
_ENABLED = os.getenv("JOURNAL_ENABLED", "true").lower() == "true"


def _active() -> bool:
    return _ENABLED and AIOSQLITE_AVAILABLE


async def init_db() -> str:
    """Initialise schema. Returns 'ok', 'disabled', or raises on DB error."""
    if not _active():
        log.info("journal disabled (JOURNAL_ENABLED=%s, aiosqlite=%s)",
                 _ENABLED, AIOSQLITE_AVAILABLE)
        return "disabled"
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                client_id TEXT PRIMARY KEY,
                venue_id TEXT,
                symbol TEXT,
                side TEXT,
                qty_requested REAL,
                qty_filled REAL DEFAULT 0,
                status TEXT,
                created_at REAL,
                updated_at REAL
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS fills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                client_id TEXT,
                price REAL,
                qty REAL,
                fee REAL,
                ts REAL
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS campaigns (
                campaign_id TEXT PRIMARY KEY,
                symbol TEXT,
                direction TEXT,
                regime TEXT,
                conviction_class TEXT,
                stage2_confidence REAL,
                mode TEXT,
                fib_low REAL,
                fib_high REAL,
                fib_range REAL,
                n_tiers INT,
                n_filled INT,
                entry_time TEXT,
                close_time TEXT,
                exit_reason TEXT,
                net_pnl REAL,
                fees REAL,
                deployed_margin REAL,
                holding_hours REAL,
                adx_at_entry REAL,
                pullback_depth REAL,
                payload_json TEXT
            )
        """)
        await db.commit()
    return "ok"


def new_client_id(symbol: str) -> str:
    return f"cb-{symbol[:6]}-{uuid.uuid4().hex[:12]}"


async def record_intent(client_id, symbol, side, qty):
    if not _active():
        return
    now = time.time()
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                "INSERT OR REPLACE INTO orders "
                "(client_id,symbol,side,qty_requested,status,created_at,updated_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (client_id, symbol, side, qty, "PENDING", now, now))
            await db.commit()
    except Exception as exc:
        log.warning("record_intent failed: %s", exc)


async def mark_unknown(client_id):
    if not _active():
        return
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                "UPDATE orders SET status='UNKNOWN', updated_at=? WHERE client_id=?",
                (time.time(), client_id))
            await db.commit()
    except Exception as exc:
        log.warning("mark_unknown failed: %s", exc)


async def resolve(client_id, venue_id, qty_filled, status):
    if not _active():
        return
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                "UPDATE orders SET venue_id=?, qty_filled=?, status=?, "
                "updated_at=? WHERE client_id=?",
                (venue_id, qty_filled, status, time.time(), client_id))
            await db.commit()
    except Exception as exc:
        log.warning("resolve failed: %s", exc)


async def record_campaign(campaign: dict, net_pnl: float, fees: float,
                          exit_reason: str) -> None:
    """Persist a closed campaign for later analysis."""
    if not _active():
        return
    try:
        import json as _json
        from datetime import datetime
        n_filled = sum(1 for t in campaign.get("tiers", [])
                       if t.get("status") == "FILLED")
        entry_time = campaign.get("entry_time", "")
        close_time = campaign.get("last_updated", "")
        try:
            e = datetime.fromisoformat(entry_time.replace("Z", "+00:00"))
            c = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
            holding = (c - e).total_seconds() / 3600
        except Exception:
            holding = 0.0
        anchor = campaign.get("fib_anchor", {})
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                "INSERT OR REPLACE INTO campaigns "
                "(campaign_id,symbol,direction,regime,conviction_class,"
                "stage2_confidence,mode,fib_low,fib_high,fib_range,"
                "n_tiers,n_filled,entry_time,close_time,exit_reason,"
                "net_pnl,fees,deployed_margin,holding_hours,"
                "adx_at_entry,pullback_depth,payload_json) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (campaign.get("campaign_id", ""),
                 campaign.get("symbol", ""),
                 campaign.get("direction", ""),
                 campaign.get("regime", "TREND"),
                 campaign.get("conviction_class", ""),
                 campaign.get("stage2_confidence", 0.0),
                 campaign.get("mode", ""),
                 anchor.get("low", 0.0),
                 anchor.get("high", 0.0),
                 anchor.get("range", 0.0),
                 len(campaign.get("tiers", [])),
                 n_filled,
                 entry_time,
                 close_time,
                 exit_reason,
                 net_pnl,
                 fees,
                 campaign.get("deployed_margin", 0.0),
                 holding,
                 campaign.get("adx_at_entry", 0.0),
                 campaign.get("pullback_depth", 0.0),
                 _json.dumps(campaign, default=str)))
            await db.commit()
    except Exception as exc:
        log.warning("record_campaign failed: %s", exc)
