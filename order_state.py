"""Durable order state + SQLite journal. Never guesses on unknown outcomes."""
import aiosqlite
import time
import uuid

DB_PATH = "trade_journal.db"

async def init_db():
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
        await db.commit()

def new_client_id(symbol: str) -> str:
    return f"cb-{symbol[:6]}-{uuid.uuid4().hex[:12]}"

async def record_intent(client_id, symbol, side, qty):
    now = time.time()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT OR REPLACE INTO orders "
            "(client_id,symbol,side,qty_requested,status,created_at,updated_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (client_id, symbol, side, qty, "PENDING", now, now))
        await db.commit()

async def mark_unknown(client_id):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE orders SET status='UNKNOWN', updated_at=? WHERE client_id=?",
            (time.time(), client_id))
        await db.commit()

async def resolve(client_id, venue_id, qty_filled, status):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE orders SET venue_id=?, qty_filled=?, status=?, updated_at=? "
            "WHERE client_id=?",
            (venue_id, qty_filled, status, time.time(), client_id))
        await db.commit()
