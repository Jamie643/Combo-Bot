"""Runtime bootstrap: uvloop install + deadman heartbeat.
Import this BEFORE any asyncio work in bot.py.
"""
import os
import time

try:
    import uvloop
    uvloop.install()
    UVLOOP_ACTIVE = True
except Exception:
    UVLOOP_ACTIVE = False

_LAST_HEARTBEAT = time.monotonic()

def heartbeat():
    global _LAST_HEARTBEAT
    _LAST_HEARTBEAT = time.monotonic()

def seconds_since_heartbeat() -> float:
    return time.monotonic() - _LAST_HEARTBEAT

DEADMAN_TIMEOUT = int(os.getenv("DEADMAN_TIMEOUT", "900"))
