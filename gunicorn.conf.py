import threading


def on_starting(server):
    # Import the main loop function from your bot.py
    from bot import run_market_loop

    # Spawn it as a background thread alongside Gunicorn
    t = threading.Thread(target=run_market_loop, daemon=True)
    t.start()
    print("🚀 Background market loop & Telegram pulse engine started!")