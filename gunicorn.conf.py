import threading

def on_starting(server):
    from bot import bot_loop
    
    t = threading.Thread(target=bot_loop, daemon=True)
    t.start()
    print("🚀 Background market loop & Telegram pulse engine started!")
