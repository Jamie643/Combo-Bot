import os
import time
import requests
import ccxt

# ==========================================
# PROXY UTILITY FOR BYBIT REGION RESTRICTION
# ==========================================

def get_working_proxy():
    """
    Fetches public HTTPS/HTTP proxies and tests them against Bybit's API 
    until a non-blocked, working proxy is found.
    """
    api_url = "https://api.proxyscrape.com/v2/?request=displayproxies&protocol=http&timeout=3000&country=all&ssl=yes"
    
    print("Fetching free proxy list to bypass geo-restrictions...")
    try:
        response = requests.get(api_url, timeout=10)
        if response.status_code == 200:
            proxies = [p.strip() for p in response.text.strip().split('\r\n') if p.strip()]
            print(f"Retrieved {len(proxies)} proxies to test.")
            
            # Test up to the first 10 proxies to find an operational non-US connection
            for proxy in proxies[:10]:
                formatted_proxy = f"http://{proxy}"
                test_params = {
                    'enableRateLimit': True,
                    'proxies': {
                        'http': formatted_proxy,
                        'https': formatted_proxy,
                    },
                    'timeout': 5000
                }
                
                try:
                    test_exchange = ccxt.bybit(test_params)
                    test_exchange.fetch_time()
                    print(f"Successfully connected to Bybit via proxy: {formatted_proxy}")
                    return formatted_proxy
                except Exception as e:
                    print(f"Proxy {formatted_proxy} failed test: {e}")
                    continue
    except Exception as e:
        print(f"Failed to fetch proxy list: {e}")
        
    print("Warning: No working proxy found. Proceeding with direct connection...")
    return None

# ==========================================
# MAIN EXECUTION SCRIPT
# ==========================================

def run_bot():
    # Fetch working proxy
    proxy_url = get_working_proxy()
    
    # Configure CCXT Exchange Instance
    exchange_params = {
        'apiKey': os.getenv('BYBIT_API_KEY', ''),
        'secret': os.getenv('BYBIT_API_SECRET', ''),
        'enableRateLimit': True,
        'options': {
            'defaultType': 'linear',  # Perpetual contract / Futures market
        }
    }
    
    if proxy_url:
        exchange_params['proxies'] = {
            'http': proxy_url,
            'https': proxy_url,
        }
    
    exchange = ccxt.bybit(exchange_params)
    
    # Optional: Enable Testnet if specified in environment
    if os.getenv('USE_TESTNET', 'true').lower() == 'true':
        exchange.set_sandbox_mode(True)
        print("Running in Bybit Testnet mode.")

    # Core Execution Logic / Market Checks
    try:
        print("Checking connection and fetching server time...")
        server_time = exchange.fetch_time()
        print(f"Bybit Server Time: {server_time}")
        
        # Example market lookup
        symbol = 'BTC/USDT:USDT'
        ticker = exchange.fetch_ticker(symbol)
        print(f"Successfully fetched ticker for {symbol} | Last Price: {ticker['last']}")
        
        # --- Add your indicators, market scanning, or strategy logic here ---
        
        # Telegram Notification Helper
        telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        if telegram_token and telegram_chat_id:
            msg = f"<b>Bot Single Run Success</b>\nSymbol: {symbol}\nLast Price: {ticker['last']}"
            tele_url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
            payload = {
                'chat_id': telegram_chat_id,
                'text': msg,
                'parse_mode': 'HTML'
            }
            res = requests.post(tele_url, data=payload, timeout=10)
            if res.status_code == 200:
                print("Telegram notification sent successfully.")
            else:
                print(f"Failed to send Telegram alert: {res.text}")

    except ccxt.BaseError as e:
        print(f"CCXT Exchange Error encountered: {e}")
    except Exception as e:
        print(f"Unexpected error during execution: {e}")

if __name__ == "__main__":
    run_bot()