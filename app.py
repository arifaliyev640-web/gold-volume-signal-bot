import os
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET")


def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    data = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }

    response = requests.post(url, json=data, timeout=10)
    response.raise_for_status()


@app.get("/")
def home():
    return "Gold Signal Bot is running!"


@app.post("/webhook/<secret>")
def webhook(secret):
    if secret != WEBHOOK_SECRET:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    side = data.get("side", "UNKNOWN")
    symbol = data.get("symbol", "XAUUSD")
    timeframe = data.get("timeframe", "5")

    entry = data.get("entry")
    sl = data.get("sl")
    tp1 = data.get("tp1")
    tp2 = data.get("tp2")

    poc = data.get("poc")
    vah = data.get("vah")
    val = data.get("val")
    vwap = data.get("vwap")
    volume_ratio = data.get("volume_ratio")

    message = (
        f"🟡 XAU/USD SIGNAL\n\n"
        f"📊 {symbol} | {timeframe}M\n"
        f"📈 SIGNAL: {side}\n\n"
        f"🎯 Entry: {entry}\n"
        f"🛑 SL: {sl}\n"
        f"✅ TP1: {tp1}\n"
        f"🚀 TP2: {tp2}\n\n"
        f"📍 POC: {poc}\n"
        f"🔴 VAH: {vah}\n"
        f"🟢 VAL: {val}\n"
        f"🟠 VWAP: {vwap}\n"
        f"📊 Volume Ratio: {volume_ratio}"
    )

    try:
        send_telegram_message(message)
    except Exception as e:
        return jsonify({
            "error": "Telegram send failed",
            "details": str(e)
        }), 500

    return jsonify({
        "status": "success",
        "signal": side
    })


if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
