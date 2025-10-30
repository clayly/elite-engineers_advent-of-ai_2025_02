#!/usr/bin/env python3
import os
import asyncio
import telegram
from dotenv import load_dotenv

load_dotenv()

async def test_telegram():
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    if not bot_token or not chat_id:
        print("❌ TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not found in environment")
        print("Please set them in your .env file")
        return False
    
    try:
        bot = telegram.Bot(bot_token)
        async with bot:
            test_message = "🤖 Test message from scheduled task - Telegram integration works!"
            await bot.send_message(text=test_message, chat_id=chat_id)
        print("✅ Message sent to Telegram successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to send message to Telegram: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_telegram())
