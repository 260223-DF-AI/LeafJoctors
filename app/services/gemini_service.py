import asyncio
import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)


def build_prompt(soil: dict, weather: dict, prediction: dict) -> str:
    temp_k = weather["main"]["temp"]
    humidity = weather["main"]["humidity"]
    condition = weather["weather"][0]["description"]

    temp_c = temp_k - 273.15

    return f"""
You are an agricultural expert AI.

--- Leaf Disease Prediction ---
{prediction}

--- Soil Data ---
Moisture: {soil.get("moisture")}
Surface Temp (K): {soil.get("t0")}
10cm Temp (K): {soil.get("t10")}

--- Weather Data ---
Temperature (C): {temp_c:.2f}
Humidity: {humidity}%
Condition: {condition}

--- Task ---
1. Diagnose plant health
2. Suggest treatments
3. Recommend watering/fertilizer adjustments
4. Warn about risks

Keep response concise. Do not format your response with markdown.
Instead, use line breaks to format in plaintext.
"""


async def get_response(prompt: str) -> str:
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: client.models.generate_content(
            model="gemini-3-flash-preview", contents=prompt
        ),
    )
    return response.text
