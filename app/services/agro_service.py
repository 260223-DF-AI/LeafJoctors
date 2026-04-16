import os

import httpx
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("AGROMONITORING_API_KEY")
BASE_URL = "http://api.agromonitoring.com/agro/1.0"


# Helper: fetch polygons
async def fetch_polygons():
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/polygons", params={"appid": API_KEY})
        response.raise_for_status()
        return response.json()


# Get soil data based on polygon id
async def get_soil(poly_id: str):
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{BASE_URL}/soil", params={"appid": API_KEY, "polyid": poly_id}
        )
        response.raise_for_status()
        return response.json()


# Get weather data using a polygon's center (by polygon ID)
async def get_weather_by_polygon(poly_id: str):
    polygons = await fetch_polygons()

    polygon = next((p for p in polygons if p["id"] == poly_id), None)
    if not polygon:
        raise Exception("Polygon not found")

    lon, lat = polygon["center"]

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{BASE_URL}/weather", params={"appid": API_KEY, "lat": lat, "lon": lon}
        )
        response.raise_for_status()
        return response.json()
