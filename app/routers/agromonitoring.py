import httpx
from fastapi import HTTPException, APIRouter, UploadFile, File

from app.services.agro_service import (
    fetch_polygons,
    get_soil,
    get_weather_by_polygon
)
from app.services.gemini_service import build_prompt, get_response

agro_router = APIRouter(prefix="/agromonitoring", tags=["agromonitoring"])

API_KEY = "261c2e50a80208dee799b287b0428c80"
BASE_URL = "http://api.agromonitoring.com/agro/1.0"


# -----------------------------
# Debug: polygons
# -----------------------------
@agro_router.get("/polygons")
async def polygon_list():
    try:
        polygons = await fetch_polygons()
        return {
            "count": len(polygons),
            "polygons": polygons
        }
    except httpx.HTTPError as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------
# Soil
# -----------------------------
@agro_router.get("/soil/{poly_id}")
async def soil(poly_id: str):
    try:
        data = await get_soil(poly_id)
        return {"polygon_id": poly_id, "soil": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------
# Weather
# -----------------------------
@agro_router.get("/weather/{poly_id}")
async def weather(poly_id: str):
    try:
        data = await get_weather_by_polygon(poly_id)
        return {"polygon_id": poly_id, "weather": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------
# Debug: polygons
# -----------------------------
@agro_router.get("/polygons")
async def polygon_list():
    try:
        polygons = await fetch_polygons()
        return {"count": len(polygons), "polygons": polygons}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------
# Soil
# -----------------------------
@agro_router.get("/soil/{poly_id}")
async def soil(poly_id: str):
    try:
        data = await get_soil(poly_id)
        return {"polygon_id": poly_id, "soil": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------
# Weather
# -----------------------------
@agro_router.get("/weather/{poly_id}")
async def weather(poly_id: str):
    try:
        data = await get_weather_by_polygon(poly_id)
        return {"polygon_id": poly_id, "weather": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------
# FULL ANALYSIS
# -----------------------------
@agro_router.post("/analyze/{poly_id}")
async def analyze(poly_id: str, file: UploadFile = File(...)):
    try:
        # 1. Read image
        # image_bytes = await file.read()

        # 2. SageMaker prediction
        # prediction = predict(image_bytes, file.content_type)
        prediction = ""

        # 3. External data
        soil = await get_soil(poly_id)
        weather = await get_weather_by_polygon(poly_id)

        # 4. Build prompt
        prompt = build_prompt(soil, weather, prediction)

        # 5. Gemini
        analysis = await get_response(prompt)

        return {
            "polygon_id": poly_id,
            "prediction": prediction,
            "analysis": analysis
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
