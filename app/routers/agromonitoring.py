import asyncio
import os

from dotenv import load_dotenv
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.agro_service import fetch_polygons, get_soil, get_weather_by_polygon
from app.services.gemini_service import build_prompt, get_response
from sage.predict import make_prediction

agro_router = APIRouter(prefix="/agromonitoring", tags=["agromonitoring"])

load_dotenv()

API_KEY = os.getenv("AGROMONITORING_API_KEY")
BASE_URL = "http://api.agromonitoring.com/agro/1.0"
ALLOWED_LOCATIONS = {"chicago": "Chicago", "india": "India", "bangladesh": "Bangladesh"}


async def _resolve_polygon_by_location(location: str):
    normalized = location.strip().lower()
    canonical_location = ALLOWED_LOCATIONS.get(normalized)
    if not canonical_location:
        raise HTTPException(
            status_code=400,
            detail="Location must be one of: Chicago, India, Bangladesh.",
        )

    polygons = await fetch_polygons()
    polygon = next(
        (p for p in polygons if str(p.get("name", "")).strip().lower() == normalized),
        None,
    )
    if not polygon:
        raise HTTPException(
            status_code=404,
            detail=f"No polygon found for location '{canonical_location}'.",
        )

    return canonical_location, polygon


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
@agro_router.post("/analyze/{location}")
async def analyze(location: str, file: UploadFile = File(...)):
    try:
        canonical_location, polygon = await _resolve_polygon_by_location(location)
        poly_id = polygon["id"]

        # read image
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # sagemaker prediction
        try:
            prediction = await asyncio.to_thread(
                make_prediction,
                image_bytes,
                content_type=file.content_type,
                endpoint_name="pytorch-inference-2026-04-16-15-51-52-235",
            )
        except ValueError as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
        except Exception as e:
            raise HTTPException(
                status_code=502,
                detail=f"SageMaker inference failed: {e}",
            ) from e

        # soil/weather  data
        soil = await get_soil(poly_id)
        weather = await get_weather_by_polygon(poly_id)

        # get gemini stuff
        prompt = build_prompt(soil, weather, prediction)
        analysis = await get_response(prompt)

        return {
            "location": canonical_location,
            "polygon_id": poly_id,
            "prediction": prediction,
            "analysis": analysis,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
