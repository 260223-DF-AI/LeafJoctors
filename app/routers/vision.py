"""
FastAPI Router that handles functionality
related to SageMaker & our leaf vision model
"""

from fastapi import APIRouter

vision_router = APIRouter(prefix="/vision", tags=["vision"])


@vision_router.get("/")
async def sample_endpoint():
    """
    Sample endpoint
    """
    pass


