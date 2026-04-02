from fastapi import FastAPI

from app.routers.vision import vision_router

app = FastAPI(
    title="LeafJoctors Leaf Saving Service", description="Central Leaf Saving API"
)


@app.get("/")
def get_root():
    return {"message": "Hello from main"}


# registers router endpoints
app.include_router(vision_router)


def start_server():
    """
    Launch the API server with Uvicorn
    """
    import uvicorn

    uvicorn.run("app.server:app", host="localhost", port=8000, reload=True)
