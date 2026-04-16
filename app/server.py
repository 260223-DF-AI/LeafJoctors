from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers.agromonitoring import agro_router

app = FastAPI(
    title="LeafJoctors Leaf Saving Service", description="Central Leaf Saving API"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Optional root route (prevents 404 on "/")
@app.get("/")
def get_root():
    return {"message": "LeafJoctors API is running"}


# Register routers
app.include_router(agro_router)


def start_server():
    import uvicorn

    uvicorn.run("app.server:app", host="localhost", port=8000, reload=True)
