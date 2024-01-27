from fastapi import FastAPI

from app.routers.signature import lifespan
from app.routers.signature import router as extract_router

app = FastAPI(lifespan=lifespan)


@app.get("/")
async def home():
    return {"Welcome to Signature Verification Project!👋"}


app.include_router(extract_router, prefix="/signature", tags=["Signature Verification Endpoint"])
