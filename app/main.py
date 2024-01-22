from fastapi import FastAPI, Path
from app.routers.extract import router as extract_router, lifespan

app = FastAPI(lifespan=lifespan)


@app.get("/")
async def home():
    return {"Welcome to Signature Verification Project!👋"}


app.include_router(extract_router, prefix="/extract", tags=["extract_signature"])
