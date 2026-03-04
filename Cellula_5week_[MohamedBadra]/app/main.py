from fastapi import FastAPI
from app.api.endpoints import router as api_router

app = FastAPI(
    title="Self-Learning Code Assistant API",
    description="LangGraph-powered API for code generation and explanation.",
    version="1.0.0"
)

app.include_router(api_router)

@app.get("/")
async def root():
    return {"message": "API is live. Send POST requests to /query."}