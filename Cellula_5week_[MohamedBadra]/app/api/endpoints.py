from fastapi import APIRouter
from pydantic import BaseModel
from app.graph.state_graph import app as app_graph

router = APIRouter()

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    mode: str
    response: str

@router.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    initial_state = {"question": request.query}
    final_state = app_graph.invoke(initial_state)
    
    return QueryResponse(
        mode=final_state.get("mode", "unknown"),
        response=final_state.get("generation", "Error generating response.")
    )