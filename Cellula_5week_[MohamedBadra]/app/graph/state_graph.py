from langgraph.graph import START, StateGraph, END
from typing_extensions import TypedDict, Optional
from app.rag.vector_db import llm, vectorstore, conversational_rag_chain

class State(TypedDict):
    question: str
    intent: Optional[str]
    is_known: Optional[bool]
    generation: Optional[str]
    mode: Optional[str]  

def classify_intent_node(state: State):
    user_query = state["question"]
    router_prompt = f"""
    Classify the intent of the following user query into exactly one of these two categories:
    1. Explain (if the user is asking for a general explanation, definition, or concept)
    2. Generate (if the user is asking to write, generate, or solve a specific coding function)
    
    Output ONLY the word 'Explain' or 'Generate'.
    
    Query: {user_query}
    """
    response = llm.invoke(router_prompt)
    intent_result = response.content.strip()
    return {"intent": intent_result}
def retrieve_node(state: State):
    query = state["question"]
    results = vectorstore.similarity_search_with_score(query, k=3)
    if not results:
        return {"is_known": False}
    best_doc, score = results[0]
    print(f"  [DEBUG] Match Score: {score}")
    if score > 1:  
        return {"is_known": False}
    return {"is_known": True} 

def explain_node(state: State):
    query = state["question"]
    history = state.get("chat_history", [])
    
    messages = [
        ("system", "You are an expert Python assistant. Use the conversation history to understand context. Keep explanations concise.")
    ]
    
    for msg in history:
        role = "human" if msg["role"] == "user" else "ai"
        messages.append((role, msg["content"]))
        
    messages.append(("human", query))
    
    response = llm.invoke(messages)
    return {"generation": response.content, "mode": "explain"}
 
    
def generate_rag_node(state: State):
    query = state["question"]
    response = conversational_rag_chain.invoke(
                {"question": query},
                config={"configurable": {"session_id": "session_002"}} 
            )
    return {"generation": response, "mode": "generate"}

def human_teaching_node(state: State):
   
    return {"generation": "UNKNOWN_FUNCTION_FLAG", "mode": "teaching"}

def route_intent(state: State):
    if state["intent"] == "Explain":
        return "explain_node"
    return "retrieve_node"

def route_knowledge(state: State):
    if state["is_known"]:
        return "generate_rag_node"
    return "human_teaching_node"

workflow = StateGraph(State)

workflow.add_node("classify_intent", classify_intent_node)
workflow.add_node("explain_node", explain_node)
workflow.add_node("retrieve_node", retrieve_node)
workflow.add_node("generate_rag_node", generate_rag_node)
workflow.add_node("human_teaching_node", human_teaching_node)

workflow.add_edge(START, "classify_intent") 
workflow.add_conditional_edges("classify_intent", route_intent) 
workflow.add_conditional_edges("retrieve_node", route_knowledge) 

workflow.add_edge("explain_node", END)
workflow.add_edge("generate_rag_node", END)
workflow.add_edge("human_teaching_node", END)

app = workflow.compile()

