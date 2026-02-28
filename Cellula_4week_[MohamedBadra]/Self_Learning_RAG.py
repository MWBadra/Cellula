from langchain_openai import ChatOpenAI
from datasets import load_dataset
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from operator import itemgetter
from langgraph.graph import START, StateGraph,END
from typing_extensions import TypedDict,Optional


class State(TypedDict):
    question: str
    intent: Optional[str]
    is_known: Optional[bool]
    generation: Optional[str]


a= "key"
dataset = load_dataset("openai/openai_humaneval")
dataset = dataset['test']
formatted_documents = []

for item in dataset:
    task_id = item['task_id']
    prompt = item['prompt']
    canonical_solution = item['canonical_solution']    
    combined_text = f"Task ID: {task_id}\n\nFunction Definition and Docstring:\n{prompt}\n\nSolution Code:\n{canonical_solution}"
    
    formatted_documents.append(combined_text)



embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

docs = [Document(page_content=text) for text in formatted_documents]

vectorstore = Chroma.from_documents(
    documents=docs,
    collection_name="humaneval_collection",
    embedding=embedding_model,
    persist_directory="./Code_generator_chromaDB",
)


# Testing the locally saved data
# vectorstore = Chroma(
#     collection_name="humaneval_collection",
#     persist_directory="./Code_generator_chromaDB",
#     embedding_function=embedding_model  
# )

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
llm = ChatOpenAI(model_name="gpt-5-nano",openai_api_key=a, temperature=1)

prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert Python coding assistant. Answer the user's coding questions using ONLY the provided context. Always format your Python code output using Markdown code blocks (```python ... ```). If the answer is not in the context, say 'I don't know'.\n\nContext:\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    RunnablePassthrough.assign(
        context=itemgetter("question") | retriever | format_docs
    )
    | prompt
    | llm
    | StrOutputParser()
)

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="question",     
    history_messages_key="chat_history"
)

def classify_intent(state: State):
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
    query=state["question"]
    results = vectorstore.similarity_search_with_score(query, k=3)
    
    if not results:
        return {"is_known": False}
        
    best_doc, score = results[0]
    
    print(f"  [DEBUG] Match Score: {score}")
    if score >1:  
        return {"is_known": False}
        
    return {"is_known": True} 

def human_teaching_node(state: State):
    function_name=input("  -> What is the function name? ")
    code = input("  -> Paste the Python code: ")
    explanation = input("  -> Give a brief explanation: ")
    document = f"""
    Function: {function_name}
    Code:
    {code}
    Explanation:
    {explanation}
    """
    
    vectorstore.add_documents([
        Document(
            page_content=document,
            metadata={"type": "function"}
        )
    ])
    return {"generation": "Thank you! I learned the function."}

def explain_node(state: State):
    query=state["question"]
    response = llm.invoke(query)
    return {"generation": response.content}
    

def generate_rag_node(state: State):
    query=state["question"]
    response = conversational_rag_chain.invoke(
                {"question": query},
                config={"configurable": {"session_id": "session_001"}} 
            )
    return {"generation": response}




def route_intent(state: State):
    if state["intent"] == "Explain":
        return "explain_node"
    return "retrieve_node"

def route_knowledge(state: State):
    if state["is_known"]:
        return "generate_rag_node"
    return "human_teaching_node"




workflow = StateGraph(State)

workflow.add_node("classify_intent", classify_intent)
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


print("\n=== WELCOME TO YOUR LANGGRAPH AI ===")
print("Type 'exit' to quit.\n")

while True:
    query = input("User: ")
    if query.lower() == 'exit':
        break
        
    initial_state = {"question": query}
    
    final_state = app.invoke(initial_state)
    
    print(f"AI:\n{final_state['generation']}\n")