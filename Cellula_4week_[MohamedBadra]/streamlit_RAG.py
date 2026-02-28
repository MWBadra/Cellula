import streamlit as st
import os
from datasets import load_dataset
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from operator import itemgetter
from langgraph.graph import START, StateGraph, END
from typing_extensions import TypedDict, Optional
from langchain_google_genai import ChatGoogleGenerativeAI


class State(TypedDict):
    question: str
    chat_history: list
    intent: Optional[str]
    is_known: Optional[bool]
    generation: Optional[str]

st.set_page_config(page_title="Self-Learning AI", page_icon="🧠")
st.title("Self-Learning Code Assistant")
st.write("Ask me a Python question. If I don't know it, you can teach me!")


@st.cache_resource
def setup_backend():
    import shutil 
    
    a = st.secrets["OPENAI_API_KEY"]  
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_db_path = os.path.join(current_dir, "Code_generator_chromaDB")
    
    writable_db_path = "/tmp/Code_generator_chromaDB"
    
    if not os.path.exists(writable_db_path):
        shutil.copytree(repo_db_path, writable_db_path)
        for root, dirs, files in os.walk(writable_db_path):
            for file in files:
                os.chmod(os.path.join(root, file), 0o666)

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = Chroma(
        collection_name="humaneval_collection",
        persist_directory=writable_db_path,
        embedding_function=embedding_model  
    )
   
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=a, temperature=1)
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
    
    return llm, vectorstore, conversational_rag_chain


llm, vectorstore, conversational_rag_chain = setup_backend()

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
    return {"generation": response.content}
 
    
def generate_rag_node(state: State):
    query = state["question"]
    response = conversational_rag_chain.invoke(
                {"question": query},
                config={"configurable": {"session_id": "session_002"}} 
            )
    return {"generation": response}

def human_teaching_node(state: State):
   
    return {"generation": "UNKNOWN_FUNCTION_FLAG"}

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

def learn_new_function(function_name, code, explanation):
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

if "messages" not in st.session_state:
    st.session_state.messages = []
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "chat"
if "pending_query" not in st.session_state:
    st.session_state.pending_query = ""

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


if st.session_state.app_mode == "teaching":
    st.error("⚠️ Unknown function! I don't know this function. Please teach me so I can learn it!")
    
    with st.form("teaching_form", clear_on_submit=True):
        st.write(f"**Question was:** {st.session_state.pending_query}")
        
        f_name = st.text_input("What is the function name?")
        f_code = st.text_area("Paste the Python code:")
        f_desc = st.text_input("Give a brief explanation:")
        
        submitted = st.form_submit_button("Save to Memory 🧠")
        
        if submitted:
            if not f_name or not f_code or not f_desc:
                st.warning("Hold up! You need to fill out all three fields before saving.")
            else:
                learn_new_function(f_name, f_code, f_desc)
                success_msg = f"✅ Thank you! I have saved `{f_name}` to my memory. Try asking me for it now!"
                st.session_state.messages.append({"role": "assistant", "content": success_msg})
                
                st.session_state.app_mode = "chat"
                st.session_state.pending_query = ""
                st.rerun()

        if st.form_submit_button("Cancel / Go Back"):
            st.session_state.app_mode = "chat"
            st.session_state.pending_query = ""
            st.rerun()

elif st.session_state.app_mode == "chat":
    if prompt := st.chat_input("User: "):
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            initial_state = {
                "question": prompt,
                "chat_history": st.session_state.messages[:-1] 
            }
            final_state = app.invoke(initial_state)
            
            
            if final_state["generation"] == "UNKNOWN_FUNCTION_FLAG":
                st.session_state.app_mode = "teaching"
                st.session_state.pending_query = prompt
                st.rerun()
            else:
                if final_state.get("is_known"):
                    st.markdown("*(Found in database. Generating...)*")
                
                answer = final_state["generation"]
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})









#test demo:

# Write a Python function called has_close_elements that checks if numbers in a list are closer to each other than a threshold.

# Can you explain exactly how the nested loops in that code work?

# Write a Python function called calculate_toxicity_score that takes a text string and a blacklist of words, and returns a toxicity percentage.

# Name: calculate_toxicity_score

# Code: 
# def calculate_toxicity_score(text, blacklist):
# words = text.lower().split()
# if not words:
# return 0.0
# toxic_count = sum(1 for word in words if word in blacklist)
# return (toxic_count / len(words)) * 100

# Explanation: Splits the text into lowercase words, counts how many appear in the blacklist, and returns the percentage of toxic words.

# Write the calculate_toxicity_score function.
