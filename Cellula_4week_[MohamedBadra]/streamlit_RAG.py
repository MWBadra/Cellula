import streamlit as st
import os
from langchain_openai import ChatOpenAI
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

# --- UI CONFIGURATION ---
st.set_page_config(page_title="Self-Learning AI", page_icon="🧠")
st.title("🧠 Badra's Self-Learning Code Assistant")
st.write("Ask me a Python question. If I don't know it, you can teach me!")

# --- YOUR EXACT BACKEND SETUP ---
@st.cache_resource
def setup_backend():
    
    a=st.secrets["OPENAI_API_KEY"]  
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # dataset = load_dataset("openai/openai_humaneval")
    # dataset = dataset['test']
    # formatted_documents = []

    # for item in dataset:
    #     task_id = item['task_id']
    #     prompt = item['prompt']
    #     canonical_solution = item['canonical_solution']    
    #     combined_text = f"Task ID: {task_id}\n\nFunction Definition and Docstring:\n{prompt}\n\nSolution Code:\n{canonical_solution}"
        
    #     formatted_documents.append(combined_text)



    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # docs = [Document(page_content=text) for text in formatted_documents]

    # vectorstore = Chroma.from_documents(
    #     documents=docs,
    #     collection_name="humaneval_collection",
    #     embedding=embedding_model,
    #     persist_directory="./Code_generator_chromaDB",
    # )


    vectorstore = Chroma(
        collection_name="humaneval_collection",
        persist_directory="./Code_generator_chromaDB",
        embedding_function=embedding_model  
    )

   
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    llm = ChatOpenAI(model_name="gpt-5-nano", openai_api_key=a, temperature=1)

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

# --- YOUR EXACT CORE FUNCTIONS ---
def classify_intent(query):
    router_prompt = f"""
    Classify the intent of the following user query into exactly one of these two categories:
    1. Explain (if the user is asking for a general explanation, definition, or concept)
    2. Generate (if the user is asking to write, generate, or solve a specific coding function)
    
    Output ONLY the word 'Explain' or 'Generate'.
    
    Query: {query}
    """
    response = llm.invoke(router_prompt)
    return response.content.strip()

def retrieve_with_confidence(query):
    results = vectorstore.similarity_search_with_score(query, k=3)
    if not results:
        return None, False
    best_doc, score = results[0]
    print(f"  [DEBUG] Match Score: {score}")
    if score > 1:  
        return None, False
    return best_doc, True

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
    print(f" Successfully learned and memorized '{function_name}'!")


# --- THE STREAMLIT UI LOGIC ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "chat"
if "pending_query" not in st.session_state:
    st.session_state.pending_query = ""

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# MODE 1: TEACHING MODE
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
                # Call YOUR exact function
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

# MODE 2: NORMAL CHAT MODE
elif st.session_state.app_mode == "chat":
    if prompt := st.chat_input("User: "):
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Call YOUR exact routing function
            intent = classify_intent(prompt)
            
            if intent == "Explain":
                # General knowledge path
                response = llm.invoke(prompt).content
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            elif intent == "Generate":
                # Call YOUR exact confidence function
                best_doc, is_known = retrieve_with_confidence(prompt)
                
                if is_known:
                    st.markdown("*(Found in database. Generating...)*")
                    # Call YOUR exact conversational chain
                    answer = conversational_rag_chain.invoke(
                        {"question": prompt},
                        config={"configurable": {"session_id": "session_001"}} 
                    )
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                else:
                    # Switch to teaching mode
                    st.session_state.app_mode = "teaching"
                    st.session_state.pending_query = prompt
                    st.rerun()