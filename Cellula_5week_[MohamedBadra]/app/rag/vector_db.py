import os
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from operator import itemgetter

github_token="github_key"
# 1. Setup Models
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=a, temperature=1)
llm = ChatOpenAI(
    model="meta/Llama-4-Scout-17B-16E-Instruct",
    api_key=github_token,
    base_url="https://models.github.ai/inference",
    temperature=0.8,
    max_tokens=2048,
    model_kwargs={"top_p": 0.1}
)

# 2. Setup Database
# Using your exact local path logic
current_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(current_dir, "../../Code_generator_chromaDB copy")
vectorstore = Chroma(
    collection_name="humaneval_collection",
    persist_directory=db_path,
    embedding_function=embedding_model  
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 3. Setup RAG Chain
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert Python coding assistant. Answer using ONLY context. Format with Markdown. If not in context, say 'I don't know'.\n\nContext:\n{context}"),
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
    RunnablePassthrough.assign(context=itemgetter("question") | retriever | format_docs)
    | prompt | llm | StrOutputParser()
)

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="question",     
    history_messages_key="chat_history"
)