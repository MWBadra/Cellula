from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from operator import itemgetter

a="key"
# base_url = "https://openrouter.ai/api/v1/"

loader = TextLoader("D:/books and slides/Cellula26/Cellula/Cellula_4week_[MohamedBadra]/my_bio.txt", encoding="utf-8")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = FAISS.from_documents(documents=splits, embedding=embedding_model)
retriever = vectorstore.as_retriever()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a latin man who only speaks spanish but understands english . You meet a man named Badra. Answer all his questions in your own way using ONLY the following context. Use three sentences maximum and keep the answer concise.\n\nContext: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]



llm = ChatOpenAI(model_name="gpt-5-nano",openai_api_key=a, temperature=1)


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


query1 = "Hi, my name is Mohamed i support Liverpool. What projects did I build?"
print(f"User: {query1}")
response1 = conversational_rag_chain.invoke(
    {"question": query1}, 
    config={"configurable": {"session_id": "session_001"}}
)
print(f"AI: {response1}\n")

query2 = "Wait, what do i support again?"
print(f"User: {query2}")
response2 = conversational_rag_chain.invoke(
    {"question": query2},
    config={"configurable": {"session_id": "session_001"}} 
)
print(f"AI: {response2}\n")