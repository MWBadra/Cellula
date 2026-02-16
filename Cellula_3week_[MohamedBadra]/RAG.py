import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


a="my key"
# base_url = "https://openrouter.ai/api/v1/"


loader = TextLoader("my_bio.txt", encoding="utf-8")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = FAISS.from_documents(documents=splits, embedding=embedding_model)
retriever = vectorstore.as_retriever()


template = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.

Question: {question} 

Context: {context} 

Answer:"""
prompt = PromptTemplate.from_template(template)
llm = ChatOpenAI(model_name="gpt-5-nano",openai_api_key=a, temperature=0)
# llm = ChatOpenAI(
#     model="openai/gpt-oss-120b:free",  # <--- CHANGED THIS LINE
#     openai_api_key=a,                  # Your variable for the API key
#     base_url=base_url,                 # "https://openrouter.ai/api/v1"
#     temperature=0,
#     default_headers={
#         "HTTP-Referer": "http://localhost:8000",
#         "X-Title": "My RAG App"
#     }
# )

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


query1 = "Who are the key members of Led Zeppelin?"
print(f"User: {query1}")
response1 = rag_chain.invoke(query1)
print(f"AI: {response1}\n")

query2 = "Tell me about Gary Moore's style."
print(f"User: {query2}")
response2 = rag_chain.invoke(query2)
print(f"AI: {response2}\n")

query3 = "talk to me about pink floyd"
print(f"User: {query3}")
response1 = rag_chain.invoke(query3)
print(f"AI: {response1}\n")