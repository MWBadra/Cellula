from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate


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
        ("human", "good morning , how are you doing?"),
        ("ai", "¡Estoy bien, gracias!"),
        ("human", "{question}"),
    ]
)



llm = ChatOpenAI(model_name="gpt-5-nano",openai_api_key=a, temperature=1)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

query = "Can you describe the architecture and database used for the Ma3loma Reddit Clone?"
print(f"User: {query}")
response = rag_chain.invoke(query)
print(f"latin man: {response}")