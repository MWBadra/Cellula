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


a="key"
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

# vectorstore = Chroma.from_documents(
#     documents=docs,
#     collection_name="humaneval_collection",
#     embedding=embedding_model,
#     persist_directory="./Code_generator_chromaDB",
# )


# Testing the locally saved data
vectorstore = Chroma(
    collection_name="humaneval_collection",
    persist_directory="./Code_generator_chromaDB",
    embedding_function=embedding_model  
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
llm = ChatOpenAI(model_name="gpt-5-nano",openai_api_key=a, temperature=1)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert Python coding assistant. Answer the user's coding questions using ONLY the provided context. If the answer is not in the context, say 'I don't know'.\n\nContext:\n{context}"),
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




# query = "Write a Python function called has_near_elements that checks if numbers in a list are closer to each other than a threshold."
query = "Write a Python function called calculate_alien_gravity that takes a planet's color and returns the gravitational pull."
print(f"User: {query}")

response = conversational_rag_chain.invoke(
    {"question": query},
    config={"configurable": {"session_id": "session_001"}} 
)
print(f"\nAI:\n{response}")