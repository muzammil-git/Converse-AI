import re
from typing import List
import chromadb
from fastapi import APIRouter, File, UploadFile
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from config import GROQ_KEY, OPEN_AI_API_KEY
from llama_index.core.llms import ChatMessage, ChatResponse
from llama_index.llms.deepseek import DeepSeek
from llama_index.llms.groq import Groq
from uuid import uuid4
from llama_index.agent.openai import OpenAIAgent
from llama_index.core import PromptTemplate
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
import os
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.readers.web import SimpleWebPageReader
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.memory import ChatMemoryBuffer
import shutil


BASE_UPLOAD_DIR = os.path.abspath("tmp_files")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
router2 = APIRouter()

messages = [
    ChatMessage(
        role="system",
        content="Be a helpful assistant, dont answer any unrealated query",
    ),
]

##--------LLM--------
llm = Groq(
    temperature=0.3,
    model="deepseek-r1-distill-llama-70b",
    api_key=GROQ_KEY,
)


##--------Embedding Model----------
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")  # Free model


chat_store = SimpleChatStore()

chat_memory = ChatMemoryBuffer.from_defaults(
    token_limit=3000,
    chat_store=chat_store,
    # chat_store_key="user1",
)


@router2.get("/chat_history_aware", description="Multi Turn history aware, completion")
def multi_turn_with_history(user_id: str, user_input: str):

    chat_memory = ChatMemoryBuffer.from_defaults(
        token_limit=3000,
        chat_store=chat_store,
        chat_store_key=user_id,
    )

    # chat_memory.chat_store_key = user_id

    single_message = ChatMessage(
        role="user",
        content=user_input,
    )
    messages.append(single_message)
    chat_memory.put(message=single_message)

    all_messages = chat_memory.get_all()

    llm_output: ChatResponse = llm.chat(
        all_messages
    )  # multi-turn chat-style interactions

    print("Message Role: ", llm_output.message.role.value)
    print("Message Content: ", llm_output.message.content)
    # print("Message Data: ", llm_output.message.additional_kwargs)

    chat_memory.put(
        message=ChatMessage(
            content=llm_output.message.content, role=llm_output.message.role.value
        )
    )
    return {"message": llm_output.message.content}


@router2.get("/single", description="Single Turn, completion")
def single_turn(query_str: str):

    template = (
        "We have provided context information below. \n"
        "---------------------\n"
        "{context_str}"
        "\n---------------------\n"
        "Given this information, please answer the question: {query_str}\n"
    )
    qa_template = PromptTemplate(template)

    prompt = qa_template.format(
        context_str="I am ghost of sushishima and people are scared of me",
        query_str=query_str,
    )

    response = llm.complete(prompt=str(prompt))

    return str(response)


# print(single_turn("yo how are you"))


company_name: str = "ABC Corporation"
collection_name = "data"
file_list: list = None
# data_file_path = '/'


def storing(file_paths: List):

    # ----- LOADING DOCUMENT ----- #
    """
    server_path = os.getcwd()
    data_file = [f'{server_path}/my_docs/ESG-Data-Portal-Factsheet_EMS.pdf']
    """
    

    server_file_paths = []

    for each_file_path in file_paths:
        t = os.path.join(BASE_UPLOAD_DIR, each_file_path)
        server_file_paths.append(t)

    data_file = server_file_paths

    isWebpage: bool = False

    if isWebpage:
        documents = SimpleWebPageReader(html_to_text=True).load_data(
            ["https://corporate.bestbuy.com/archive/"]
        )
    else:
        reader = SimpleDirectoryReader(input_files=data_file)
        documents = reader.load_data()

    """
    
    # PRINTS DOCUMENTS
      
    # for document in documents:
    #     print(document)
    # print(documents)
    
    # -----INDEXING----- #

    
    # index = VectorStoreIndex.from_documents(
    #     documents=documents, 
    #     show_progress=True, 
    # )
    
    # index.storage_context.persist(persist_dir="./storage")
    
    """

    # ----- CHROMADB SETUP ----- #
    chroma_client = chromadb.PersistentClient(path=f"./chroma_db/{company_name}")
    chroma_client.clear_system_cache()

    
    chroma_collection = chroma_client.get_or_create_collection(collection_name)

    print("ChromaDB is now stored in:", "./chroma_db")

    # ----- VECTOR STORE SETUP ----- #

    ## Creates Vector Store with no text embedded
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, embed_model=embed_model
    )

    stored_items = chroma_collection.count()

    if stored_items > 0:
        print(f"✅ Successfully stored {stored_items} documents in ChromaDB!")
    else:
        print("❌ Error: No documents were stored in ChromaDB.")

    return index


# storing()


@router2.post("/upload")
async def upload_file(files: List[UploadFile] = File(...)):

    account_id = company_name  # This will actually be extracted from bearer token which is account id.
    saved_files = []
    saved_file_path = []

    files_location = os.path.join(BASE_UPLOAD_DIR, account_id)

    # Check if the directory exists, create if not
    os.makedirs(files_location, exist_ok=True)

    for each_file in files:
        full_file_path = os.path.join(files_location, each_file.filename)

        relative_path = os.path.relpath(full_file_path, BASE_UPLOAD_DIR)
        saved_file_path.append(relative_path)

        with open(full_file_path, "wb+") as f:
            f.write(each_file.file.read())
            saved_files.append(each_file.filename)

    s = storing(saved_file_path)

    return {
        "filenames": [file.filename for file in files],
        "saved_files": saved_files,
        "file_relpath": saved_file_path,
    }


@router2.get("/delete_db")
async def deleteDB():
    try:
        # Delete the entire database
        shutil.rmtree(os.path.abspath(f"chroma_db/{company_name}"))
        print("Database deleted successfully.")

        return {"message": "Database deleted successfully."}
    except Exception as e:
        print(f"Error deleting database: {e}")

        return {"message": f"Error deleting database: {e}"}


@router2.get("/single_rag", description="RAG Single-Turn")
async def retrieve_data(user_input: str):

    chroma_client = chromadb.PersistentClient(path=f"./chroma_db/{company_name}")
    chroma_collection = chroma_client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model,
    )
    # print(chroma_collection.count())  # Check the number of stored documents
    # print(chroma_collection.peek(3))  # Peek at some stored documents

    retriever = index.as_retriever(similarity_top_k=5)
    retrieved_docs = retriever.retrieve(user_input)

    retrieved_texts = "\n\n".join([doc.text for doc in retrieved_docs])
    prompt = f"Based on the following documents, answer the query:\n\n{retrieved_texts}\n\nQuery: {user_input}"

    # response = llm.complete(prompt)
    query_engine = index.as_query_engine(
        llm=llm, retriever_mode="embedding", similarity_top_k=5
    )
    response = query_engine.query(user_input)

    print("🔍 Query Results:", response)
    cleaned_response = str(
        re.sub(r"<think>.*?</think>", "", response.response, flags=re.DOTALL)
    ).strip()

    return {
        "question": user_input,
        "answer": cleaned_response,
        "result": response.response,
    }


# retrieve_data(user_input="List all the social metrics")


@router2.get("/rag_chat", description="RAG Multi-Turn")
async def retrieve_data(user_id: str, user_input: str):

    chroma_client = chromadb.PersistentClient(path=f"./chroma_db/{company_name}")
    chroma_collection = chroma_client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model,
    )
    # print(chroma_collection.count())  # Check the number of stored documents
    # print(chroma_collection.peek(3))  # Peek at some stored documents

    retriever = index.as_retriever(similarity_top_k=5)
    retrieved_docs = retriever.retrieve(user_input)

    retrieved_texts = "\n\n".join([doc.text for doc in retrieved_docs])
    
    # Initialize chat memory for user
    chat_memory = ChatMemoryBuffer.from_defaults(
        token_limit=3000,
        chat_store=chat_store,
        chat_store_key=user_id  # Stores chat history per user
    )

    # Retrieve past chat context
    chat_history = chat_memory.get_all()  # Fetch stored messages
    
    formatted_chat_history = "\n".join(
        [f"{msg.role.value}: {msg.content}" for msg in chat_history]
    )
    
    print(formatted_chat_history)
    
    prompt = f"Based on the following documents, answer the query:\n\n{retrieved_texts}\n\nChat History:\n{formatted_chat_history}\n\nQuery: {user_input}"

    # response = llm.complete(prompt)
    query_engine = index.as_query_engine(
        llm=llm, retriever_mode="embedding", similarity_top_k=5
    )
    response = query_engine.query(prompt)

    print("🔍 Query Results:", response)
    cleaned_response = str(
        re.sub(r"<think>.*?</think>", "", response.response, flags=re.DOTALL)
    ).strip()

    # Store response in chat memory
    chat_memory.put(
        message=ChatMessage(
            content=response.response,
            role="user"
        ),
    )
    
    chat_memory.put(
        message=ChatMessage(
            content=cleaned_response,
            role="assistant"
        ),
    )


    return {
        "question": user_input,
        "answer": cleaned_response,
        "result": response.response,
        "chat_history": chat_memory.get_all(),
    }


# retrieve_data(user_input="List all the social metrics")
