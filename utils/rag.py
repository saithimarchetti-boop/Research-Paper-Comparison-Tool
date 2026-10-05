from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


def create_vector_store(text, collection_name):

    chunks = []

    chunk_size = 1000

    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vector_store = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory="./chroma_db"
    )

    return vector_store