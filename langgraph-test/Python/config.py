import os
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI


def load_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",  # current stable
        temperature=0,
        google_api_key=os.environ["GOOGLE_API_KEY"],
    )


def load_embeddings():
    """Free local sentence-transformer embeddings."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def build_vectorstore(texts: List[str], embeddings) -> Chroma:
    """Create a temporary, lightning-fast in-memory ChromaDB collection."""
    docs = [Document(page_content=t) for t in texts]

    # Keeps it strictly in RAM to keep execution smooth on Windows
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="crag_collection",
    )
    return vectorstore