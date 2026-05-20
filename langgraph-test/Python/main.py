from config import load_embeddings, build_vectorstore, load_llm
from graph import build_graph
from dotenv import load_dotenv
load_dotenv()

if __name__ == "__main__":
    # Sample local dataset
    SAMPLE_DOCS = [
        "LangGraph is a library for building stateful, multi-actor applications with LLMs using graph-based workflows.",
        "ChromaDB is an open-source vector database designed for storing and querying embeddings locally.",
        "Retrieval-Augmented Generation (RAG) improves LLM answers by fetching relevant documents before generating a response.",
        "Corrective RAG adds a grading step to verify document relevance, falling back to web search when needed.",
        "HuggingFace Transformers provides thousands of pre-trained models available for free download and local inference.",
    ]

    print("=" * 60)
    print("  CRAG Agent — ChromaDB + DuckDuckGo + HuggingFace")
    print("=" * 60)

    # Initialization
    embeddings  = load_embeddings()
    vectorstore = build_vectorstore(SAMPLE_DOCS, embeddings)
    retriever   = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm         = load_llm()
    
    # Build the Compiled Application
    app = build_graph(retriever, llm)

    # Run Test evaluation questions
    test_questions = [
        "What is Corrective RAG?",                          # Expect Vector DB hit
        "Who won the FIFA World Cup in 2022?",              # Fallback to DuckDuckGo
    ]

    for question in test_questions:
        print("\n" + "─" * 60)
        print(f"❓ Question: {question}")
        result = app.invoke({"question": question})
        print(f"\n💬 Answer: {result['generation']}")

    print("\n" + "=" * 60)
    print("Done! Add your own documents to SAMPLE_DOCS to customise the knowledge base.")