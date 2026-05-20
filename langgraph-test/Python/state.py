from typing import TypedDict, List
from langchain_core.documents import Document

class AgenticRAGState(TypedDict):
    question: str
    documents: List[Document]
    run_web_search: bool
    generation: str