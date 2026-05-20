from langchain_core.documents import Document
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from state import AgenticRAGState


class CRAGNodes:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.web_search = DuckDuckGoSearchRun()

        # Setup chains
        self._init_chains()

    def _init_chains(self):
        # Grading prompt
        grade_prompt = PromptTemplate.from_template(
            """You are a relevance grader. 
Given the question and the document below, answer ONLY with 'yes' or 'no'.
Is the document relevant to the question?

Question: {question}
Document: {document}

Answer (yes/no):"""
        )
        self.grader_chain = grade_prompt | self.llm | StrOutputParser()

        # Generation prompt
        gen_prompt = PromptTemplate.from_template(
            """Answer the question using ONLY the context below.
If you cannot find the answer in the context, say "I don't know."

Context:
{context}

Question: {question}

Answer:"""
        )
        self.gen_chain = gen_prompt | self.llm | StrOutputParser()

    def retrieve_node(self, state: AgenticRAGState) -> dict:
        print("\n [Node 1] RETRIEVING from ChromaDB...")
        docs = self.retriever.invoke(state["question"])
        print(f"   Found {len(docs)} document(s).")
        return {"documents": docs, "run_web_search": False}

    def grade_docs_node(self, state: AgenticRAGState) -> dict:
        print("\n [Node 2] GRADING documents for relevance...")
        question = state["question"]
        docs = state["documents"]
        relevant_docs = []

        for doc in docs:
            score = self.grader_chain.invoke({
                "question": question,
                "document": doc.page_content[:1000],
            }).strip().lower()
            print(f"   Grade: '{score}' → {doc.page_content[:60]}...")
            if "yes" in score:
                relevant_docs.append(doc)

        if relevant_docs:
            print(f"    {len(relevant_docs)} relevant doc(s) found. Skipping web search.")
            return {"documents": relevant_docs, "run_web_search": False}
        else:
            print("    No relevant docs. Will fall back to web search.")
            return {"documents": [], "run_web_search": True}

    def web_search_node(self, state: AgenticRAGState) -> dict:
        print("\n [Node 3] RUNNING DuckDuckGo web search fallback...")
        results = self.web_search.run(state["question"])
        new_doc = Document(page_content=results, metadata={"source": "duckduckgo"})
        print(f"   Web results snippet: {results[:120]}...")
        return {"documents": [new_doc]}

    def generate_node(self, state: AgenticRAGState) -> dict:
        print("\n  [Node 4] GENERATING answer...")
        context = "\n\n".join(d.page_content for d in state["documents"])
        answer = self.gen_chain.invoke({
            "context": context,
            "question": state["question"],
        })
        return {"generation": answer.strip()}