package com.crag.nodes;

import com.crag.search.DDGS;
import com.crag.state.AgenticRAGState;
import com.crag.tracing.LangSmithTracer;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingStore;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class CRAGNodes {

    private final EmbeddingStore<TextSegment> vectorStore;
    private final EmbeddingModel embeddingModel;
    private final ChatLanguageModel llm;
    private final LangSmithTracer tracer;
    private final DDGS webSearch = new DDGS();

    private static final int TOP_K = 3;

    public CRAGNodes(EmbeddingStore<TextSegment> vectorStore,
                     EmbeddingModel embeddingModel,
                     ChatLanguageModel llm,
                     LangSmithTracer tracer) {
        this.vectorStore = vectorStore;
        this.embeddingModel = embeddingModel;
        this.llm = llm;
        this.tracer = tracer;
    }

    public Map<String, Object> retrieveNode(AgenticRAGState state) {
        try {
            return tracer.trace("retrieve", "retriever", Map.of(
                    "question", state.question().orElse("")
            ), () -> {
                System.out.println("\n[Node 1] RETRIEVING from in-memory vector store...");
                String question = state.question().orElseThrow();

                Embedding queryEmbedding = embeddingModel.embed(question).content();
                EmbeddingSearchRequest searchRequest = EmbeddingSearchRequest.builder()
                        .queryEmbedding(queryEmbedding)
                        .maxResults(TOP_K)
                        .build();
                List<EmbeddingMatch<TextSegment>> matches =
                        vectorStore.search(searchRequest).matches();

                List<String> docs = matches.stream()
                        .map(match -> match.embedded().text())
                        .toList();

                System.out.printf("   Found %d document(s).%n", docs.size());
                return Map.of(
                        "documents", docs,
                        "run_web_search", false
                );
            });
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public Map<String, Object> gradeDocsNode(AgenticRAGState state) {
        try {
            return tracer.trace("grade_docs", "chain", Map.of(
                    "question", state.question().orElse(""),
                    "documents", state.documents().orElse(List.of())
            ), () -> {
                System.out.println("\n[Node 2] GRADING documents for relevance...");
                String question = state.question().orElseThrow();
                List<String> docs = state.documents().orElse(List.of());

                List<String> relevant = new ArrayList<>();
                for (String doc : docs) {
                    String prompt = """
                            You are a relevance grader.
                            Given the question and the document below, answer ONLY with 'yes' or 'no'.
                            Is the document relevant to the question?

                            Question: %s
                            Document: %s

                            Answer (yes/no):""".formatted(question, doc.substring(0, Math.min(1000, doc.length())));

                    String score = chat(prompt).toLowerCase().strip();
                    System.out.printf("   Grade: '%s' -> %s...%n", score, doc.substring(0, Math.min(60, doc.length())));

                    if (score.contains("yes")) {
                        relevant.add(doc);
                    }
                }

                if (!relevant.isEmpty()) {
                    System.out.printf("   %d relevant doc(s) found. Skipping web search.%n", relevant.size());
                    return Map.of("documents", relevant, "run_web_search", false);
                }

                System.out.println("   No relevant docs. Will fall back to web search.");
                return Map.of("documents", List.of(), "run_web_search", true);
            });
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public Map<String, Object> webSearchNode(AgenticRAGState state) {
        try {
            return tracer.trace("web_search", "tool", Map.of(
                    "question", state.question().orElse("")
            ), () -> {
                System.out.println("\n[Node 3] RUNNING DuckDuckGo web search fallback...");
                String question = state.question().orElseThrow();
                String results = webSearch.search(question);

                System.out.printf("   Web results snippet: %s...%n",
                        results.substring(0, Math.min(120, results.length())));

                return Map.of("documents", List.of(results));
            });
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public Map<String, Object> generateNode(AgenticRAGState state) {
        try {
            return tracer.trace("generate", "chain", Map.of(
                    "question", state.question().orElse(""),
                    "documents", state.documents().orElse(List.of())
            ), () -> {
                System.out.println("\n[Node 4] GENERATING answer...");
                String question = state.question().orElseThrow();
                String context = String.join("\n\n", state.documents().orElse(List.of()));

                String prompt = """
                        Answer the question using ONLY the context below.
                        If you cannot find the answer in the context, say "I don't know."

                        Context:
                        %s

                        Question: %s

                        Answer:""".formatted(context, question);

                String answer = chat(prompt).strip();
                return Map.of("generation", answer);
            });
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private String chat(String prompt) {
        ChatRequest request = ChatRequest.builder()
                .messages(UserMessage.from(prompt))
                .build();
        return llm.chat(request).aiMessage().text();
    }
}
