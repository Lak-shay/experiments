package com.crag;

import com.crag.graph.CRAGGraph;
import com.crag.state.AgenticRAGState;
import com.crag.tracing.LangSmithChatModelListener;
import com.crag.tracing.LangSmithTracer;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import org.bsc.langgraph4j.CompiledGraph;

import java.util.List;
import java.util.Map;

public class Main {

    private static final List<String> SAMPLE_DOCS = List.of(
            "LangGraph is a library for building stateful, multi-actor applications with LLMs using graph-based workflows.",
            "ChromaDB is an open-source vector database designed for storing and querying embeddings locally.",
            "Retrieval-Augmented Generation (RAG) improves LLM answers by fetching relevant documents before generating a response.",
            "Corrective RAG adds a grading step to verify document relevance, falling back to web search when needed.",
            "HuggingFace Transformers provides thousands of pre-trained models available for free download and local inference."
    );

    private static final List<String> TEST_QUESTIONS = List.of(
            "What is Corrective RAG?",
            "Who won the FIFA World Cup in 2022?"
    );

    public static void main(String[] args) throws Exception {
        String apiKey = System.getenv("GOOGLE_API_KEY");
        if (apiKey == null || apiKey.isBlank()) {
            throw new IllegalStateException("GOOGLE_API_KEY not set in environment");
        }

        System.out.println("=".repeat(60));
        System.out.println("  CRAG Agent - LangGraph4j + LangChain4j + Gemini");
        System.out.println("=".repeat(60));

        LangSmithTracer tracer = new LangSmithTracer();
        if (tracer.isEnabled()) {
            System.out.println("LangSmith tracing enabled.");
        }

        System.out.println("\nLoading embedding model...");
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        System.out.println("Building in-memory vector store...");
        EmbeddingStore<TextSegment> vectorStore = new InMemoryEmbeddingStore<>();
        for (String text : SAMPLE_DOCS) {
            TextSegment segment = TextSegment.from(text);
            Embedding embedding = embeddingModel.embed(segment).content();
            vectorStore.add(embedding, segment);
        }
        System.out.printf("   Indexed %d document(s).%n", SAMPLE_DOCS.size());

        System.out.println("Connecting to Gemini 2.5 Flash...");
        ChatLanguageModel llm = GoogleAiGeminiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-2.5-flash-lite")
                .temperature(0.0)
                .listeners(List.of(new LangSmithChatModelListener(tracer)))
                .build();

        CompiledGraph<AgenticRAGState> app =
                CRAGGraph.build(vectorStore, embeddingModel, llm, tracer);

        long totalStart = System.currentTimeMillis();

        for (String question : TEST_QUESTIONS) {
            System.out.println("\n" + "-".repeat(60));
            System.out.println("Question: " + question);

            long qStart = System.currentTimeMillis();
            LangSmithTracer.TraceSpan rootSpan = tracer.startRun(
                    "crag_question",
                    "chain",
                    Map.of("question", question)
            );

            try {
                var result = app.invoke(Map.of("question", question));
                long qElapsed = System.currentTimeMillis() - qStart;

                String answer = result
                        .flatMap(AgenticRAGState::generation)
                        .orElse("(no answer generated)");

                rootSpan.recordOutputs(Map.of(
                        "answer", answer,
                        "elapsed_ms", qElapsed
                ));

                System.out.println("\nAnswer: " + answer);
                System.out.printf("Time: %.2f s%n", qElapsed / 1000.0);
            } catch (Exception e) {
                rootSpan.recordError(e);
                throw e;
            } finally {
                rootSpan.close();
            }
        }

        long totalElapsed = System.currentTimeMillis() - totalStart;
        System.out.println("\n" + "=".repeat(60));
        System.out.printf("Done. Total time: %.2f s%n", totalElapsed / 1000.0);
    }
}
