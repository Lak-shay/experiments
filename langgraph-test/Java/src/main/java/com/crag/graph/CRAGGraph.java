package com.crag.graph;

import com.crag.nodes.CRAGNodes;
import com.crag.state.AgenticRAGState;
import com.crag.tracing.LangSmithTracer;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingStore;
import org.bsc.langgraph4j.CompiledGraph;
import org.bsc.langgraph4j.StateGraph;
import org.bsc.langgraph4j.action.EdgeAction;

import java.util.Map;

import static org.bsc.langgraph4j.StateGraph.END;
import static org.bsc.langgraph4j.StateGraph.START;
import static org.bsc.langgraph4j.action.AsyncEdgeAction.edge_async;
import static org.bsc.langgraph4j.action.AsyncNodeAction.node_async;

public class CRAGGraph {

    public static CompiledGraph<AgenticRAGState> build(
            EmbeddingStore<TextSegment> vectorStore,
            EmbeddingModel embeddingModel,
            ChatLanguageModel llm,
            LangSmithTracer tracer) throws Exception {

        CRAGNodes nodes = new CRAGNodes(vectorStore, embeddingModel, llm, tracer);

        EdgeAction<AgenticRAGState> routeAfterGrading = state ->
                state.runWebSearch() ? "web_search" : "generate";

        return new StateGraph<>(AgenticRAGState.SCHEMA, AgenticRAGState::new)
                .addNode("retrieve", node_async(nodes::retrieveNode))
                .addNode("grade_docs", node_async(nodes::gradeDocsNode))
                .addNode("web_search", node_async(nodes::webSearchNode))
                .addNode("generate", node_async(nodes::generateNode))
                .addEdge(START, "retrieve")
                .addEdge("retrieve", "grade_docs")
                .addEdge("web_search", "generate")
                .addEdge("generate", END)
                .addConditionalEdges(
                        "grade_docs",
                        edge_async(routeAfterGrading),
                        Map.of("web_search", "web_search", "generate", "generate")
                )
                .compile();
    }
}
