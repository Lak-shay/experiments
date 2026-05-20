package com.crag.state;

import org.bsc.langgraph4j.state.AgentState;
import org.bsc.langgraph4j.state.Channel;
import org.bsc.langgraph4j.state.Channels;

import java.util.List;
import java.util.Map;
import java.util.Optional;

public class AgenticRAGState extends AgentState {

    private static <T> Channel<T> overwrite() {
        return Channels.base((current, update) -> update);
    }

    public static final Map<String, Channel<?>> SCHEMA = Map.of(
            "question", overwrite(),
            "documents", overwrite(),
            "run_web_search", overwrite(),
            "generation", overwrite()
    );

    public AgenticRAGState(Map<String, Object> initData) {
        super(initData);
    }

    public Optional<String> question() {
        return value("question");
    }

    @SuppressWarnings("unchecked")
    public Optional<List<String>> documents() {
        return value("documents");
    }

    public boolean runWebSearch() {
        return this.<Boolean>value("run_web_search").orElse(false);
    }

    public Optional<String> generation() {
        return value("generation");
    }
}
