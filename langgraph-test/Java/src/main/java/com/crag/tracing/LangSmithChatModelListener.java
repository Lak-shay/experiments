package com.crag.tracing;

import dev.langchain4j.model.chat.listener.ChatModelErrorContext;
import dev.langchain4j.model.chat.listener.ChatModelListener;
import dev.langchain4j.model.chat.listener.ChatModelRequest;
import dev.langchain4j.model.chat.listener.ChatModelRequestContext;
import dev.langchain4j.model.chat.listener.ChatModelResponse;
import dev.langchain4j.model.chat.listener.ChatModelResponseContext;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class LangSmithChatModelListener implements ChatModelListener {

    private static final String SPAN_KEY = LangSmithChatModelListener.class.getName() + ".span";

    private final LangSmithTracer tracer;

    public LangSmithChatModelListener(LangSmithTracer tracer) {
        this.tracer = tracer;
    }

    @Override
    public void onRequest(ChatModelRequestContext requestContext) {
        if (!tracer.isEnabled()) {
            return;
        }
        ChatModelRequest request = requestContext.request();
        Map<String, Object> inputs = new LinkedHashMap<>();
        inputs.put("model", request.model());
        inputs.put("messages", request.messages().stream().map(Object::toString).collect(Collectors.toList()));
        inputs.put("temperature", request.temperature());
        inputs.put("top_p", request.topP());
        inputs.put("max_tokens", request.maxTokens());
        inputs.put("tools", request.toolSpecifications() == null
                ? List.of()
                : request.toolSpecifications().stream().map(Object::toString).toList());

        LangSmithTracer.TraceSpan span = tracer.startRun(
                "chat_model",
                "llm",
                inputs
        );
        requestContext.attributes().put(SPAN_KEY, span);
    }

    @Override
    public void onResponse(ChatModelResponseContext responseContext) {
        Object value = responseContext.attributes().get(SPAN_KEY);
        if (!(value instanceof LangSmithTracer.TraceSpan span)) {
            return;
        }
        ChatModelResponse response = responseContext.response();
        Map<String, Object> outputs = new LinkedHashMap<>();
        outputs.put("message", response.aiMessage() == null ? null : response.aiMessage().text());
        outputs.put("finish_reason", response.finishReason() == null ? null : response.finishReason().name());
        outputs.put("model", response.model());
        outputs.put("token_usage", response.tokenUsage() == null ? null : response.tokenUsage().toString());
        span.recordOutputs(outputs);
        span.close();
    }

    @Override
    public void onError(ChatModelErrorContext errorContext) {
        Object value = errorContext.attributes().get(SPAN_KEY);
        if (!(value instanceof LangSmithTracer.TraceSpan span)) {
            return;
        }
        span.recordError(errorContext.error());
        span.close();
    }
}
