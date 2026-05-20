package com.crag.tracing;

import com.fasterxml.jackson.databind.ObjectMapper;
import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.time.Instant;
import java.util.ArrayDeque;
import java.util.Collection;
import java.util.Deque;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

public class LangSmithTracer {

    private static final Logger log = LoggerFactory.getLogger(LangSmithTracer.class);
    private static final MediaType JSON = MediaType.get("application/json");

    private final boolean enabled;
    private final String endpoint;
    private final String apiKey;
    private final String workspaceId;
    private final String projectName;
    private final OkHttpClient http;
    private final ObjectMapper json;
    private final ThreadLocal<Deque<String>> runStack = ThreadLocal.withInitial(ArrayDeque::new);

    public LangSmithTracer() {
        this(
                isTracingEnabled(),
                getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"),
                System.getenv("LANGSMITH_API_KEY"),
                System.getenv("LANGSMITH_WORKSPACE_ID"),
                getenv("LANGSMITH_PROJECT", "crag-agent"),
                new OkHttpClient(),
                new ObjectMapper()
        );
    }

    LangSmithTracer(boolean enabled,
                    String endpoint,
                    String apiKey,
                    String workspaceId,
                    String projectName,
                    OkHttpClient http,
                    ObjectMapper json) {
        this.enabled = enabled && apiKey != null && !apiKey.isBlank();
        this.endpoint = stripTrailingSlash(endpoint);
        this.apiKey = apiKey;
        this.workspaceId = workspaceId;
        this.projectName = projectName;
        this.http = http;
        this.json = json;
    }

    public boolean isEnabled() {
        return enabled;
    }

    public TraceSpan startRun(String name, String runType, Map<String, Object> inputs) {
        String runId = UUID.randomUUID().toString();
        String parentRunId = currentParentRunId();
        if (enabled) {
            createRun(runId, parentRunId, name, runType, inputs);
        }
        runStack.get().push(runId);
        return new TraceSpan(runId);
    }

    public <T> T trace(String name,
                       String runType,
                       Map<String, Object> inputs,
                       ThrowingSupplier<T> action) throws Exception {
        TraceSpan span = startRun(name, runType, inputs);
        try {
            T result = action.get();
            span.recordOutputs(toOutputs(result));
            return result;
        } catch (Exception e) {
            span.recordError(e);
            throw e;
        } finally {
            span.close();
        }
    }

    public final class TraceSpan implements AutoCloseable {
        private final String runId;
        private Map<String, Object> outputs;
        private String error;
        private boolean closed;

        private TraceSpan(String runId) {
            this.runId = runId;
        }

        public String runId() {
            return runId;
        }

        public void recordOutputs(Map<String, Object> outputs) {
            this.outputs = outputs;
        }

        public void recordError(Throwable error) {
            this.error = error == null ? null : error.toString();
        }

        @Override
        public void close() {
            if (closed) {
                return;
            }
            closed = true;

            Deque<String> stack = runStack.get();
            if (!stack.isEmpty() && runId.equals(stack.peek())) {
                stack.pop();
            } else {
                stack.remove(runId);
            }

            if (enabled) {
                patchRun(runId, outputs, error);
            }
        }
    }

    @FunctionalInterface
    public interface ThrowingSupplier<T> {
        T get() throws Exception;
    }

    private void createRun(String runId,
                           String parentRunId,
                           String name,
                           String runType,
                           Map<String, Object> inputs) {
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("id", runId);
        payload.put("name", name);
        payload.put("run_type", runType);
        payload.put("inputs", sanitizeMap(inputs));
        payload.put("start_time", Instant.now().toString());
        payload.put("session_name", projectName);
        payload.put("extra", sanitizeMap(Map.of("metadata", Map.of("source", "langgraph4j-java"))));
        if (parentRunId != null) {
            payload.put("parent_run_id", parentRunId);
        }
        send("POST", runsUrl(), payload);
    }

    private void patchRun(String runId, Map<String, Object> outputs, String error) {
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("end_time", Instant.now().toString());
        if (outputs != null) {
            payload.put("outputs", sanitizeMap(outputs));
        }
        if (error != null && !error.isBlank()) {
            payload.put("error", error);
        }
        send("PATCH", runsUrl() + "/" + runId, payload);
    }

    private void send(String method, String url, Map<String, Object> payload) {
        try {
            RequestBody body = RequestBody.create(json.writeValueAsBytes(payload), JSON);
            Request.Builder builder = new Request.Builder()
                    .url(url)
                    .addHeader("x-api-key", apiKey)
                    .addHeader("Content-Type", "application/json");

            if (workspaceId != null && !workspaceId.isBlank()) {
                builder.addHeader("x-tenant-id", workspaceId);
            }

            Request request = builder.method(method, body).build();
            try (Response response = http.newCall(request).execute()) {
                if (!response.isSuccessful()) {
                    String errorBody = response.body() == null ? "" : response.body().string();
                    log.warn("LangSmith {} {} failed: {} {}", method, url, response.code(), errorBody);
                }
            }
        } catch (IOException e) {
            log.warn("LangSmith request failed: {}", e.toString());
        }
    }

    private Map<String, Object> toOutputs(Object result) {
        if (result instanceof Map<?, ?> map) {
            Map<String, Object> output = new LinkedHashMap<>();
            for (Map.Entry<?, ?> entry : map.entrySet()) {
                output.put(String.valueOf(entry.getKey()), entry.getValue());
            }
            return sanitizeMap(output);
        }
        Map<String, Object> output = new LinkedHashMap<>();
        output.put("result", result);
        return sanitizeMap(output);
    }

    private String currentParentRunId() {
        Deque<String> stack = runStack.get();
        return stack.isEmpty() ? null : stack.peek();
    }

    private String runsUrl() {
        return endpoint + "/runs";
    }

    private Map<String, Object> sanitizeMap(Map<String, Object> input) {
        Map<String, Object> sanitized = new LinkedHashMap<>();
        if (input == null) {
            return sanitized;
        }
        for (Map.Entry<String, Object> entry : input.entrySet()) {
            if (entry.getKey() == null || entry.getValue() == null) {
                continue;
            }
            sanitized.put(entry.getKey(), sanitizeValue(entry.getValue()));
        }
        return sanitized;
    }

    private Object sanitizeValue(Object value) {
        if (value instanceof Map<?, ?> map) {
            Map<String, Object> nested = new LinkedHashMap<>();
            for (Map.Entry<?, ?> entry : map.entrySet()) {
                if (entry.getKey() == null || entry.getValue() == null) {
                    continue;
                }
                nested.put(String.valueOf(entry.getKey()), sanitizeValue(entry.getValue()));
            }
            return nested;
        }
        if (value instanceof Collection<?> collection) {
            return collection.stream()
                    .filter(item -> item != null)
                    .map(this::sanitizeValue)
                    .toList();
        }
        return value;
    }

    private static boolean isTracingEnabled() {
        String value = System.getenv("LANGSMITH_TRACING");
        if (value == null || value.isBlank()) {
            return false;
        }
        return value.equalsIgnoreCase("true") || value.equals("1");
    }

    private static String getenv(String key, String fallback) {
        String value = System.getenv(key);
        return (value == null || value.isBlank()) ? fallback : value;
    }

    private static String stripTrailingSlash(String value) {
        if (value == null) {
            return "";
        }
        return value.endsWith("/") ? value.substring(0, value.length() - 1) : value;
    }
}
