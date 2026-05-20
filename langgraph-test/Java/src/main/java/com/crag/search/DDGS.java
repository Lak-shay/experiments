package com.crag.search;

import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;

import java.io.IOException;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Lightweight DuckDuckGo HTML search wrapper.
 *
 * This intentionally avoids the Instant Answer API because it often returns
 * empty results for ordinary web-search queries such as sports questions.
 */
public class DDGS {

    private static final String SEARCH_URL = "https://html.duckduckgo.com/html/?q=%s";
    private static final Pattern RESULT_PATTERN = Pattern.compile(
            "<a[^>]*class=\"result__a\"[^>]*>(.*?)</a>(.*?)(?:<a[^>]*class=\"result__snippet\"[^>]*>(.*?)</a>|<div[^>]*class=\"result__snippet\"[^>]*>(.*?)</div>)",
            Pattern.CASE_INSENSITIVE | Pattern.DOTALL
    );
    private static final Pattern TAG_PATTERN = Pattern.compile("<[^>]+>");
    private static final Pattern WHITESPACE_PATTERN = Pattern.compile("\\s+");
    private static final int MAX_RESULTS = 3;

    private final OkHttpClient http = new OkHttpClient();

    public String search(String query) {
        String encoded = URLEncoder.encode(query, StandardCharsets.UTF_8);
        String url = String.format(SEARCH_URL, encoded);

        Request request = new Request.Builder()
                .url(url)
                .header("User-Agent", "Mozilla/5.0 (CRAG-Agent/1.0)")
                .build();

        try (Response response = http.newCall(request).execute()) {
            if (!response.isSuccessful() || response.body() == null) {
                return "No results found for: " + query;
            }

            String html = response.body().string();
            List<String> snippets = extractSnippets(html);
            if (!snippets.isEmpty()) {
                return String.join("\n", snippets);
            }

            return "No web results found for: " + query;
        } catch (IOException e) {
            return "Web search failed: " + e.getMessage();
        }
    }

    private List<String> extractSnippets(String html) {
        List<String> results = new ArrayList<>();
        Matcher matcher = RESULT_PATTERN.matcher(html);

        while (matcher.find() && results.size() < MAX_RESULTS) {
            String title = normalize(matcher.group(1));
            String snippet = normalize(firstNonBlank(matcher.group(3), matcher.group(4)));

            if (title.isBlank() && snippet.isBlank()) {
                continue;
            }

            if (snippet.isBlank()) {
                results.add(title);
            } else if (title.isBlank()) {
                results.add(snippet);
            } else {
                results.add(title + ": " + snippet);
            }
        }

        return results;
    }

    private String normalize(String value) {
        if (value == null || value.isBlank()) {
            return "";
        }
        String withoutTags = TAG_PATTERN.matcher(value).replaceAll(" ");
        String decoded = withoutTags
                .replace("&nbsp;", " ")
                .replace("&amp;", "&")
                .replace("&quot;", "\"")
                .replace("&#x27;", "'")
                .replace("&#39;", "'")
                .replace("&lt;", "<")
                .replace("&gt;", ">");
        return WHITESPACE_PATTERN.matcher(decoded).replaceAll(" ").trim();
    }

    private String firstNonBlank(String first, String second) {
        if (first != null && !first.isBlank()) {
            return first;
        }
        return second == null ? "" : second;
    }
}
