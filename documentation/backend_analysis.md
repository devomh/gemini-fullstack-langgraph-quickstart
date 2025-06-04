# Backend Architecture of Gemini Fullstack LangGraph Quickstart

This document provides a deep dive into how the backend leverages **LangGraph** and **Google Gemini** models to implement a research‑augmented conversational AI. The backend code resides in [`backend/src/agent`](../backend/src/agent) and is exposed through a FastAPI application. Below is a breakdown of the key components and their interactions.

## 1. Configuration

The `Configuration` class in [`configuration.py`](../backend/src/agent/configuration.py) collects settings for model names and loop parameters. Values can come from environment variables or the runtime `RunnableConfig` provided by LangGraph:

```python
class Configuration(BaseModel):
    query_generator_model: str = "gemini-2.0-flash"
    reflection_model: str = "gemini-2.5-flash-preview-04-17"
    answer_model: str = "gemini-2.5-pro-preview-05-06"
    number_of_initial_queries: int = 3
    max_research_loops: int = 2
    @classmethod
    def from_runnable_config(cls, config: Optional[RunnableConfig] = None) -> "Configuration":
        # merges env vars and config parameters
```
【F:backend/src/agent/configuration.py†L1-L47】

This configuration determines how many initial search queries to generate, the allowed number of research loops, and which Gemini models are used for each stage of the agent.

## 2. Data Structures

`state.py` defines several `TypedDict` classes describing the evolving graph state, e.g. `OverallState`, `QueryGenerationState`, and `ReflectionState`:

```python
class OverallState(TypedDict):
    messages: Annotated[list, add_messages]
    search_query: Annotated[list, operator.add]
    web_research_result: Annotated[list, operator.add]
    sources_gathered: Annotated[list, operator.add]
    initial_search_query_count: int
    max_research_loops: int
    research_loop_count: int
    reasoning_model: str
```
【F:backend/src/agent/state.py†L1-L20】

These structures help LangGraph merge state across nodes and maintain counters for search loops.

## 3. Utility Functions

The helper functions in [`utils.py`](../backend/src/agent/utils.py) perform tasks such as building a concatenated conversation history, resolving Google Search result URLs into shortened forms, and injecting citation markers into generated text. Citation extraction is implemented in `get_citations`:

```python
for support in candidate.grounding_metadata.grounding_supports:
    citation = {}
    start_index = support.segment.start_index or 0
    if support.segment.end_index is None:
        continue
    citation["start_index"] = start_index
    citation["end_index"] = support.segment.end_index
    citation["segments"] = []
    if support.grounding_chunk_indices:
        for ind in support.grounding_chunk_indices:
            chunk = candidate.grounding_metadata.grounding_chunks[ind]
            resolved_url = resolved_urls_map.get(chunk.web.uri, None)
            citation["segments"].append({
                "label": chunk.web.title.split(".")[:-1][0],
                "short_url": resolved_url,
                "value": chunk.web.uri,
            })
```
【F:backend/src/agent/utils.py†L52-L96】

The final citations are inserted back into the response text before being returned to the graph.

## 4. Prompts

`prompts.py` defines template strings for each stage. For example, `query_writer_instructions` guides Gemini to produce a short list of web search queries along with a rationale. The `reflection_instructions` ask the model to analyze gathered summaries and suggest follow-up queries if more information is needed.

## 5. Graph Nodes

The heart of the system is implemented in [`graph.py`](../backend/src/agent/graph.py). A `StateGraph` is assembled with four main nodes:

1. **generate_query** – Creates a set of search queries using Gemini and structured output (`SearchQueryList`).
2. **web_research** – Performs Google web searches via the Gemini model and inserts citation markers into the results.
3. **reflection** – Reviews current summaries to decide if the answer is sufficient and, if not, produces follow-up queries (`Reflection` structure).
4. **finalize_answer** – Synthesizes the final answer using the reasoning model, replacing short URLs with the original sources.

The code snippet below shows the decision logic controlling the research loop:

```python
def evaluate_research(state: ReflectionState, config: RunnableConfig) -> OverallState:
    configurable = Configuration.from_runnable_config(config)
    max_research_loops = state.get("max_research_loops") or configurable.max_research_loops
    if state["is_sufficient"] or state["research_loop_count"] >= max_research_loops:
        return "finalize_answer"
    else:
        return [
            Send("web_research", {"search_query": follow_up_query, "id": state["number_of_ran_queries"] + int(idx)})
            for idx, follow_up_query in enumerate(state["follow_up_queries"])
        ]
```
【F:backend/src/agent/graph.py†L148-L173】

This routing function either continues with additional web research or transitions to `finalize_answer` once the loop limit is reached or sufficient information has been gathered.

The complete graph is assembled using `StateGraph`:

```python
builder = StateGraph(OverallState, config_schema=Configuration)
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("reflection", reflection)
builder.add_node("finalize_answer", finalize_answer)

builder.add_edge(START, "generate_query")
builder.add_conditional_edges("generate_query", continue_to_web_research, ["web_research"])
builder.add_edge("web_research", "reflection")
builder.add_conditional_edges("reflection", evaluate_research, ["web_research", "finalize_answer"])
builder.add_edge("finalize_answer", END)

graph = builder.compile(name="pro-search-agent")
```
【F:backend/src/agent/graph.py†L188-L214】

## 6. FastAPI Integration

`app.py` mounts a small FastAPI application. It can serve a built React frontend under `/app` and exposes the LangGraph HTTP API defined in `langgraph.json`:

```json
{
  "graphs": { "agent": "./src/agent/graph.py:graph" },
  "http": { "app": "./src/agent/app.py:app" },
  "env": ".env"
}
```
【F:backend/langgraph.json†L1-L11】

When `langgraph dev` or `langgraph run` is executed, this JSON file tells LangGraph where to load the graph and HTTP application. The React frontend communicates with the backend using the LangGraph SDK’s streaming APIs.

## 7. End‑to‑End Flow

1. **User submits a question from the frontend**. The frontend sends the conversation history and configuration values (effort level, reasoning model) to the LangGraph endpoint.
2. **`generate_query`** uses Gemini to produce a list of search queries aimed at answering the user’s question.
3. **`web_research`** runs each search query through the Google Search tool integrated in Gemini models, returning short summaries with embedded citations.
4. **`reflection`** analyzes collected summaries. If the information is lacking, it generates follow-up queries; otherwise it marks the search as sufficient.
5. **The loop repeats** until either the maximum number of research loops is reached or the summaries are deemed sufficient.
6. **`finalize_answer`** crafts the final response, replacing shortened URLs with full links and yielding an AI message containing the answer and citations.
7. **Streaming updates** are sent back to the frontend during each node execution, enabling real-time UI updates.

## 8. Key Takeaways

- **LangGraph** orchestrates complex multi-step reasoning, letting the agent loop through search and reflection until confidence is achieved.
- **Google Gemini models** handle query generation, summarization, and reflection, with built-in Google Search tooling for factual grounding.
- **Utility functions** manage conversation context and transform Gemini's grounding metadata into inline citations.
- **FastAPI** serves as the lightweight web server, delivering both the API and optionally the compiled React frontend.

This architecture demonstrates how LangGraph can coordinate iterative research tasks to provide well-sourced conversational answers in a fullstack application.
