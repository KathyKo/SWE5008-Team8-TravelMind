# Agents Module Guide

This directory contains the core multi-agent implementation for TravelMind, including LangGraph orchestration, shared state transitions, and specialist agent collaboration.

## Directory Structure

```text
agents/
├── __init__.py
├── __main__.py
├── README.md
├── agent_tools.py
├── graph.py
├── llm_config.py
├── nodes.py
├── state.py
├── visualize.py
└── specialists/
    ├── __init__.py
    ├── booking_agent.py
    ├── concierge.py
    ├── local_guide.py
    ├── orchestrator.py
    └── summarizer.py
```

## File Responsibilities (agents root)

| File | Responsibility |
|---|---|
| __init__.py | Exposes core agent functions: `travel_orchestrator`, `concierge`, `booking_agent`, `local_guide`, `travel_summarizer` |
| __main__.py | Package entrypoint. Runs CLI conversation flow when executing `python -m agents` |
| llm_config.py | Central model config, provides `OPENAI_MODEL` (default: `gpt-5-mini-2025-08-07`) |
| state.py | Defines shared `State` (user preferences, stage state, search results, confirmation state, etc.) |
| agent_tools.py | Defines per-agent tool allowlist (`TOOLS_BY_AGENT`) and `get_tools_for_agent()` |
| nodes.py | Defines LangGraph node and routing functions: `human_node()`, `orchestrator_node()`, `booking_node()`, etc. |
| graph.py | Builds and compiles the graph: `build_travel_graph()`; CLI runner: `run_cli()` |
| visualize.py | Generates workflow visualization in ASCII / Mermaid / PNG |

## File Responsibilities (agents/specialists)

| File | Responsibility |
|---|---|
| specialists/__init__.py | Exports specialist agent functions |
| specialists/orchestrator.py | `travel_orchestrator()`: routes to the right specialist based on context and stage |
| specialists/concierge.py | `concierge()`: collects core trip info (origin/destination/dates/budget/preferences) |
| specialists/booking_agent.py | `booking_agent()`: stage-based flow for research, itinerary draft, selection parsing, price lookup, and confirmation summary |
| specialists/local_guide.py | `local_guide()`: weather/attractions/activity guidance (does not handle booking prices) |
| specialists/summarizer.py | `travel_summarizer()`: compiles final confirmed travel plan output |

## How to Run

Run from the repository root:

- Start CLI: `python -m agents`
- Visualize workflow: `python -m agents.visualize`

## State and Flow Highlights

- Graph entry: `START -> human -> orchestrator`
- Routing: `orchestrator_routing()` selects `concierge` / `booking_agent` / `local_guide` / `summarizer`
- Exit condition: `check_exit_condition()` detects `exit/quit/done`
- End: `summarizer -> END`

## Tool Permission Model

`agent_tools.py` uses an explicit allowlist:

- `booking_agent`: `web_search`, `search_hotels`, `search_flights`, `google_search`
- `local_guide`: `search_weather`, `search_attractions`

All other agents have no tool access by default.
