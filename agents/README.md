# Agents Service README

`agents/` is the agent-service layer of TravelMind. It is responsible for:

- Multi-agent orchestration (LangGraph)
- HTTP-based specialist agent invocation and aggregation
- Input/output security guardrails
- A unified API surface consumed by `backend`

---

## 1) Current Directory Structure

```text
agents/
├── __init__.py
├── README.md
├── Dockerfile
├── requirements.txt
├── main.py
├── graph.py
├── nodes.py
├── state.py
├── llm_config.py
├── agent_tools.py
├── prompts/
│   ├── security_policy.yaml
│   └── security_prompts.yaml
├── specialists/
│   ├── __init__.py
│   ├── input_guard_agent.py
│   ├── intent_profile.py
│   ├── research_agent.py
│   ├── planner_agent.py
│   ├── explainability_agent.py
│   └── output_guard_agent.py
└── tests/
    ├── intent/
    ├── orchestrator/
    └── security/
```

---

## 2) Core Files

### `main.py`

FastAPI service entry point. Exposes:

- `GET /health`
- `POST /api/invoke/input_guard`
- `POST /api/invoke/intent_profile`
- `POST /api/invoke/search`
- `POST /api/invoke/planner`
- `POST /api/invoke/debate`
- `POST /api/invoke/explain`
- `POST /api/invoke/output_guard`
- `POST /api/invoke/replanner`

All invoke endpoints use a unified request body:

```json
{
  "state": {}
}
```

Each endpoint also runs `_enforce_agent_port()` to prevent requests from reaching the wrong agent port.

### `graph.py`

Defines and builds the LangGraph workflow:

- `build_travel_graph()`
- `run_cli()` (local interactive debugging entrypoint)

Primary flow:

`START -> input_guard -> orchestrator -> (intent/search/planner/debate/explain/output_guard) -> END`

### `nodes.py`

- Wraps remote calls for each node via `call_remote_agent`
- Implements orchestrator decision logic in `orchestrator_node()`
- Implements conditional routing output via `orchestrator_routing()`

Agent endpoint mapping is managed by `AGENT_URLS`, defaulting to:

- `http://localhost:8100` to `http://localhost:8107`

Each URL can be overridden via environment variables (for example, `AGENT_PLANNER_URL`).

### `state.py`

Defines the shared `State (TypedDict)`, including:

- User intent fields (`origin`, `destination`, `dates`, `budget`, etc.)
- Security status (`threat_blocked`, `output_flagged`)
- Research/planning outputs (`search_results`, `itineraries`)
- Orchestration control fields (`next_node`, `is_valid`, `debate_count`, `error_message`)

---

## 3) Specialists

The current `specialists/` implementations are:

- `input_guard_agent.py`: input security validation
- `intent_profile.py`: user intent structuring
- `research_agent.py`: research and information retrieval aggregation
- `planner_agent.py`: itinerary generation and revision
- `explainability_agent.py`: explanation generation
- `output_guard_agent.py`: output safety validation

`specialists/__init__.py` exports the main specialist interfaces.

---

## 4) Run Locally

From repository root:

```bash
docker compose up --build agents
```

Health check:

```bash
curl http://localhost:8001/health
```

---

## 5) Development Notes

- `main.py` enforces strict per-agent port checks. If an invoke call returns `409`, verify that the request is sent to the expected port.
- `orchestrator_node()` in `nodes.py` relies on field presence/emptiness in `state`; update `state.py` and related tests together when changing routing fields.
- `tests/orchestrator`, `tests/intent`, and `tests/security` cover the key orchestration and guardrail paths. Run these suites after flow-related changes.
