import json
import types
import sys
import importlib
from pathlib import Path

import pytest


def _load_research_agent_module():
    module_path = Path(__file__).resolve().parents[2] / "specialists" / "research_agent.py"

    # Prevent heavy optional dependency chain during import:
    # langchain_openai -> langchain_core -> transformers -> torch
    if "langchain_openai" not in sys.modules:
        lc_openai_mod = types.ModuleType("langchain_openai")

        class _ImportSafeChatOpenAI:
            def __init__(self, *args, **kwargs):
                pass

            def invoke(self, messages):
                return _DummyLLMResponse('{"remove_ids": []}')

        lc_openai_mod.ChatOpenAI = _ImportSafeChatOpenAI
        sys.modules["langchain_openai"] = lc_openai_mod

    if "langchain_core.messages" not in sys.modules:
        lc_messages_mod = types.ModuleType("langchain_core.messages")

        class _SystemMessage:
            def __init__(self, content):
                self.content = content

        lc_messages_mod.SystemMessage = _SystemMessage
        sys.modules["langchain_core.messages"] = lc_messages_mod

    if "agents" not in sys.modules:
        agents_pkg = types.ModuleType("agents")
        agents_pkg.__path__ = [str(module_path.parents[1])]
        sys.modules["agents"] = agents_pkg
    if "agents.specialists" not in sys.modules:
        specialists_pkg = types.ModuleType("agents.specialists")
        specialists_pkg.__path__ = [str(module_path.parent)]
        sys.modules["agents.specialists"] = specialists_pkg
    if "agents.llm_config" not in sys.modules:
        llm_cfg = types.ModuleType("agents.llm_config")
        llm_cfg.OPENAI_MODEL = "test-model"
        sys.modules["agents.llm_config"] = llm_cfg

    module_name = "agents.specialists.research_agent"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from: {module_path}")
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "agents.specialists"
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


ra = _load_research_agent_module()


class _DummyLLMResponse:
    def __init__(self, content: str):
        self.content = content


class _DummyLLM:
    def __init__(self, *args, **kwargs):
        pass

    def invoke(self, messages):
        return _DummyLLMResponse('{"remove_ids": []}')


@pytest.fixture
def fake_tools_modules(monkeypatch):
    tools_pkg = types.ModuleType("tools")
    serp_mod = types.ModuleType("tools.serp_search")
    places_mod = types.ModuleType("tools.google_places_search")
    web_mod = types.ModuleType("tools.web_search")
    google_mod = types.ModuleType("tools.google_search")

    serp_mod._flight_options_outbound = []
    serp_mod._flight_options_return = []
    serp_mod._hotel_options = []
    serp_mod._hotel_tokens = {}
    serp_mod._place_data_ids = {}

    def serp_local_structured(dest, topic):
        if "restaurant" in topic.lower():
            return [{"name": f"{dest} Noodle House", "type": "restaurant", "lat": 1.30, "lng": 103.80}]
        return [{"name": f"{dest} Museum", "type": "museum", "lat": 1.31, "lng": 103.81, "price": "USD 20"}]

    def serp_tripadvisor_structured(topic, dest):
        if "restaurant" in topic.lower():
            return [{"name": f"{dest} TA Eatery", "type": "restaurant", "lat": 1.301, "lng": 103.801}]
        return [{"name": f"{dest} TA Garden", "type": "attraction", "lat": 1.311, "lng": 103.811, "price": "USD 15"}]

    def serp_hotel_details(token, outbound_date, return_date):
        return "hotel details"

    def serp_hotel_reviews(token, num=3):
        return "hotel reviews"

    def serp_maps_reviews(name, num=3):
        return "maps reviews"

    serp_mod.serp_local_structured = serp_local_structured
    serp_mod.serp_tripadvisor_structured = serp_tripadvisor_structured
    serp_mod.serp_hotel_details = serp_hotel_details
    serp_mod.serp_hotel_reviews = serp_hotel_reviews
    serp_mod.serp_maps_reviews = serp_maps_reviews

    def google_places_search_bundle(query, _unused, place_kind="attraction", max_results=5):
        rows = []
        for i in range(max_results):
            if place_kind == "restaurant":
                rows.append(
                    {
                        "id": f"rest-{i}",
                        "name": f"Resto {i}",
                        "type": "restaurant",
                        "rating": 4.4,
                        "price": "USD 12",
                        "address": "Downtown",
                        "lat": 1.30 + i * 0.0001,
                        "lng": 103.80 + i * 0.0001,
                    }
                )
            else:
                rows.append(
                    {
                        "id": f"att-{i}",
                        "name": f"Attraction {i}",
                        "type": "attraction",
                        "rating": 4.6,
                        "price": "USD 25",
                        "address": "City Center",
                        "lat": 1.31 + i * 0.0001,
                        "lng": 103.81 + i * 0.0001,
                    }
                )
        return {"original_query": query, "selected_places": rows}

    places_mod.google_places_search_bundle = google_places_search_bundle
    web_mod.web_search = lambda query: "travel guide text"
    google_mod.google_search = lambda query: "google fallback text"

    monkeypatch.setitem(sys.modules, "tools", tools_pkg)
    monkeypatch.setitem(sys.modules, "tools.serp_search", serp_mod)
    monkeypatch.setitem(sys.modules, "tools.google_places_search", places_mod)
    monkeypatch.setitem(sys.modules, "tools.web_search", web_mod)
    monkeypatch.setitem(sys.modules, "tools.google_search", google_mod)
    monkeypatch.setattr(ra, "ChatOpenAI", _DummyLLM)

    return serp_mod


def test_slugify_and_candidate_id():
    assert ra._slugify(" Kyoto / Japan 2026 ") == "kyoto_japan_2026"
    assert ra._slugify("") == "item"
    assert ra._candidate_id({"place_id": "p1"}) == "p1"
    assert ra._candidate_id({"id": "x"}) == "x"
    assert ra._candidate_id({"name": "A"}) == "A"


def test_merge_and_dedupe_candidates():
    a = {"name": "Gion", "price": "", "rating": 4.5}
    b = {"name": "Gion District", "price": "USD 20", "address": "Kyoto"}
    merged = ra._merge_place_candidate(a.copy(), b)
    assert merged["price"] == "USD 20"
    assert merged["address"] == "Kyoto"

    rows = [a, b, {"name": "Nijo Castle", "price": "USD 10"}]
    out = ra._dedupe_place_candidates(rows)
    assert len(out) == 2


def test_debug_dir_and_save_debug_file(tmp_path, monkeypatch):
    monkeypatch.setenv("GOOGLE_PLACES_DEBUG_SAVE", "1")
    monkeypatch.setenv("GOOGLE_PLACES_DEBUG_DIR", str(tmp_path))
    assert ra._places_debug_enabled() is True
    out_dir = ra._places_debug_dir()
    assert out_dir.exists()

    ra._save_places_group_debug(
        place_kind="attractions",
        city="Kyoto",
        preferences_text="culture",
        queries=["best attractions in kyoto"],
        max_output=10,
        min_keep=8,
        raw_candidates=[{"name": "A"}],
        llm_filtered_candidates=[{"name": "A"}],
        removed_ids=[],
        skipped_llm_filter=True,
    )
    files = list(tmp_path.glob("*.json"))
    assert files, "expected debug json file to be written"
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["city"] == "Kyoto"


def test_llm_filter_places_skip_path():
    tool_log = []
    rows = [{"id": "a1", "name": "A1"}]
    kept, removed, skipped = ra._llm_filter_places(
        place_kind="attractions",
        candidates=rows,
        queries=["q"],
        preferences_text="culture",
        city="Kyoto",
        max_output=12,
        tool_log=tool_log,
    )
    assert kept == rows
    assert removed == []
    assert skipped is True
    assert any("skipped" in line for line in tool_log)


def test_llm_filter_places_success_nothing_to_remove_and_error(monkeypatch):
    # nothing_to_remove branch: raw == min_keep
    tool_log = []
    rows = [{"id": f"a{i}", "name": f"A{i}", "category": "museum"} for i in range(4)]
    kept, removed, skipped = ra._llm_filter_places(
        place_kind="attractions",
        candidates=rows,
        queries=["q"],
        preferences_text="culture",
        city="Kyoto",
        max_output=5,  # min_keep=4, max_remove=0
        tool_log=tool_log,
    )
    assert kept == rows
    assert removed == []
    assert skipped is True
    assert any("nothing_to_remove" in line for line in tool_log)

    # success branch with bounded remove ids
    class _RemoveSomeLLM:
        def __init__(self, *args, **kwargs):
            pass

        def invoke(self, messages):
            return _DummyLLMResponse('{"remove_ids": ["a0", "missing", "a0", "a1"]}')

    monkeypatch.setattr(ra, "ChatOpenAI", _RemoveSomeLLM)
    tool_log = []
    rows = [{"id": f"a{i}", "name": f"A{i}", "category": "museum"} for i in range(10)]
    kept, removed, skipped = ra._llm_filter_places(
        place_kind="attractions",
        candidates=rows,
        queries=["q"],
        preferences_text="culture",
        city="Kyoto",
        max_output=10,  # min_keep=8, max_remove=2
        tool_log=tool_log,
    )
    assert skipped is False
    assert removed == ["a0", "a1"]
    assert len(kept) == 8
    assert any("removed=2" in line for line in tool_log)

    # error branch
    class _ErrorLLM:
        def __init__(self, *args, **kwargs):
            pass

        def invoke(self, messages):
            raise RuntimeError("boom")

    monkeypatch.setattr(ra, "ChatOpenAI", _ErrorLLM)
    tool_log = []
    kept, removed, skipped = ra._llm_filter_places(
        place_kind="restaurants",
        candidates=rows,
        queries=["q"],
        preferences_text="food",
        city="Kyoto",
        max_output=10,
        tool_log=tool_log,
    )
    assert kept == rows
    assert removed == []
    assert skipped is True
    assert any("error:" in line for line in tool_log)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("$30", "SGD 30"),
        ("JPY 1000", "SGD 9"),
        ("AUD 20", "SGD 18"),
        ("free", "free"),
    ],
)
def test_price_conversion(raw, expected):
    assert ra._to_sgd(raw) == expected


def test_safe_price_caps_large_values():
    assert ra._safe_price("USD 1000") == "TBC"
    assert ra._safe_price("USD 40").startswith("SGD ")
    assert ra._safe_price("") == ""


def test_duration_dietary_and_city_inference():
    assert ra._duration_days("5 days") == 5
    assert ra._duration_days("") == 1
    assert ra._detect_dietary_style("vegetarian friendly") == "vegetarian"
    assert ra._detect_dietary_style("halal food") == "halal"
    assert ra._detect_dietary_style("kosher options") == "kosher"
    assert ra._detect_dietary_style("gluten free meals") == "gluten-free"
    assert ra._detect_dietary_style("plant based") == "vegan"
    assert ra._detect_dietary_style("no dietary keyword") == ""
    assert ra._infer_query_city(["Best food in Kyoto for couples"], "Tokyo") == "Kyoto"
    assert ra._infer_query_city([], "Tokyo") == "Tokyo"


def test_geo_related_helpers():
    assert ra._safe_float("1.23") == 1.23
    assert ra._safe_float("bad") is None
    assert ra._haversine_km(1.3, 103.8, 1.3, 103.8) == pytest.approx(0.0, abs=0.001)
    assert ra._place_latlng({"lat": "1.3", "lng": "103.8"}) == (1.3, 103.8)
    tokens = ra._destination_tokens("Best places in Kyoto city")
    assert "kyoto" in tokens
    assert "city" not in tokens
    assert ra._looks_like_destination_text({"name": "Kyoto Museum"}, {"kyoto"}) is True
    assert ra._dominant_geo_center([{"name": "x"}]) is None
    assert ra._is_food_place({"type": "restaurant"}) is True
    assert ra._is_corporate_noise({"name": "Acme Pte Ltd"}) is True
    assert ra._strip_dollar("$120") == "120"
    # text-only filtering path (no lat/lng), covers destination text checks
    out = ra._filter_candidates_to_destination_cluster(
        [{"name": "Kyoto Old Town", "address": "Kyoto", "lat": None, "lng": None}],
        place_kind="attractions",
        target_city="Kyoto",
        destination="Kyoto",
        tool_log=[],
        reference_candidates=[{"name": "No geo"}],
    )
    assert len(out) == 1


def test_cluster_filter_and_query_plan():
    tool_log = []
    rows = [
        {"name": "Near A", "lat": 1.3000, "lng": 103.8000},
        {"name": "Near B", "lat": 1.3002, "lng": 103.8002},
        {"name": "Far C", "lat": 35.0, "lng": 139.0},
    ]
    filtered = ra._filter_candidates_to_destination_cluster(
        rows,
        place_kind="attractions",
        target_city="Singapore",
        destination="Singapore",
        tool_log=tool_log,
        cluster_radius_km=5.0,
    )
    assert len(filtered) == 2
    assert any("destination_geofence" in line for line in tool_log)

    plan = ra._allocate_query_plan(
        [" q1 "],
        total_quota=25,
        max_per_query=12,
        fallback_query_builder=lambda idx: f"fallback-{idx}",
    )
    assert len(plan) >= 2
    assert all(q and quota >= 1 for q, quota in plan)

    plan2 = ra._allocate_query_plan(
        [],
        total_quota=30,
        max_per_query=8,
        fallback_query_builder=lambda idx: f"fb-{idx}",
    )
    assert len(plan2) >= 4
    assert all(quota <= 8 for _, quota in plan2)


def test_normalize_trip_state_and_display_helpers():
    state = {
        "hard_constraints": {
            "origin": "SIN",
            "destination": "Kyoto",
            "start_date": "2026-05-01",
            "end_date": "2026-05-05",
            "requirements": ["vegetarian"],
            "budget": {"currency": "SGD", "amount": 3000},
        },
        "soft_preferences": {"interest_tags": ["culture"], "travel_style": "relaxed"},
    }
    norm = ra._normalize_trip_state(state)
    assert norm["origin"] == "SIN"
    assert norm["destination"] == "Kyoto"
    assert norm["duration"] == "5 days"
    assert "vegetarian" in norm["preferences"]

    flight_display = ra._build_flight_display(
        {
            "airline": "SQ",
            "flight_number": "SQ618",
            "departure_airport": "SIN",
            "arrival_airport": "KIX",
            "departure_time": "08:30",
            "arrival_time": "16:10",
            "duration_min": 460,
            "travel_class": "economy",
            "price_usd": "500",
        }
    )
    assert "SQ SQ618" in flight_display

    hotel_display = ra._build_hotel_display(
        {"name": "Hotel A", "hotel_class": "4-star", "rating": 4.3, "price_per_night_usd": "100"}
    )
    assert "Hotel A" in hotel_display
    assert ra._build_hotel_display({"display": "preformatted"}) == "preformatted"

    norm2 = ra._normalize_trip_state(
        {
            "origin": "SIN",
            "destination": "Tokyo",
            "dates": "2026-01-01 to 2026-01-02",
            "budget": {"currency": "USD", "amount": 1000, "flexibility": "strict"},
            "preferences": "",
            "hard_constraints": {"requirements": ["quiet"]},
            "soft_preferences": {"interest_tags": ["museum"], "vibe": "slow"},
        }
    )
    assert "USD 1000 strict" in norm2["budget"]
    assert "quiet" in norm2["preferences"]

    # invalid dates -> duration fallback exception branch
    norm3 = ra._normalize_trip_state(
        {
            "hard_constraints": {"start_date": "bad-date", "end_date": "also-bad"},
            "soft_preferences": {},
        }
    )
    assert norm3["duration"] == ""


def test_flight_sort_helpers():
    f = {
        "display": "1 stop",
        "departure_time": "13:20",
        "price_usd": "300",
        "duration_min": "400",
    }
    assert ra._flight_stop_rank({"display": "Nonstop flight"}) == 0
    assert ra._flight_stop_rank(f) == 1
    assert ra._time_pref_penalty({"departure_time": "09:10"}, "morning") == 0
    assert ra._time_pref_penalty({"departure_time": "20:10"}, "morning") == 1
    assert ra._time_pref_penalty({"departure_time": "20:10"}, "evening") == 0
    assert ra._time_pref_penalty({"departure_time": "ab:cd"}, "morning") == 0
    sort_key = ra._flight_sort_key(f, "afternoon")
    assert isinstance(sort_key, tuple)
    assert len(sort_key) == 5
    assert ra._flight_sort_key({"price_usd": "x", "duration_min": "x"}, "")[2] == 999999
    assert ra._flight_stop_rank({"display": "mystery"}) == 9


def test_price_conversion_edge_error_branches():
    # S$ parse failure
    assert ra._to_sgd("S$not-a-number") == "S$not-a-number"
    # dollar range branch with invalid pieces
    assert ra._to_sgd("$abc-$def") == "$abc-$def"
    # single dollar parse failure
    assert ra._to_sgd("$abc") == "$abc"
    # fx parse failure
    assert ra._to_sgd("AUD xyz") == "AUD xyz"


def test_research_agent_end_to_end_with_mocked_dependencies(fake_tools_modules):
    serp_mod = fake_tools_modules

    def search_flights(origin, dest, date, time_pref, direction):
        target = (
            serp_mod._flight_options_outbound
            if direction == "outbound"
            else serp_mod._flight_options_return
        )
        target.extend(
            [
                {
                    "airline": "SQ",
                    "flight_number": f"SQ-{direction}-1",
                    "departure_airport": origin,
                    "arrival_airport": dest,
                    "departure_time": "08:30",
                    "arrival_time": "16:30",
                    "duration_min": 480,
                    "travel_class": "economy",
                    "price_usd": 500,
                    "display": "nonstop",
                }
            ]
        )
        return "ok"

    def search_hotels(dest, budget, dates):
        serp_mod._hotel_options.extend(
            [
                {
                    "name": "City Hotel",
                    "hotel_class": "4-star",
                    "rating": 4.5,
                    "price_per_night_usd": "120",
                    "description": "central",
                    "display": "City Hotel | 4-star | rating 4.5",
                }
            ]
        )
        serp_mod._hotel_tokens["City Hotel"] = "token-1"
        return "ok"

    state = {
        "origin": "Singapore",
        "destination": "Kyoto",
        "dates": "2026-05-01 to 2026-05-03",
        "duration": "3 days",
        "budget": "SGD 3000",
        "preferences": "culture, vegetarian",
        "search_queries": [
            {"type": "rag_attraction", "query": "Best attractions in Kyoto"},
            {"type": "rag_restaurant", "query": "Best vegetarian restaurants in Kyoto"},
        ],
        "user_profile": {"prefs": ["low-intensity"]},
    }
    tools = {
        "search_weather": lambda dest, dates: "Sunny",
        "search_flights": search_flights,
        "search_hotels": search_hotels,
        "web_search": lambda q: "Kyoto guide",
    }

    result = ra.research_agent(state, tools=tools)

    assert "inventory" in result
    assert "research" in result
    assert len(result["inventory"]["attractions"]) >= 1
    assert len(result["inventory"]["restaurants"]) >= 1
    assert len(result["inventory"]["hotels"]) >= 1
    assert len(result["inventory"]["flights_outbound"]) >= 1
    assert len(result["inventory"]["flights_return"]) >= 1
    assert any("research_agent_1" in line for line in result["tool_log"])


def test_research_agent_non_explicit_path_and_pref_collection(fake_tools_modules):
    state = {
        "origin": "Singapore",
        "destination": "Kyoto",
        "dates": "2026-05-01 to 2026-05-04",
        "duration": "4 days",
        "budget": "SGD 3000",
        "preferences": "culture, museums",
        "search_queries": [],
        "user_profile": {"prefs": ["vegetarian"]},
    }
    # No external tools: exercise no-op branches for weather/web/flight/hotel callbacks
    result = ra.research_agent(state, tools={})

    assert "maps_attractions" in result["research"]
    assert "maps_restaurants" in result["research"]
    assert "pref_structured" in result["research"]
    assert result["research"]["weather"] == ""
    assert result["research"]["web_general"] == ""
    assert any("maps_structured(" in line for line in result["tool_log"])
    assert any("pref_search(" in line for line in result["tool_log"])


def test_research_agent_supplement_and_price_enrichment(fake_tools_modules, monkeypatch):
    serp_mod = fake_tools_modules
    places_mod = sys.modules["tools.google_places_search"]
    web_mod = sys.modules["tools.web_search"]
    google_mod = sys.modules["tools.google_search"]

    def sparse_places_bundle(query, _unused, place_kind="attraction", max_results=5):
        q = query.lower()
        if place_kind == "restaurant":
            # Initial explicit restaurant queries return no data -> trigger supplement
            if "vegetarian restaurants in kyoto" in q:
                return {"original_query": query, "selected_places": []}
            place_name = f"Kyoto Local Bistro {abs(hash(query)) % 1000}"
            return {
                "original_query": query,
                "selected_places": [
                    {
                        "id": "supp-rest-1",
                        "name": place_name,
                        "type": "restaurant",
                        "rating": 4.2,
                        "price": "USD 18",
                        "lat": 35.01,
                        "lng": 135.75,
                    }
                ],
            }
        # Attractions with missing price -> trigger price enrichment
        return {
            "original_query": query,
            "selected_places": [
                {
                    "id": "att-alpha",
                    "name": "Attraction Alpha",
                    "type": "attraction",
                    "rating": 4.7,
                    "price": "",
                    "address": "Kyoto Center",
                    "lat": 35.02,
                    "lng": 135.76,
                },
                {
                    "id": "att-beta",
                    "name": "Attraction Beta",
                    "type": "attraction",
                    "rating": 4.5,
                    "price": None,
                    "address": "Kyoto East",
                    "lat": 35.03,
                    "lng": 135.77,
                },
            ],
        }

    places_mod.google_places_search_bundle = sparse_places_bundle
    web_mod.web_search = lambda q: '[{"title":"Attraction Alpha","snippet":"ticket is USD 30"}]'
    google_mod.google_search = lambda q: '{"answer":"Attraction Beta costs SGD 40"}'

    class _PriceAwareLLM:
        def __init__(self, *args, **kwargs):
            pass

        def invoke(self, messages):
            prompt = getattr(messages[0], "content", "") if messages else ""
            if "Attraction Name" in prompt:
                return _DummyLLMResponse(
                    '{"Attraction Alpha":"USD 30","Attraction Beta":"SGD 40","Unknown":"tbc"}'
                )
            return _DummyLLMResponse('{"remove_ids":[]}')

    monkeypatch.setattr(ra, "ChatOpenAI", _PriceAwareLLM)

    # Enable place reviews branch; research_agent() clears this map on entry,
    # so provide a dict-like object with no-op clear().
    class _PersistentPlaceIds(dict):
        def clear(self):  # pragma: no cover - trivial override for test control
            return None

    serp_mod._place_data_ids = _PersistentPlaceIds(
        {"attraction alpha": "pid-1", "attraction beta": "pid-2"}
    )

    state = {
        "origin": "Singapore",
        "destination": "Kyoto",
        "dates": "2026-05-01 to 2026-05-03",
        "duration": "3 days",
        "budget": "SGD 3000",
        "preferences": "culture, vegetarian",
        "search_queries": [
            {"type": "rag_attraction", "query": "Best attractions in Kyoto"},
            {"type": "rag_restaurant", "query": "Best vegetarian restaurants in Kyoto"},
        ],
    }

    result = ra.research_agent(state, tools={})

    assert any("restaurant_supplement" in line for line in result["tool_log"])
    assert any("price_search" in line for line in result["tool_log"])
    assert any("price_enrichment" in line for line in result["tool_log"])
    assert "place_reviews" in result["research"]
    assert len(result["inventory"]["restaurants"]) >= 1
    # Price got enriched into compact output format
    prices = [item.get("price", "") for item in result["inventory"]["attractions"]]
    assert any(str(p).startswith("SGD ") for p in prices if p)
