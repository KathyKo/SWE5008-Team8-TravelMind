"""
Test runner for the new research-first places pipeline.

Supported flows:
- `research`: run `research_agent_1` only
- `plan-from-research`: run `planner_from_research_1` on a saved research result
- `explain-from-plan`: run `explainability_agent` on a saved planner result
- `full`: run `planner_agent_1` + `explainability_agent`

Compatibility aliases:
- `generate` -> `pipeline`
- `local` -> `full`
"""

import argparse
import contextlib
import io
import json
import os
import re
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "backend"))

load_dotenv()

DEFAULT_INPUT_PATH = project_root / "test_inputs" / "places_pipeline_input.json"
DEFAULT_PLACES_DEBUG_DIR = project_root / "debug" / "google_places_raw"


def _print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def _load_test_state(input_path: Path) -> dict:
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_research_result(input_path: Path) -> dict:
    with open(input_path, "r", encoding="utf-8") as f:
        result = json.load(f)
    if not isinstance(result, dict):
        raise ValueError("Research result JSON must be an object")
    required = ["state", "research", "inventory", "tool_log"]
    missing = [key for key in required if key not in result]
    if missing:
        raise ValueError(
            "Input is not a saved research result JSON; missing keys: "
            + ", ".join(missing)
        )
    return result


def _load_plan_result(input_path: Path) -> dict:
    with open(input_path, "r", encoding="utf-8") as f:
        result = json.load(f)
    if not isinstance(result, dict):
        raise ValueError("Plan result JSON must be an object")
    required_any = ["itineraries", "final_itineraries", "validated_itineraries"]
    if not any(isinstance(result.get(key), dict) and result.get(key) for key in required_any):
        raise ValueError(
            "Input is not a saved planner result JSON; expected one of: "
            + ", ".join(required_any)
        )
    if "state" not in result:
        raise ValueError("Saved planner result JSON must contain a `state` object")
    return result


def _normalise_mode(mode: str) -> str:
    return {
        "generate": "pipeline",
        "local": "full",
    }.get(mode, mode)


def _safe_sgd_from_usd(value) -> str:
    if value in (None, "", "?"):
        return "?"
    try:
        return str(round(float(str(value).replace(",", "")) * 1.35))
    except Exception:
        return "?"


def _terminal_text(value) -> str:
    text = str(value or "")
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = (
        text.replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("–", "-")
        .replace("—", "-")
        .replace("→", "->")
        .replace("←", "<-")
    )
    text = "".join(
        ch for ch in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(ch)
    )
    text = "".join(
        ch for ch in text
        if ch == "\n" or ch == "\t" or 32 <= ord(ch) <= 126
    )
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _summarise_itineraries(itineraries: dict) -> list[str]:
    lines: list[str] = []
    for opt, days in itineraries.items():
        if not isinstance(days, list):
            lines.append(f"  Option {opt}: (no days)")
            continue
        lines.append(f"  Option {opt}: {len(days)} days")
        for day in days:
            items = day.get("items", [])
            lines.append(f"    {day.get('day', '?')} - {len(items)} items")
            for item in items:
                lines.append(
                    f"      [{item.get('time', '')}] {_terminal_text(item.get('name', ''))}  {_terminal_text(item.get('cost', ''))}"
                )
    return lines


def _selected_option_key(result: dict) -> str:
    for key in ("selected_option", "explain_option", "option"):
        value = str(result.get(key) or "").strip().upper()
        if value:
            return value
    itineraries = (
        result.get("final_itineraries")
        or result.get("validated_itineraries")
        or result.get("itineraries")
        or {}
    )
    if "A" in itineraries:
        return "A"
    return next(iter(itineraries), "")


def _selected_itinerary_payload(result: dict) -> tuple[str, list, dict]:
    option = _selected_option_key(result)
    itineraries = (
        result.get("final_itineraries")
        or result.get("validated_itineraries")
        or result.get("itineraries")
        or {}
    )
    option_meta = (result.get("option_meta") or {}).get(option, {})
    return option, itineraries.get(option, []) or [], option_meta


def _compact_explain_result(result: dict) -> dict:
    option, itinerary, option_meta = _selected_itinerary_payload(result)
    return {
        "selected_option": option,
        "itinerary": itinerary,
        "option_meta": option_meta,
        "summary": result.get("summary", {}) or {},
        "item_explanations": result.get("item_explanations", {}) or {},
        "evidence": result.get("evidence", {}) or {},
        "agent_steps": result.get("agent_steps", []) or [],
    }


def _name_key(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.replace("\u2019", "'").replace("\u2018", "'").replace("`", "'")
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"_+", " ", text)
    return re.sub(r"\s+", " ", text.strip()).casefold()


def _collect_plan_item_names(itineraries: dict) -> tuple[set[str], set[str], set[str]]:
    activity_names: set[str] = set()
    restaurant_names: set[str] = set()
    hotel_names: set[str] = set()

    for days in (itineraries or {}).values():
        if not isinstance(days, list):
            continue
        for day in days:
            for item in day.get("items", []) or []:
                icon = str(item.get("icon") or "")
                name = str(item.get("name") or "").strip()
                if not name:
                    continue
                if icon == "activity":
                    activity_names.add(_name_key(name))
                elif icon == "restaurant":
                    restaurant_names.add(_name_key(name))
                elif icon == "hotel":
                    if name.lower().startswith("checkout from "):
                        name = name[len("Checkout from "):].strip()
                    hotel_names.add(_name_key(name))

    return activity_names, restaurant_names, hotel_names


def _filter_named_rows(rows: list, allowed_names: set[str]) -> list:
    filtered: list = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        if _name_key(row.get("name", "")) in allowed_names:
            filtered.append(row)
    return filtered


def _compact_research_for_plan(result: dict, activity_names: set[str], restaurant_names: set[str]) -> dict:
    research = result.get("research", {}) or {}
    if not isinstance(research, dict):
        return {}
    compact: dict = {}
    for key in ("maps_attractions", "ta_attractions"):
        if key in research:
            compact[key] = _filter_named_rows(research.get(key, []), activity_names)
    for key in ("maps_restaurants", "ta_restaurants"):
        if key in research:
            compact[key] = _filter_named_rows(research.get(key, []), restaurant_names)
    return compact


def _compact_inventory_for_plan(result: dict, activity_names: set[str], restaurant_names: set[str]) -> dict:
    inventory = result.get("inventory", {}) or {}
    if not isinstance(inventory, dict):
        return {}
    compact: dict = {}
    if "attractions" in inventory:
        compact["attractions"] = _filter_named_rows(inventory.get("attractions", []), activity_names)
    if "restaurants" in inventory:
        compact["restaurants"] = _filter_named_rows(inventory.get("restaurants", []), restaurant_names)
    return compact


def _compact_plan_result(result: dict, *, include_explain: bool = False) -> dict:
    itineraries = result.get("itineraries", {}) or {}
    activity_names, restaurant_names, hotel_names = _collect_plan_item_names(itineraries)

    compact = {
        "itineraries": itineraries,
        "option_meta": result.get("option_meta", {}) or {},
        "validation_report": result.get("validation_report", {}) or {},
        "planner_decision_trace": result.get("planner_decision_trace", {}) or {},
        "planner_chain_of_thought": result.get("planner_chain_of_thought", ""),
        "research": _compact_research_for_plan(result, activity_names, restaurant_names),
        "inventory": _compact_inventory_for_plan(result, activity_names, restaurant_names),
        "state": result.get("state", {}) or {},
        "tool_log": result.get("tool_log", []) or [],
        "hotel_options": _filter_named_rows(result.get("hotel_options", []) or [], hotel_names),
    }

    raw_planner_output = result.get("raw_planner_output", {}) or {}
    planner_mode = str(raw_planner_output.get("planner_mode") or "").strip()
    if planner_mode:
        compact["planner_mode"] = planner_mode

    if include_explain:
        compact.update(
            {
                "selected_option": result.get("selected_option", ""),
                "explain_option": result.get("explain_option", ""),
                "summary": result.get("summary", {}) or {},
                "item_explanations": result.get("item_explanations", {}) or {},
                "evidence": result.get("evidence", {}) or {},
                "agent_steps": result.get("agent_steps", []) or [],
                "explainability_chain_of_thought": result.get("explainability_chain_of_thought", ""),
                "combined_chain_of_thought": result.get("combined_chain_of_thought", ""),
            }
        )

    return compact


def _get_tools(agent_name: str) -> dict:
    from agents.agent_tools import get_tools_for_agent

    return get_tools_for_agent(agent_name)


def _run_quiet(fn, *args, quiet: bool = True, **kwargs):
    if not quiet:
        return fn(*args, **kwargs)
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
        return fn(*args, **kwargs)


def run_research(test_state: dict, *, quiet: bool = True) -> dict:
    from agents.specialists.research_agent_1 import research_agent_1

    t0 = time.time()
    result = _run_quiet(
        research_agent_1,
        test_state,
        tools=_get_tools("research_agent_1"),
        quiet=quiet,
    )
    result["_elapsed_s"] = time.time() - t0
    return result


def run_pipeline(test_state: dict, *, quiet: bool = True) -> dict:
    from agents.specialists.planner_agent_1 import planner_from_research_1

    t0 = time.time()
    research_result = run_research(test_state, quiet=quiet)
    if "error" in research_result:
        return research_result
    result = _run_quiet(
        planner_from_research_1,
        test_state,
        research_result,
        quiet=quiet,
    )
    if isinstance(result, dict) and "inventory" not in result:
        result["inventory"] = research_result.get("inventory", {})
    if isinstance(result, dict) and "research" not in result:
        result["research"] = research_result.get("research", {})
    result["_elapsed_s"] = time.time() - t0
    return result


def run_plan_from_research(research_result: dict, *, quiet: bool = True) -> dict:
    from agents.specialists.planner_agent_1 import planner_from_research_1

    t0 = time.time()
    test_state = research_result.get("state", {})
    result = _run_quiet(
        planner_from_research_1,
        test_state,
        research_result,
        quiet=quiet,
    )
    if isinstance(result, dict) and "inventory" not in result:
        result["inventory"] = research_result.get("inventory", {})
    if isinstance(result, dict) and "research" not in result:
        result["research"] = research_result.get("research", {})
    result["_elapsed_s"] = time.time() - t0
    return result


def run_full(test_state: dict, *, quiet: bool = True) -> dict:
    from agents.specialists.explainability_agent import explainability_agent

    t0 = time.time()
    plan_result = run_pipeline(test_state, quiet=quiet)
    if "error" in plan_result:
        return plan_result
    explain_result = _run_quiet(
        explainability_agent,
        {**test_state, **plan_result},
        quiet=quiet,
    )
    return {**plan_result, **explain_result, "_elapsed_s": time.time() - t0}


def run_explain_from_plan(plan_result: dict, *, option: str | None = None, quiet: bool = True) -> dict:
    from agents.specialists.explainability_agent import explainability_agent

    t0 = time.time()
    payload = dict(plan_result)
    if option:
        payload["selected_option"] = str(option).strip().upper()
    explain_result = _run_quiet(
        explainability_agent,
        payload,
        quiet=quiet,
    )
    merged = {**plan_result, **explain_result}
    merged["_elapsed_s"] = time.time() - t0
    return merged


def print_report(result: dict, mode: str, input_path: Path, test_state: dict) -> None:
    elapsed = result.get("_elapsed_s", 0)

    _print_section("INPUT")
    print(f"  Source file: {input_path}")
    print(f"  Mode: {mode}")
    print(f"  Destination: {_terminal_text(test_state.get('destination', ''))}")
    print(f"  Dates: {_terminal_text(test_state.get('dates', ''))}")
    print(f"  Search queries: {len(test_state.get('search_queries', []))}")
    print(f"  Google Places raw debug dir: {_terminal_text(os.getenv('GOOGLE_PLACES_DEBUG_DIR', str(DEFAULT_PLACES_DEBUG_DIR)))}")

    if mode == "explain-from-plan":
        option, itinerary, option_meta = _selected_itinerary_payload(result)
        summary = result.get("summary", {}) or {}
        item_explanations = (result.get("item_explanations", {}) or {}).get("by_occurrence", {}) or {}

        _print_section("SELECTED ITINERARY")
        print(f"  Option: {option}")
        if option_meta:
            print(f"  Label: {_terminal_text(option_meta.get('label', ''))}")
            print(f"  Style: {_terminal_text(option_meta.get('style', ''))}")
            print(f"  Budget: {_terminal_text(option_meta.get('budget', ''))}")
        for line in _summarise_itineraries({option: itinerary}):
            print(line)

        if summary.get("overall_summary") or summary.get("day_summaries"):
            _print_section("SUMMARY")
            if summary.get("overall_summary"):
                print(f"  Overall: {_terminal_text(summary.get('overall_summary', ''))}")
            for day_label, text in (summary.get("day_summaries", {}) or {}).items():
                if text:
                    print(f"  {day_label}: {_terminal_text(text)}")

        if item_explanations:
            _print_section("ITEM EXPLANATIONS")
            print(f"  Option: {_terminal_text(option)}")
            print(f"  Explained items: {len(item_explanations)}")
            for occurrence_id, payload in list(item_explanations.items())[:5]:
                print(f"  {occurrence_id}")
                print(f"    day/time: {_terminal_text(payload.get('day', ''))} {_terminal_text(payload.get('time', ''))}")
                print(f"    name: {_terminal_text(payload.get('name', ''))}")
                reasons = payload.get("why", []) or []
                if reasons:
                    print(f"    why: {_terminal_text(reasons[0])}")
                rating = payload.get("rating", "")
                if rating:
                    print(f"    rating: {_terminal_text(rating)}")

        _print_section("RUNTIME")
        tool_log = result.get("tool_log", [])
        print(f"  Tool calls: {len(tool_log)}")
        print(f"  Total time: {elapsed:.1f}s")
        return

    if "inventory" in result:
        inventory = result.get("inventory", {})
        _print_section("RESEARCH SUMMARY")
        print(f"  Attractions: {len(inventory.get('attractions', []))}")
        print(f"  Restaurants: {len(inventory.get('restaurants', []))}")
        print(f"  Hotels: {len(inventory.get('hotels', []))}")
        print(f"  Outbound flights: {len(inventory.get('flights_outbound', []))}")
        print(f"  Return flights: {len(inventory.get('flights_return', []))}")
        if inventory.get("att_list_text"):
            print("\n  Attraction inventory sample:")
            for line in inventory["att_list_text"].splitlines()[:5]:
                print(f"    {_terminal_text(line)}")

    itineraries = result.get("itineraries", {})
    if itineraries:
        _print_section("ITINERARIES")
        for line in _summarise_itineraries(itineraries):
            print(line)

    if result.get("flight_options_outbound") or result.get("flight_options_return"):
        _print_section("FLIGHTS")
        for flight in result.get("flight_options_outbound", [])[:3]:
            print(
                f"  OUT  {_terminal_text(flight.get('airline', ''))} {_terminal_text(flight.get('flight_number', ''))} "
                f"{_terminal_text(flight.get('departure_time', ''))}->{_terminal_text(flight.get('arrival_time', ''))}  "
                f"SGD {_safe_sgd_from_usd(flight.get('price_usd'))}"
            )
        for flight in result.get("flight_options_return", [])[:3]:
            print(
                f"  RET  {_terminal_text(flight.get('airline', ''))} {_terminal_text(flight.get('flight_number', ''))} "
                f"{_terminal_text(flight.get('departure_time', ''))}->{_terminal_text(flight.get('arrival_time', ''))}  "
                f"SGD {_safe_sgd_from_usd(flight.get('price_usd'))}"
            )

    if result.get("hotel_options"):
        _print_section("HOTELS")
        for hotel in result.get("hotel_options", [])[:3]:
            print(
                f"  {_terminal_text(hotel.get('name', ''))}  "
                f"SGD {_safe_sgd_from_usd(hotel.get('price_per_night_usd'))}/night  "
                f"rating {_terminal_text(hotel.get('rating', '?'))}"
            )

    summary = result.get("summary", {}) or {}
    if summary.get("overall_summary") or summary.get("day_summaries"):
        _print_section("SUMMARY")
        if summary.get("overall_summary"):
            print(f"  Overall: {_terminal_text(summary.get('overall_summary', ''))}")
        for day_label, text in (summary.get("day_summaries", {}) or {}).items():
            if text:
                print(f"  {day_label}: {_terminal_text(text)}")

    item_explanations = (result.get("item_explanations", {}) or {}).get("by_occurrence", {}) or {}
    if item_explanations:
        _print_section("ITEM EXPLANATIONS")
        print(f"  Option: {_terminal_text(result.get('selected_option') or result.get('explain_option') or '?')}")
        print(f"  Explained items: {len(item_explanations)}")
        for occurrence_id, payload in list(item_explanations.items())[:5]:
            print(f"  {occurrence_id}")
            print(f"    day/time: {_terminal_text(payload.get('day', ''))} {_terminal_text(payload.get('time', ''))}")
            print(f"    name: {_terminal_text(payload.get('name', ''))}")
            reasons = payload.get("why", []) or []
            if reasons:
                print(f"    why: {_terminal_text(reasons[0])}")
            rating = payload.get("rating", "")
            if rating:
                print(f"    rating: {_terminal_text(rating)}")

    _print_section("RUNTIME")
    tool_log = result.get("tool_log", [])
    print(f"  Tool calls: {len(tool_log)}")
    print(f"  Total time: {elapsed:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["research", "pipeline", "plan-from-research", "explain-from-plan", "full", "generate", "local"],
        default="pipeline",
        help=(
            "research=research_agent_1 only | "
            "pipeline=research_agent_1 -> planner_from_research_1 | "
            "plan-from-research=planner_from_research_1 on a saved research JSON | "
            "explain-from-plan=explainability_agent on a saved planner JSON | "
            "full=pipeline + explainability_agent"
        ),
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_PATH),
        help="Path to request JSON file",
    )
    parser.add_argument(
        "--option",
        choices=["A", "B", "C", "a", "b", "c"],
        help="Which itinerary option to explain when using explain-from-plan",
    )
    parser.add_argument("--save", action="store_true", help="Save full result JSON to file")
    parser.add_argument("--verbose", action="store_true", help="Print report to terminal")
    args = parser.parse_args()

    mode = _normalise_mode(args.mode)
    input_path = Path(args.input)
    if mode == "plan-from-research":
        research_result = _load_research_result(input_path)
        test_state = research_result.get("state", {})
    elif mode == "explain-from-plan":
        plan_result = _load_plan_result(input_path)
        research_result = None
        test_state = plan_result.get("state", {})
    else:
        research_result = None
        plan_result = None
        test_state = _load_test_state(input_path)

    try:
        if mode == "research":
            result = run_research(test_state, quiet=not args.verbose)
        elif mode == "plan-from-research":
            result = run_plan_from_research(research_result, quiet=not args.verbose)
        elif mode == "explain-from-plan":
            result = run_explain_from_plan(
                plan_result,
                option=(args.option.upper() if args.option else None),
                quiet=not args.verbose,
            )
        elif mode == "full":
            result = run_full(test_state, quiet=not args.verbose)
        else:
            result = run_pipeline(test_state, quiet=not args.verbose)
    except Exception as e:
        print(f"\nFAILED: {e}")
        raise

    if args.verbose:
        print_report(result, mode, input_path, test_state)

    if args.save:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "test_places_explain_result" if mode == "explain-from-plan" else "test_places_result"
        out_path = project_root / f"{prefix}_{ts}.json"
        if mode == "explain-from-plan":
            save_result = _compact_explain_result(result)
        elif mode == "full":
            save_result = _compact_plan_result(result, include_explain=True)
        elif mode in {"pipeline", "plan-from-research"}:
            save_result = _compact_plan_result(result, include_explain=False)
        else:
            save_result = {k: v for k, v in result.items() if not k.startswith("_")}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(save_result, f, ensure_ascii=False, indent=2, default=str)
        print(str(out_path))
    elif args.verbose:
        _print_section("DONE")


if __name__ == "__main__":
    main()
