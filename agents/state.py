from typing import TypedDict, Optional, Annotated, List
import operator


class State(TypedDict):
    """
    Overall state of the Travel Agency multi-agent system.
    All specialist agents read/write subsets of this state.
    The orchestrator inspects key fields to decide routing.
    """

    # ── Conversation ──────────────────────────────────────
    messages: Annotated[list, operator.add]

    # ── User Input / Intent Profile ───────────────────────
    origin: Optional[str]
    destination: Optional[str]
    budget: Optional[str]
    dates: Optional[str]
    preferences: Optional[str]       # e.g. "budget", "moderate", "luxury"
    duration: Optional[str]          # e.g. "3 days"
    user_profile: Optional[str]      # derived tag string
    travelers: Optional[int]
    outbound_time_pref: Optional[str]
    return_time_pref: Optional[str]
    session_id: Optional[str]
    intent_profile_output: Optional[dict]
    user_profile_structured: Optional[dict]
    search_queries: Optional[dict]
    hard_constraints: Optional[dict]
    soft_preferences: Optional[dict]
    is_complete: Optional[bool]

    # ── Input Guard (Security) ────────────────────────────
    threat_blocked: Optional[bool]   # True → orchestrator terminates flow
    threat_type: Optional[str]
    threat_detail: Optional[str]
    sanitised_input: Optional[str]
    input_guard_decision: Optional[str]
    security_audit_log: Optional[List]

    # ── Research / Search ─────────────────────────────────
    search_results: Optional[dict]
    research: Optional[dict]
    inventory: Optional[dict]        # attractions / hotels / flights
    maps_attractions: Optional[str]

    # ── Planner ──────────────────────────────────────────
    itineraries: Optional[list]
    final_itineraries: Optional[list]
    validated_itineraries: Optional[list]
    planner_decision_trace: Optional[list]
    planner_chain_of_thought: Optional[str]
    chain_of_thought: Optional[str]

    # ── Debate ───────────────────────────────────────────
    # is_valid: None=尚未评审 / True=通过 / False=未通过
    is_valid: Optional[bool]
    debate_count: Optional[int]      # 累计 debate→planner 循环次数
    critique: Optional[str]          # debate 输出的改进意见，供 planner 重新规划
    composite_score: Optional[float] # debate 综合评分
    approval_threshold: Optional[float]  # 通过阈值 (default 75.0)
    debate_output: Optional[dict]

    # ── Replanner ────────────────────────────────────────
    # replanner 仅由用户反馈触发，不参与 debate 循环
    user_feedback: Optional[str]     # 用户修改/反馈内容 → 触发 replanner
    replan_attempts: Optional[int]
    max_replan_attempts: Optional[int]
    replanner_output: Optional[dict]

    # ── Explainability ───────────────────────────────────
    explanation: Optional[dict]
    explain_data: Optional[dict]
    summary: Optional[dict]

    # ── Output Guard (Security) ──────────────────────────
    output_flagged: Optional[bool]   # True → output was sanitised/replaced
    output_flag_reason: Optional[str]
    output_guard_decision: Optional[str]

    # ── Orchestrator Routing ─────────────────────────────
    next_node: Optional[str]         # orchestrator_node 写入，orchestrator_routing 读取
    error_message: Optional[str]
    final_output: Optional[str]
