import os

# Unified default model for agents that still read OPENAI_MODEL
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-nano-2026-03-17")

# DMX provider config (requested for planner/debate/judge paths)
DMX_BASE_URL = os.getenv("DMX_BASE_URL", "https://www.dmxapi.cn/v1")
DMX_API_KEY = os.getenv("DMX_API_KEY")

# Planner/Agent3
PLANNER_MODEL = os.getenv("PLANNER_MODEL", "gpt-4.1")

# Debate Agent4 critique model
DEBATE_MODEL = os.getenv("DEBATE_MODEL", "gpt-4.1")

# Judge model (better than debate model, while cheaper than top-tier)
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-5-mini")
