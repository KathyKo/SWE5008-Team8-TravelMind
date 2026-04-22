"""
agents/server.py — Single-process multi-port startup

Runs all 9 agent ports inside one asyncio event loop so that
module-level singletons (LLM Guard DeBERTa model, etc.) are
loaded exactly once, not once per port.

Architecture intent preserved:
  - Each port still maps to one logical agent (port enforcement in main.py works)
  - Callers route to specific ports as if agents were separate processes
  - In reality: shared memory, one model load, ~8× less RAM than the 9-process design

Port map:
  8001  general / graph stream / health
  8100  input_guard
  8101  intent_profile
  8102  search (research)
  8103  planner
  8104  debate
  8105  explain
  8106  output_guard
  8107  replanner
"""

import asyncio
import logging
import signal

import uvicorn

from agents.main import app  # single import — all module-level models load here, once

log = logging.getLogger(__name__)

AGENT_PORTS = [8001, 8100, 8101, 8102, 8103, 8104, 8105, 8106, 8107]


async def main() -> None:
    servers: list[uvicorn.Server] = []

    for port in AGENT_PORTS:
        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=port,
            log_level="info",
            # Disable uvicorn's per-server lifespan so FastAPI startup/shutdown
            # events fire only once (from the first server below).
            lifespan="off" if port != AGENT_PORTS[0] else "on",
        )
        server = uvicorn.Server(config)
        # Disable per-instance signal handlers; we install one shared handler below.
        server.install_signal_handlers = lambda: None  # type: ignore[method-assign]
        servers.append(server)

    loop = asyncio.get_running_loop()

    def _shutdown(*_):
        log.info("Shutdown signal received — stopping all agent servers")
        for s in servers:
            s.should_exit = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _shutdown)

    log.info("Starting %d agent listeners on ports %s", len(AGENT_PORTS), AGENT_PORTS)
    await asyncio.gather(*[s.serve() for s in servers])


if __name__ == "__main__":
    asyncio.run(main())
