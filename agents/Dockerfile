FROM python:3.12-slim

WORKDIR /app/agents

ENV PYTHONPATH=/app

RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY agents/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code needed by agents service
COPY agents /app/agents
COPY tools /app/tools

EXPOSE 8001 8100 8101 8102 8103 8104 8105 8106 8107

HEALTHCHECK --interval=30s --timeout=15s --start-period=120s --retries=5 \
  CMD curl -f http://localhost:8001/health || exit 1

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 8001 & \
uvicorn main:app --host 0.0.0.0 --port 8100 & \
uvicorn main:app --host 0.0.0.0 --port 8101 & \
uvicorn main:app --host 0.0.0.0 --port 8102 & \
uvicorn main:app --host 0.0.0.0 --port 8103 & \
uvicorn main:app --host 0.0.0.0 --port 8104 & \
uvicorn main:app --host 0.0.0.0 --port 8105 & \
uvicorn main:app --host 0.0.0.0 --port 8106 & \
uvicorn main:app --host 0.0.0.0 --port 8107 & \
wait"]
