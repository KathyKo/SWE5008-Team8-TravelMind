FROM python:3.12-slim

WORKDIR /app/backend

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements and agents requirements
COPY backend/requirements.txt .
COPY agents/requirements.txt ../agents_requirements.txt

# Install all dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r ../agents_requirements.txt

# Copy backend code
COPY backend/ .

# Copy agents and tools from project root (needed at runtime)
COPY agents /app/agents
COPY tools /app/tools

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000"]
