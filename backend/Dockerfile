# FROM python:3.12-slim
FROM registry.cn-hangzhou.aliyuncs.com/docker-library/python:3.12-slim

WORKDIR /app/backend

ENV PYTHONPATH=/app

RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*


# Copy backend requirements
COPY backend/requirements.txt .

# Install backend dependencies only
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000"]
