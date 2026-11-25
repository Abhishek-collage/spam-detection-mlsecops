FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY config/ ./config/
COPY artifacts/ ./artifacts/
COPY audit/ ./audit/
COPY pipeline/ ./pipeline/
COPY src/ ./src/
COPY templates/ ./templates/
COPY application.py .

RUN mkdir -p /app/logs

EXPOSE 8000

CMD ["uvicorn", "application:app", "--host", "0.0.0.0", "--port", "8000"]
