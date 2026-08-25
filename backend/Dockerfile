FROM python:3.11-slim
WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng \
        tesseract-ocr-chi-sim \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /app/data
COPY pyproject.toml uv.lock ./
COPY src ./src
RUN pip install --no-cache-dir uv && uv sync --frozen
ENV AUTOFLOW_JWT_SECRET=change-me-in-production
ENV AUTOFLOW_DATABASE_URL=sqlite:///./data/autoflow.db
EXPOSE 8000
CMD ["uv", "run", "uvicorn", "autoflow_scheduling.app:app", "--host", "0.0.0.0", "--port", "8000"]
