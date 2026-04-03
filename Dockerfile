FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY ddi_rag/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY ddi_rag/ .

# Pre-download embedding model so container starts faster
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

EXPOSE 5000

CMD ["python", "run.py"]
