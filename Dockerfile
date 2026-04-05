FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source and data
COPY ddi_rag/ ./ddi_rag/
COPY streamlit_app.py .
COPY data/datasets/ ./data/datasets/

WORKDIR /app

# Pre-download embedding model so container starts faster
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Flask API on 5000, Streamlit on 8501
EXPOSE 5000 8501

# Default: run the Flask API. Override with:
#   docker run ... streamlit run streamlit_app.py --server.port 8501
CMD ["python", "ddi_rag/run.py"]
