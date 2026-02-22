FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first (for Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY configs/ configs/
COPY mlops_assignment/ mlops_assignment/
COPY src/ src/
COPY models/ models/
COPY data/raw/ data/raw/
COPY .streamlit/ .streamlit/

# Expose Streamlit port (Railway assigns PORT dynamically)
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:${PORT:-8501}/_stcore/health || exit 1

# Run the Streamlit app — uses $PORT from Railway, defaults to 8501 locally
CMD streamlit run src/webapp/app.py \
    --server.port=${PORT:-8501} \
    --server.address=0.0.0.0 \
    --server.headless=true
