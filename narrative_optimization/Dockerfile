# Narrative Optimization Framework - Docker Container

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create cache directory
RUN mkdir -p /root/.narrative_optimization/cache

# Set environment variables
ENV PYTHONPATH=/app
ENV NARRATIVE_CACHE_DIR=/root/.narrative_optimization/cache

# Expose API port
EXPOSE 5000

# Run initialization
RUN python INITIALIZE_SYSTEM.py || true

# Default command: run API server
CMD ["python", "api/api_server.py", "--host", "0.0.0.0", "--port", "5000"]

