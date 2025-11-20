FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY narrative_optimization/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for Flask
RUN pip install --no-cache-dir flask flask-cors gunicorn networkx psutil

# Copy application
COPY . .

# Expose port
EXPOSE 5738

# Run with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5738", "--workers", "4", "--timeout", "120", "app:app"]

