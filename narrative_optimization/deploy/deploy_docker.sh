#!/bin/bash

# Docker Deployment Script

echo "=================================="
echo "DOCKER DEPLOYMENT"
echo "=================================="

# Build image
echo -e "\nBuilding Docker image..."
docker build -t narrative-optimization:latest .

# Test image
echo -e "\nTesting image..."
docker run --rm narrative-optimization:latest python scripts/health_check.py

# Run with docker-compose
echo -e "\nStarting services with docker-compose..."
docker-compose up -d

# Wait for services
echo -e "\nWaiting for services to start..."
sleep 5

# Check health
echo -e "\nChecking API health..."
curl -f http://localhost:5000/api/health || echo "API not ready yet"

echo -e "\n=================================="
echo "DEPLOYMENT COMPLETE"
echo "=================================="
echo -e "\nServices running:"
echo "  - API: http://localhost:5000"
echo "  - Continuous learner: background"
echo -e "\nView logs: docker-compose logs -f"
echo "Stop services: docker-compose down"

