# 🐳 Docker Setup for FlyPrice

## Quick Start

### Prerequisites
- Docker installed on your system
- Docker Compose installed

### 1. Build and Run with Docker Compose

```bash
# Build and start all services
docker-compose up --build

# Run in background
docker-compose up -d --build
```

### 2. Access Your Services

- **Main API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **MLflow UI**: http://localhost:5000
- **Health Check**: http://localhost:8000/health

### 3. Stop Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

## Services Included

### 🚀 flyprice-api
- **Port**: 8000
- **Description**: Main FastAPI application with all endpoints
- **Features**: Flight prediction, chatbot, recommendations, authentication

### 📊 mlflow-ui (Optional)
- **Port**: 5000  
- **Description**: MLflow tracking UI for model experiments
- **Features**: Model metrics, artifacts, experiment tracking

## Environment Variables

Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

## Volumes

The following directories are mounted as volumes:
- `mlflow_tracking/` - MLflow experiment data
- `rag_knowledge/` - Chatbot knowledge base
- `models/` - Trained ML models
- `data/` - Dataset files

## Development

### Rebuild after changes

```bash
# Rebuild specific service
docker-compose up --build flyprice-api

# View logs
docker-compose logs -f flyprice-api

# Execute commands in container
docker-compose exec flyprice-api bash
```

### Production Considerations

1. **Security**: Remove MLflow UI in production
2. **Performance**: Add resource limits in docker-compose.yml
3. **Scaling**: Use Docker Swarm or Kubernetes for scaling
4. **Monitoring**: Add health checks and monitoring

## Troubleshooting

### Common Issues

1. **Port conflicts**: Change ports in docker-compose.yml
2. **Permission issues**: Ensure Docker has proper permissions
3. **Build failures**: Check requirements.txt and Dockerfile

### Logs

```bash
# View all logs
docker-compose logs

# View specific service logs
docker-compose logs flyprice-api
```

## Docker Commands Reference

```bash
# Build image only
docker build -t flyprice .

# Run container manually
docker run -p 8000:8000 -v $(pwd)/mlflow_tracking:/app/mlflow_tracking flyprice

# Clean up unused images
docker image prune -f
```
