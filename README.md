# NFT Image Metadata Extraction Service

A production-grade, fully offline, self-hosted NFT image metadata extraction service optimized for CPU-only environments.

## Features
- **Image Captioning**: Powered by Salesforce BLIP base.
- **Feature Extraction**: Powered by OpenCLIP ViT-B-32.
- **Structured Metadata**: Powered by Phi-3 Mini Instruct (GGUF via llama-cpp-python).
- **FastAPI**: High-performance REST API.
- **Fully Offline**: Models are downloaded at build time; no internet required at runtime.
- **CPU Optimized**: Designed to run efficiently on 8-core CPU environments.

## API Endpoints

### POST `/analyze`
Analyze an image and generate structured NFT metadata.
- **Request**: `multipart/form-data`
  - `image`: The image file to analyze.
- **Response**: `application/json`
```json
{
  "title": "Cyber Horse",
  "description": "A futuristic cyberpunk horse in neon city",
  "style": "Cyberpunk",
  "color": "Neon blue",
  "mood": "Futuristic",
  "tags": "cyberpunk, horse, neon"
}
```

### GET `/health`
Health check endpoint.
- **Response**: `{"status": "ok"}`

## Requirements
- Docker
- Docker Compose

## Setup and Running
Build and start the service using Docker Compose:
```bash
docker compose up --build
```
The first build will take some time as it downloads several GBs of model files.

## Architecture
- `/app/main.py`: Entry point for the FastAPI application.
- `/app/services/`: Core logic for captioning, feature extraction, and LLM metadata generation.
- `/app/models/`: Local storage for model weights (populated during Docker build).
- `Dockerfile`: Multi-stage build for a clean and optimized image.

## Performance
- Target latency: 2–5 seconds per image on an 8-core CPU.
- Memory usage: Under 8GB RAM.
