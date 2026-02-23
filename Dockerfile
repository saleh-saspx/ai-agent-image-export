# Stage 1: Build and Download
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install torch for CPU first to save space
RUN pip install --no-cache-dir torch==2.1.1 --index-url https://download.pytorch.org/whl/cpu

# Install other python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --force-reinstall numpy==1.26.4
RUN pip install --no-cache-dir -r requirements.txt

# Create model directories
RUN mkdir -p app/models/blip app/models/clip app/models/llm app/models/huggingface app/models/torch

# Set cache directories for download
ENV HF_HOME=/build/app/models/huggingface
ENV TORCH_HOME=/build/app/models/torch

# Copy download script and download models
COPY download_models.py .
RUN python download_models.py

# Stage 2: Final Image
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy models from builder
COPY --from=builder /build/app/models /app/app/models

# Copy application code
COPY app/ /app/app/

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/app/models/huggingface
ENV TORCH_HOME=/app/app/models/torch

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
