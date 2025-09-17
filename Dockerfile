# Pinned PyTorch/CUDA image (cu118) – goed voor bitsandbytes 0.43.x
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# Handige tools voor snelle model pulls
RUN apt-get update && \
    apt-get install -y --no-install-recommends git git-lfs && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

COPY handler.py .

# Start de RunPod serverless worker
CMD ["python", "handler.py"]
