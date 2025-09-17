# ✅ Serverless base-image van RunPod (GPU + serverless runtime)
FROM runpod/serverless:gpu

# Snellere, duidelijke logs
ENV PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    TRANSFORMERS_CACHE=/root/.cache/huggingface

WORKDIR /app

# Up-to-date pip
RUN python -m pip install --upgrade pip

# Dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Code
COPY . /app

# Start je serverless handler (app.py roept runpod.serverless.start(...) aan)
CMD ["python", "-u", "app.py"]
