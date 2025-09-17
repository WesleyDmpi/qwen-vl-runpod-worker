# Gebruik een basis-image van RunPod met PyTorch en CUDA
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Zet de werkdirectory
WORKDIR /app

# Kopieer de requirements en installeer de packages
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Kopieer de handler code
COPY handler.py .

# Start de RunPod Python worker wanneer de container start
CMD ["python", "-m", "runpod.serverless"]
