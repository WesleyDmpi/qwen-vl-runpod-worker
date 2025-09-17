# DEZE REGEL IS VERANDERD naar een oudere, stabielere PyTorch/CUDA versie
FROM runpod/pytorch:2.1.2-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app.py .
CMD ["python3", "-u", "app.py"]
