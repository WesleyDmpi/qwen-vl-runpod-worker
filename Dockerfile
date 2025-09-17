FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# Installeer git
RUN apt-get update && apt-get install -y git

# Kopieer de requirements
COPY requirements.txt .

# Installeer de packages via requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Kopieer de applicatiecode
COPY app.py .

# Start de applicatie
CMD ["python3", "-u", "app.py"]
