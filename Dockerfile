FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y git

COPY requirements.txt .
# --- TERUG NAAR DE NORMALE INSTALLATIE ---
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

CMD ["python3", "-u", "app.py"]
