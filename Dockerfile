FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# Installeer git, wat altijd een goed idee is
RUN apt-get update && apt-get install -y git

# Forceer de installatie van alle benodigde packages in één keer.
# - We upgraden pip eerst.
# - We schonen de pip cache op om conflicten te voorkomen.
# - We gebruiken --no-cache-dir om schijfproblemen te vermijden.
# - We forceren de upgrade van alle packages.
RUN pip install --upgrade pip && \
    pip cache purge && \
    pip install --no-cache-dir --upgrade \
    "runpod" \
    "torch" \
    "pillow" \
    "requests" \
    "sentencepiece" \
    "einops" \
    "accelerate" \
    "transformers>=4.41.0"

# Kopieer de applicatiecode
COPY app.py .

# Specificeer het commando om de applicatie te starten
CMD ["python3", "-u", "app.py"]
