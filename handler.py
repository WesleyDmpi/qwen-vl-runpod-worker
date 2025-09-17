import runpod
import torch
from PIL import Image
import requests
from io import BytesIO
from transformers import AutoModelForCausalLM, AutoProcessor

# --- Variabelen ---
MODEL_ID = "HuggingFaceM4/SmolVLM-Llama-2.7b-Instruct"
model = None
processor = None

# --- Initialisatie Functie ---
def init():
    """
    Deze functie wordt één keer uitgevoerd wanneer de worker start.
    Hier laden we het model in het geheugen van de GPU.
    """
    global model, processor
    
    # Gebruik bfloat16 voor minder geheugengebruik en snellere inferentie
    torch_dtype = torch.bfloat16

    try:
        print(f"Model {MODEL_ID} aan het laden...")
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            load_in_4bit=True # Gebruik 4-bit kwantisatie voor minder VRAM-gebruik
        )
        print("Model succesvol geladen.")
    except Exception as e:
        print(f"Fout bij het laden van het model: {e}")
        # Zorg ervoor dat de worker stopt als het model niet kan laden
        return None
    
    return {"model": model, "processor": processor}

# --- Handler Functie ---
def handler(job):
    """
    Deze functie wordt voor elke API-request uitgevoerd.
    Het 'job' object bevat de input van de gebruiker.
    """
    global model, processor

    if model is None or processor is None:
        return {"error": "Model is niet geïnitialiseerd."}

    job_input = job.get('input', {})
    
    # Input valideren
    image_url = job_input.get('image_url')
    prompt_text = job_input.get('prompt', "Describe this image in detail.")

    if not image_url:
        return {"error": "Parameter 'image_url' is verplicht."}

    try:
        # Download en open de afbeelding van de URL
        response = requests.get(image_url)
        response.raise_for_status() # Geeft een error bij slechte statuscodes
        image = Image.open(BytesIO(response.content)).convert('RGB')
    except Exception as e:
        return {"error": f"Kon afbeelding niet downloaden of openen: {e}"}

    # Formatteer de prompt zoals het model het verwacht
    # Dit is CRUCIAAL voor goede resultaten!
    prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"

    try:
        # Verwerk de input
        inputs = processor(prompt, image, return_tensors="pt").to("cuda", dtype=torch.bfloat16)

        # Genereer de output
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False, # Zet op True voor meer creatieve antwoorden
            num_beams=1,
        )
        
        # Decodeer de gegenereerde tekst
        generated_text = processor.batch_decode(output, skip_special_tokens=True)[0]
        
        # Maak de output schoon (verwijder de input prompt)
        clean_output = generated_text.split('<|im_start|>assistant\n')[-1].strip()

        return {"description": clean_output}

    except Exception as e:
        return {"error": f"Fout tijdens de inferentie: {e}"}

# Start de RunPod serverless worker
runpod.serverless.start({"handler": handler, "init": init})
