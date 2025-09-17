import runpod
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import base64
import io
import requests

# --- Model laden ---
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_id = "vikhyatk/moondream2"

print(f"Starting model initialization for {model_id}...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch_dtype,
    revision="2024-05-20" # Gebruik een vaste, stabiele versie
).to(device)

tokenizer = AutoTokenizer.from_pretrained(model_id, revision="2024-05-20")
print("Model initialization complete.")

def handler(job):
    job_input = job['input']
    prompt = job_input.get('prompt', 'Describe this image in detail.')

    try:
        # Afbeelding logica
        if 'image_url' in job_input:
            url = job_input['image_url']
            response = requests.get(url, stream=True)
            response.raise_for_status()
            image = Image.open(response.raw).convert("RGB")
        elif 'image_base_64' in job_input: # Let op: ik heb hier image_base64 gecorrigeerd
            image_data = base64.b64decode(job_input['image_base_64'])
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
        else:
            return {"error": "Input moet ofwel 'image_url' of 'image_base_64' bevatten."}

        # Moondream2's specifieke manier om de afbeelding en prompt te verwerken
        enc_image = model.encode_image(image)
        
        # Voer de inferentie uit
        answer = model.answer_question(
            enc_image,
            prompt,
            tokenizer
        )

        return {"result": answer}

    except Exception as e:
        import traceback
        return {"error": f"Er is een fout opgetreden: {str(e)}", "traceback": traceback.format_exc()}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
