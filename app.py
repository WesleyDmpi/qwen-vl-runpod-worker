import runpod
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
import base64
import io

# --- Globale Variabelen ---
# Laad het model en de processor één keer wanneer de worker start (cold boot)
# Dit voorkomt dat het model bij elke request opnieuw geladen wordt.
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Starting model initialization...")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.bfloat16, # Gebruik bfloat16 voor betere performance op geschikte GPUs
    trust_remote_code=True
).to(device)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True)
print("Model initialization complete.")


def handler(job):
    """
    Deze functie wordt uitgevoerd voor elke API call.
    De 'job' input is een dictionary met de data die je verstuurt.
    """
    job_input = job['input']

    # Valideer de input
    if 'image_base64' not in job_input or 'prompt' not in job_input:
        return {"error": "Zorg ervoor dat 'image_base64' en 'prompt' aanwezig zijn in de input."}

    try:
        # 1. Decodeer de Base64 afbeelding
        image_data = base64.b64decode(job_input['image_base64'])
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # 2. Haal de prompt op
        prompt = job_input['prompt']
        
        # Bereid de input voor het model voor
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]}
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = processor([text], images=[image], return_tensors="pt").to(device)

        # 3. Genereer de output
        generated_ids = model.generate(
            model_inputs.input_ids,
            pixel_values=model_inputs.pixel_values,
            max_new_tokens=1024,
            do_sample=False,
            num_beams=1,
            # Hier kun je meer generate-parameters toevoegen indien nodig
        )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        # 4. Decodeer de output naar tekst
        response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        # 5. Geef het resultaat terug in een JSON-formaat
        return {"result": response}

    except Exception as e:
        # Geef een duidelijke foutmelding terug als er iets misgaat
        return {"error": f"Er is een fout opgetreden: {str(e)}"}


# Start de Runpod handler
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})