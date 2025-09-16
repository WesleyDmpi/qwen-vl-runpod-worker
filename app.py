import runpod
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
import base64
import io
import requests # Nieuwe import

# --- Model laden blijft hetzelfde ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Starting model initialization...")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).to(device)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True)
print("Model initialization complete.")


def handler(job):
    job_input = job['input']
    prompt = job_input.get('prompt', 'Beschrijf deze afbeelding.') # Veilige manier om prompt op te halen

    try:
        # --- NIEUWE LOGICA VOOR AFBEELDING ---
        if 'image_url' in job_input:
            # Optie 1: Afbeelding downloaden van URL
            url = job_input['image_url']
            response = requests.get(url, stream=True)
            response.raise_for_status() # Geeft een error als de download mislukt
            image = Image.open(response.raw).convert("RGB")
        elif 'image_base64' in job_input:
            # Optie 2: Afbeelding decoderen van Base64 (voor Make.com)
            image_data = base64.b64decode(job_input['image_base64'])
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
        else:
            return {"error": "Input moet ofwel 'image_url' of 'image_base64' bevatten."}
        # ------------------------------------

        # Bereid de input voor het model voor
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]}
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = processor([text], images=[image], return_tensors="pt").to(device)

        # Genereer de output
        generated_ids = model.generate(
            model_inputs.input_ids,
            pixel_values=model_inputs.pixel_values,
            max_new_tokens=job_input.get('max_new_tokens', 1024), # Maak max_new_tokens optioneel
            do_sample=False,
            num_beams=1,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        return {"result": response_text}

    except Exception as e:
        return {"error": f"Er is een fout opgetreden: {str(e)}"}

# Start de Runpod handler
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
