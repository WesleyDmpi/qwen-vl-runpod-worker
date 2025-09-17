import runpod
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
import base64
import io
import requests

# --- Aangepast Model laden ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "microsoft/Phi-3-vision-128k-instruct"

print(f"Starting model initialization for {model_id} on device {device}...")

# Laad het model eerst op de CPU
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype="auto"
).to(device).eval() # Stuur het model daarna in zijn geheel naar de GPU en zet in evaluatie-modus

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

print("Model initialization complete.")

def handler(job):
    job_input = job['input']
    prompt = job_input.get('prompt', 'Describe what is in this image.')

    try:
        if 'image_url' in job_input:
            url = job_input['image_url']
            response = requests.get(url, stream=True)
            response.raise_for_status()
            image = Image.open(response.raw).convert("RGB")
        elif 'image_base64' in job_input:
            image_data = base64.b64decode(job_input['image_base64'])
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
        else:
            return {"error": "Input moet ofwel 'image_url' of 'image_base64' bevatten."}

        messages = [
            {"role": "user", "content": f"<|image_1|>\n{prompt}"},
        ]

        prompt_text = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Stuur de inputs ook expliciet naar hetzelfde device als het model
        inputs = processor(prompt_text, [image], return_tensors="pt").to(device)

        generation_args = {
            "max_new_tokens": job_input.get('max_new_tokens', 1024),
            "temperature": 0.0,
            "do_sample": False,
        }

        # Voer de generatie uit binnen een no_grad blok voor efficiëntie
        with torch.no_grad():
            generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)

        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response_text = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        return {"result": response_text.strip()}

    except Exception as e:
        # Voeg de traceback toe voor betere debugging
        import traceback
        return {"error": f"Er is een fout opgetreden: {str(e)}", "traceback": traceback.format_exc()}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
