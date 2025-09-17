import runpod
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
import base64
import io
import requests

# --- Model laden ---
model_id = "microsoft/Phi-3-vision-128k-instruct"

print(f"Starting model initialization for {model_id} with 4-bit quantization...")

# Laad het model in 4-bit om geheugen te besparen
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    load_in_4bit=True, # DEZE REGEL IS DE MAGIE
    device_map="auto"   # Met kwantisatie is device_map weer de beste methode
)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

print("Model initialization complete.")

def handler(job):
    job_input = job['input']
    prompt = job_input.get('prompt', 'Describe what is in this image.')

    try:
        # Afbeelding logica (blijft hetzelfde)
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
        # De inputs moeten naar het device waar het model is
        inputs = processor(prompt_text, [image], return_tensors="pt").to(model.device)

        generation_args = {
            "max_new_tokens": job_input.get('max_new_tokens', 1024),
            "temperature": 0.0,
            "do_sample": False,
        }
        
        with torch.no_grad():
            generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)

        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response_text = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        return {"result": response_text.strip()}

    except Exception as e:
        import traceback
        return {"error": f"Er is een fout opgetreden: {str(e)}", "traceback": traceback.format_exc()}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
