import runpod
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
import base64
import io
import requests

# --- Model laden ---
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_id = "HuggingFaceTB/SmolVLM-Instruct"

print(f"Starting model initialization for {model_id}...")
model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    trust_remote_code=True
).to(device).eval()

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
print("Model initialization complete.")

# Dit is de specifieke prompt-structuur die SmolVLM verwacht
prompt_template = "A chat between a curious user and an artificial intelligence assistant. The user provides an image and asks a question. The assistant follows the user's instructions carefully and answers the question. USER: <image>\n{prompt} ASSISTANT:"

def handler(job):
    job_input = job['input']
    prompt = job_input.get('prompt', 'Describe this image.')

    try:
        # Afbeelding logica
        if 'image_url' in job_input:
            url = job_input['image_url']
            response = requests.get(url, stream=True)
            response.raise_for_status()
            image = Image.open(response.raw)
        elif 'image_base_64' in job_input:
            image_data = base64.b64decode(job_input['image_base_64'])
            image = Image.open(io.BytesIO(image_data))
        else:
            return {"error": "Input moet ofwel 'image_url' of 'image_base_64' bevatten."}

        # Vul de prompt-template in
        final_prompt = prompt_template.format(prompt=prompt)

        # Verwerk de input
        inputs = processor(text=[final_prompt], images=[image], return_tensors="pt").to(device, torch_dtype)

        # Genereer de output
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=job_input.get('max_new_tokens', 256),
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.pad_token_id
            )
        
        # Decodeer en geef het resultaat terug
        result_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
        # Soms bevat de output de prompt nog, verwijder die voor een schone output
        if final_prompt in result_text:
             result_text = result_text.replace(final_prompt, "").strip()

        return {"result": result_text}

    except Exception as e:
        import traceback
        return {"error": f"Er is een fout opgetreden: {str(e)}", "traceback": traceback.format_exc()}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
