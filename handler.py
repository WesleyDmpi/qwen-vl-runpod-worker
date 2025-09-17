import runpod
import torch
from PIL import Image
import requests
from io import BytesIO
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

MODEL_ID = "HuggingFaceM4/SmolVLM-Llama-2.7b-Instruct"
model = None
processor = None

torch.set_grad_enabled(False)

def init():
    global model, processor

    try:
        print(f"[init] Loading processor/model: {MODEL_ID}")
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            quantization_config=quant_config,
            device_map="cuda",
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
        )

        model.eval()
        print("[init] Model loaded.")
        return {"status": "ok"}
    except Exception as e:
        print(f"[init] Failed: {e}")
        return None

def _load_image_from_url(url: str) -> Image.Image:
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGB")

def handler(job):
    global model, processor
    if model is None or processor is None:
        return {"error": "Model not initialized."}

    inp = (job or {}).get("input", {}) or {}
    image_url = inp.get("image_url")
    prompt_text = inp.get("prompt", "Describe this image in detail.")
    max_new_tokens = int(inp.get("max_new_tokens", 192))

    if not image_url:
        return {"error": "Missing required parameter: 'image_url'."}

    try:
        image = _load_image_from_url(image_url)
    except Exception as e:
        return {"error": f"Failed to load image: {e}"}

    # Gebruik de officiële chat-template + multimodal schema
    # Let op: veel SmolVLM builds verwachten [{"type":"image"}, {"type":"text", ...}]
    messages = [
        {"role": "system", "content": "You are a helpful vision assistant."},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        },
    ]

    try:
        text = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        inputs = processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        )

        # Stuur tensors naar GPU en juiste dtype
        inputs = {k: (v.to("cuda", dtype=torch.bfloat16) if hasattr(v, "to") else v)
                  for k, v in inputs.items()}

        gen_out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

        # Enkel nieuwe tokens decoderen (na de prompt)
        input_len = inputs["input_ids"].shape[-1]
        new_tokens = gen_out[0, input_len:]
        text_out = processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        return {
            "description": text_out,
            "tokens_generated": int(new_tokens.shape[-1]),
        }

    except Exception as e:
        return {"error": f"Inference error: {e}"}

runpod.serverless.start({"handler": handler, "init": init})
