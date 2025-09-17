import runpod, os, torch
from PIL import Image
import requests
from io import BytesIO
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

MODEL_ID = "HuggingFaceM4/SmolVLM-Llama-2.7b-Instruct"
model = None
processor = None

def _diag():
    print("[diag] torch.cuda.is_available:", torch.cuda.is_available())
    print("[diag] torch.cuda.device_count:", torch.cuda.device_count())
    print("[diag] torch.version.cuda:", torch.version.cuda)
    print("[diag] torch.__version__:", torch.__version__)

def init():
    global model, processor
    try:
        _diag()
        if not torch.cuda.is_available():
            # Duidelijke fout i.p.v. stille crash
            raise RuntimeError("No CUDA GPU available. Use a GPU-enabled Serverless pool.")

        print("[init] loading processor...")
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

        print("[init] loading model (4-bit)...")
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            device_map="cuda",
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            quantization_config=quant,
        )
        model.eval()
        print("[init] model loaded.")
        return {"ok": True}
    except Exception as e:
        # Laat init niet stil falen—toon reden in logs én in response
        print("[init][error]", repr(e))
        return {"error": str(e)}

def _load_image(url: str) -> Image.Image:
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGB")

def handler(job):
    global model, processor
    inp = (job or {}).get("input", {}) or {}
    if model is None or processor is None:
        return {"error": "Model not initialized (see init logs)."}

    image_url = inp.get("image_url")
    prompt_text = inp.get("prompt", "Describe this image in detail.")
    max_new_tokens = int(inp.get("max_new_tokens", 160))

    if not image_url:
        return {"error": "Missing 'image_url'."}

    try:
        image = _load_image(image_url)
    except Exception as e:
        return {"error": f"Failed to load image: {e}"}

    messages = [
        {"role": "system", "content": "You are a helpful vision assistant."},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]},
    ]

    try:
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=[text], images=[image], return_tensors="pt")
        inputs = {k: (v.to("cuda", dtype=torch.bfloat16) if hasattr(v, "to") else v) for k,v in inputs.items()}

        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

        input_len = inputs["input_ids"].shape[-1]
        new_tokens = gen[0, input_len:]
        out = processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return {"description": out, "tokens_generated": int(new_tokens.shape[-1])}
    except Exception as e:
        return {"error": f"Inference error: {e}"}

runpod.serverless.start({"handler": handler, "init": init})
