import os, io, json, re, requests, logging, sys
from PIL import Image

# Unbuffered logs
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("smolvlm")

try:
    import torch
    from transformers import AutoProcessor, AutoModelForCausalLM, __version__ as trf_version
except Exception as e:
    print("FATAL: imports failed:", e)
    raise

try:
    import runpod
except Exception:
    runpod = None

# Default to SmolVLM
MODEL_NAME = os.getenv("MODEL_NAME", "HuggingFaceTB/SmolVLM-Instruct")
HF_TOKEN   = os.getenv("HF_TOKEN", None)

processor = None
model = None

def _lazy_init():
    global processor, model
    if model is not None and processor is not None:
        return

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    log.info(f"Transformers version: {trf_version}")
    log.info(f"Torch version: {torch.__version__}")
    log.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    log.info(f"Loading model: {MODEL_NAME} (dtype={dtype})")

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME, trust_remote_code=True, use_auth_token=HF_TOKEN
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        use_auth_token=HF_TOKEN
    ).eval()
    log.info("Model loaded.")

def _load_image(inp):
    if "image_url" in inp and inp["image_url"]:
        r = requests.get(inp["image_url"], timeout=30)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    if "image_base64" in inp and inp["image_base64"]:
        b64 = inp["image_base64"]
        if b64.startswith("data:"):
            b64 = b64.split(",", 1)[-1]
        import base64
        raw = base64.b64decode(b64)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    raise ValueError("Provide image_url or image_base64")

def _extract_json(text):
    m = re.search(r'\{.*\}', text, flags=re.S)
    if not m:
        raise ValueError("No JSON object found in model output")
    return json.loads(m.group(0))

def predict(inp):
    _lazy_init()
    prompt = inp.get("prompt") or "Return STRICT JSON with keys: caption, detected_objects, colors, nsfw. No prose."
    max_new = int(inp.get("max_new_tokens", 160))

    img = _load_image(inp)

    # Try chat template first
    messages = [
        {"role": "system", "content": "Return STRICT JSON with keys: caption, detected_objects, colors, nsfw. No prose, no markdown."},
        {"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": prompt}
        ]}
    ]

    try:
        chat = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=[chat], images=[img], return_tensors="pt").to(model.device)
    except Exception as e:
        # Fallback: prepend an <image> token if template not available
        log.warning("apply_chat_template failed (%s); using <image> fallback.", e)
        text = "<image>\n" + prompt
        inputs = processor(text=[text], images=[img], return_tensors="pt").to(model.device)

    with torch.no_grad():
        out_ids = model.generate(
            **inputs, do_sample=False, temperature=0.0, max_new_tokens=max_new
        )

    out_text = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
    log.info("Raw model output (truncated): %s", out_text[:200].replace("\\n"," "))

    try:
        data = _extract_json(out_text)
    except Exception as e:
        log.warning("JSON parse failed: %s", e)
        data = {
            "caption": out_text.strip()[:300],
            "detected_objects": [],
            "colors": [],
            "nsfw": False
        }

    data.setdefault("caption","")
    data.setdefault("detected_objects",[])
    data.setdefault("colors",[])
    data.setdefault("nsfw",False)
    return {"output": data}

def handler(event):
    try:
        return predict(event.get("input", {}))
    except Exception as e:
        log.exception("Handler error")
        return {"error": str(e)}

if runpod is not None:
    runpod.serverless.start({"handler": handler})
