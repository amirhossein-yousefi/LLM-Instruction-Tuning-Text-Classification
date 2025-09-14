# Custom inference handler for SageMaker Hugging Face DLC

import os
import json
from typing import Any, Dict, List, Union

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# BitsAndBytes and PEFT are optional imports; install via requirements.txt if not in the DLC
try:
    from transformers import BitsAndBytesConfig  # type: ignore
except Exception:  # pragma: no cover
    BitsAndBytesConfig = None  # type: ignore

try:
    from peft import PeftModel  # type: ignore
except Exception:  # pragma: no cover
    PeftModel = None  # type: ignore

try:
    # Convenience JSON encoder/decoder provided by the HF inference toolkit
    from sagemaker_huggingface_inference_toolkit import decoder_encoder  # type: ignore
except Exception:  # pragma: no cover
    decoder_encoder = None  # type: ignore


DEFAULT_PROMPT = (
    "Classify the text into {labels_str} and return the answer as the exact text label.\n"
    "text: {text}\n"
    "label:"
)


def _build_prompt(text: str, labels: List[str]) -> str:
    labels_str = ", ".join(labels)
    tpl = os.environ.get("PROMPT_TEMPLATE", DEFAULT_PROMPT)
    return tpl.format(labels_str=labels_str, text=text)


def _decode_label(generated: str, labels: List[str]) -> str:
    # Normalize model output and pick the first token/line that matches a known label
    y = (generated or "").strip().splitlines()[0].strip()
    # Remove trailing special tokens or punctuation
    y = y.strip().strip('"').strip("'").strip()
    # Exact match or best-effort: case-sensitive first, then case-insensitive
    if y in labels:
        return y
    y_low = y.lower()
    for lab in labels:
        if lab.lower() == y_low:
            return lab
    # As a last resort, return the raw model output
    return y


def model_fn(model_dir: str) -> Dict[str, Any]:
    """
    Load base model + attach LoRA adapters trained by scripts/train.py.
    The training job saved adapters directly into /opt/ml/model (model_dir).
    """
    base_model_id = os.environ.get("BASE_MODEL_ID", "meta-llama/Llama-3.2-1B")
    token = os.environ.get("HF_TOKEN", None)
    use_4bit = os.environ.get("USE_4BIT", "true").lower() == "true"

    # Build optional 4-bit quantization config
    quant_config = None
    if use_4bit and BitsAndBytesConfig is not None and torch.cuda.is_available():
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(base_model_id, use_fast=True, token=token)

    # Base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        token=token,
        quantization_config=quant_config,
    )

    # Attach LoRA adapters saved by training
    # Adapters are expected at model_dir or model_dir/adapter
    if PeftModel is None:
        raise RuntimeError("peft is required to load LoRA adapters. "
                           "Add it to code/requirements.txt.")

    peft_path = model_dir
    alt_path = os.path.join(model_dir, "adapter")
    try_paths = [p for p in [peft_path, alt_path] if os.path.exists(p)]
    if not try_paths:
        # Allow nested directories (common if users keep output_dir subfolders)
        for root, dirs, files in os.walk(model_dir):
            if "adapter_config.json" in files:
                try_paths.append(root)
                break

    if not try_paths:
        raise FileNotFoundError(
            f"Could not find LoRA adapter files under {model_dir}. "
            "Ensure training saved adapters to /opt/ml/model."
        )

    model = PeftModel.from_pretrained(model, try_paths[0], is_trainable=False)
    model.eval()

    return {"model": model, "tokenizer": tok}


def input_fn(input_data: Union[str, bytes], content_type: str) -> Dict[str, Any]:
    if content_type and "application/json" in content_type:
        return json.loads(input_data)
    if decoder_encoder is not None:
        return decoder_encoder.decode(input_data, content_type)
    # Fallback
    return json.loads(input_data)


def predict_fn(data: Dict[str, Any], bundle: Dict[str, Any]) -> Dict[str, Any]:
    model = bundle["model"]
    tok = bundle["tokenizer"]

    texts = data.get("text")
    labels = data.get("labels")
    if texts is None:
        raise ValueError("Request JSON must include 'text' (string or list of strings).")
    if isinstance(texts, str):
        texts = [texts]

    if labels is None:
        # allow default from env if provided
        default_labels = os.environ.get("DEFAULT_LABELS")
        if default_labels:
            labels = json.loads(default_labels)
        else:
            raise ValueError("Request JSON must include 'labels': [..] or set DEFAULT_LABELS env var.")

    prompts = [_build_prompt(t, labels) for t in texts]

    enc = tok(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=int(os.environ.get("MAX_INPUT_LENGTH", "512")),
    )
    if torch.cuda.is_available():
        enc = {k: v.to(model.device if hasattr(model, "device") else "cuda") for k, v in enc.items()}

    # Generation defaults tuned for deterministic label emission
    temperature = float(os.environ.get("TEMPERATURE", "0.0"))
    do_sample = temperature > 0.0
    gen_kwargs = dict(
        max_new_tokens=int(os.environ.get("MAX_NEW_TOKENS", "8")),
        do_sample=do_sample,
        temperature=temperature,
        num_beams=int(os.environ.get("NUM_BEAMS", "1")),
        pad_token_id=tok.eos_token_id,
    )

    with torch.inference_mode():
        out = model.generate(**enc, **gen_kwargs)

    # Slice generated continuation from the prompt
    cont = out[:, enc["input_ids"].shape[1]:]
    decoded = tok.batch_decode(cont, skip_special_tokens=True)

    preds = [_decode_label(y, labels) for y in decoded]
    return {"predictions": preds, "labels": labels}


def output_fn(prediction: Dict[str, Any], accept: str) -> Union[str, bytes]:
    if accept and "application/json" in accept:
        return json.dumps(prediction)
    if decoder_encoder is not None:
        return decoder_encoder.encode(prediction, accept)
    return json.dumps(prediction)
