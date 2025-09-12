from __future__ import annotations
from typing import List, Tuple
import os
import torch
import bitsandbytes as bnb
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig

def _get_hf_token(env_var: str) -> str | None:
    token = os.getenv(env_var)
    if not token:
        print(f"[WARN] Env var {env_var} not set; if model is gated/private, set it: export {env_var}=hf_***")
    return token

def make_bnb_config(load_in_4bit: bool, bnb_4bit_compute_dtype: str,
                    bnb_4bit_use_double_quant: bool, bnb_4bit_quant_type: str) -> BitsAndBytesConfig:
    compute_dtype = torch.bfloat16 if bnb_4bit_compute_dtype.lower() == "bfloat16" else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
    )

def load_model_and_tokenizer(base_model_name: str, hf_token_env: str,
                             bnb_cfg: BitsAndBytesConfig):
    token = _get_hf_token(hf_token_env)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_cfg,
        token=token,
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, token=token, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer

def get_lora_target_modules(model) -> List[str]:
    cls = bnb.nn.Linear4bit
    names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            parts = name.split(".")
            names.add(parts[-1] if len(parts) > 1 else parts[0])
    names.discard("lm_head")
    return sorted(list(names))

def make_lora_config(model, r: int, alpha: int, dropout: float) -> LoraConfig:
    target_modules = get_lora_target_modules(model)
    if not target_modules:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    print(f"[LoRA] Target modules: {target_modules}")
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
