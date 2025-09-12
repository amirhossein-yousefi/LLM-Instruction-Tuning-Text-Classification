from __future__ import annotations
from typing import List
from tqdm import tqdm
from transformers import pipeline as hf_pipeline

def build_generation_pipeline(model, tokenizer, max_new_tokens: int,
                              do_sample: bool, temperature: float):
    return hf_pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        pad_token_id=tokenizer.pad_token_id,
    )

def extract_predicted_label(generated_text: str, candidate_labels: List[str]) -> str:
    lower = generated_text.lower()
    anchor = "label:"
    if anchor in lower:
        pos = lower.rfind(anchor)
        raw = generated_text[pos + len(anchor):].strip()
    else:
        raw = generated_text.strip()
    raw = raw.splitlines()[0].strip().strip('"').strip("'")
    for lab in sorted(candidate_labels, key=len, reverse=True):
        if lab.lower() in raw.lower():
            return lab
    return "none"

def predict_labels(prompts_df, gen_pipe, candidate_labels: List[str]) -> List[str]:
    preds = []
    for text in tqdm(prompts_df["text"].tolist(), desc="Predict"):
        out = gen_pipe(text)
        gen_text = out[0]["generated_text"]
        preds.append(extract_predicted_label(gen_text, candidate_labels))
    return preds
