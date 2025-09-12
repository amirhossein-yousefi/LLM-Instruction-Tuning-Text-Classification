from __future__ import annotations
import argparse, sys, os
from pathlib import Path

# Ensure 'src' is on PYTHONPATH when run directly
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from src.config import TrainConfig
from src.data import seed_everything, load_splits, build_prompts
from src.model import make_bnb_config, load_model_and_tokenizer
from src.infer import build_generation_pipeline, predict_labels
from src.metrics import evaluate_predictions
from peft import PeftModel
import pandas as pd

def parse_args():
    p = argparse.ArgumentParser(description="Predict on test set using trained LoRA adapters")
    p.add_argument("--base_path", type=str, default=r"C:\Users\amiru\Downloads\lora_text_cls_project\lora_text_cls_project\csv_top5_5class", help="Folder containing test CSV")
    p.add_argument("--test_file", type=str, default=None)
    p.add_argument("--output_dir", type=str,default= "llama-3.2-1b-arxiver-lora" , help="Directory with saved LoRA adapters")
    p.add_argument("--save_csv", type=str, default="predictions.csv", help="Optional: path to save test predictions CSV")
    p.add_argument("--base_model_name", type=str, default=None, help="Base model (must match training)" )
    return p.parse_args()

def main():
    args = parse_args()
    cfg = TrainConfig()
    cfg.base_path = args.base_path
    if args.test_file: cfg.test_file = args.test_file
    if args.base_model_name: cfg.base_model_name = args.base_model_name
    cfg.output_dir = args.output_dir

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for 4-bit inference (bitsandbytes)." )

    seed_everything(cfg.seed)

    # Only need test split here; still reading train/val to reuse builder signature
    train_df, val_df, test_df = load_splits(cfg.base_path, cfg.train_file, cfg.val_file, cfg.test_file)
    _, _, test_prompts, y_true = build_prompts(
        train_df=train_df, val_df=val_df, test_df=test_df,
        label_col=cfg.label_column, text_fields=cfg.text_fields, labels=cfg.labels
    )

    # Load base + attach LoRA
    bnb_cfg = make_bnb_config(cfg.load_in_4bit, cfg.bnb_4bit_compute_dtype, cfg.bnb_4bit_use_double_quant, cfg.bnb_4bit_quant_type)
    base_model, tokenizer = load_model_and_tokenizer(cfg.base_model_name, cfg.hf_token_env, bnb_cfg)

    # Attach adapter
    model = PeftModel.from_pretrained(base_model, cfg.output_dir)
    # Optionally: model = model.merge_and_unload()  # if you want a merged checkpoint

    gen_pipe = build_generation_pipeline(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=cfg.gen_max_new_tokens,
        do_sample=cfg.gen_do_sample,
        temperature=cfg.gen_temperature,
    )

    y_pred = predict_labels(test_prompts, gen_pipe, cfg.labels)
    metrics = evaluate_predictions(y_true, y_pred, cfg.labels)

    if args.save_csv:
        out_df = pd.DataFrame({
            "prompt": test_prompts["text"].tolist(),
            "y_true": y_true.tolist(),
            "y_pred": y_pred,
        })
        out_df.to_csv(args.save_csv, index=False)
        print(f"Saved predictions to: {args.save_csv}")

if __name__ == "__main__":
    main()
