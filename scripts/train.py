from __future__ import annotations
import argparse, sys
from pathlib import Path

# Ensure 'src' is on PYTHONPATH when run directly
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from src.config import TrainConfig
from src.data import seed_everything, load_splits, build_prompts, to_hf_dataset
from src.model import make_bnb_config, load_model_and_tokenizer, make_lora_config
from src.training import make_trainer

def parse_args():
    p = argparse.ArgumentParser(description="Train LoRA classifier with TRL SFTTrainer")
    p.add_argument("--base_path", type=str, default=r"C:\Users\amiru\Downloads\lora_text_cls_project\lora_text_cls_project\csv_top5_5class", help="Folder containing train/validation/test CSVs")
    p.add_argument("--train_file", type=str, default=None)
    p.add_argument("--val_file", type=str, default=None)
    p.add_argument("--test_file", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="llama-3.2-1b-arxiver-lora")
    p.add_argument("--labels", type=str, nargs="*", default=None, help="Override label list (space-separated)" )
    p.add_argument("--label_column", type=str, default=None)
    p.add_argument("--text_fields", type=str, nargs="*", default=None,
                   help="Text columns to concatenate (space-separated)" )
    p.add_argument("--base_model_name", type=str, default=None)
    return p.parse_args()

def main():
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    args = parse_args()
    cfg = TrainConfig()

    if args.base_path: cfg.base_path = args.base_path
    if args.train_file: cfg.train_file = args.train_file
    if args.val_file: cfg.val_file = args.val_file
    if args.test_file: cfg.test_file = args.test_file
    if args.output_dir: cfg.output_dir = args.output_dir
    if args.label_column: cfg.label_column = args.label_column
    if args.text_fields: cfg.text_fields = args.text_fields
    if args.base_model_name: cfg.base_model_name = args.base_model_name
    if args.labels: cfg.labels = args.labels

    if not (ROOT / cfg.output_dir).parent.exists():
        pass  # just using as relative path

    # 0) Preflight
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for 4-bit training (bitsandbytes)." )

    seed_everything(cfg.seed)

    # 1) Data
    train_df, val_df, test_df = load_splits(cfg.base_path, cfg.train_file, cfg.val_file, cfg.test_file)

    train_prompts, val_prompts, test_prompts, y_true = build_prompts(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        label_col=cfg.label_column,
        text_fields=cfg.text_fields,
        labels=cfg.labels,
    )

    train_ds = to_hf_dataset(train_prompts, "text")
    val_ds = to_hf_dataset(val_prompts, "text")

    # 2) Model + LoRA
    bnb_cfg = make_bnb_config(cfg.load_in_4bit, cfg.bnb_4bit_compute_dtype, cfg.bnb_4bit_use_double_quant, cfg.bnb_4bit_quant_type)
    model, tokenizer = load_model_and_tokenizer(cfg.base_model_name, cfg.hf_token_env, bnb_cfg)
    lora_cfg = make_lora_config(model, cfg.lora_r, cfg.lora_alpha, cfg.lora_dropout)

    # 3) Trainer
    trainer = make_trainer(
        model=model,
        tokenizer=tokenizer,
        train_ds=train_ds,
        val_ds=val_ds,
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=cfg.logging_steps,
        evaluation_strategy=cfg.evaluation_strategy,
        save_strategy=cfg.save_strategy,
        save_total_limit=cfg.save_total_limit,
        report_to=cfg.report_to,
        seed=cfg.seed,
        max_seq_length=cfg.max_seq_length,
        lora_cfg=lora_cfg,
    )

    # 4) Train
    trainer.train()

    # 5) Save adapters + tokenizer
    trainer.model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    print(f"Training complete. Saved LoRA adapters to: {cfg.output_dir}")

if __name__ == "__main__":
    main()
