from __future__ import annotations

import os

import torch
from trl import SFTTrainer
from transformers import TrainingArguments

def make_trainer(model, tokenizer, train_ds, val_ds, output_dir: str,
                 num_train_epochs: int, per_device_train_batch_size: int,
                 per_device_eval_batch_size: int, gradient_accumulation_steps: int,
                 learning_rate: float, weight_decay: float, warmup_ratio: float,
                 logging_steps: int, evaluation_strategy: str, save_strategy: str,
                 save_total_limit: int, report_to: str, seed: int, max_seq_length: int,
                 lora_cfg):
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        eval_strategy=evaluation_strategy,
        save_strategy=save_strategy,
        save_total_limit=save_total_limit,
        report_to=[report_to],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=False,
        fp16=True,
        tf32=True,
        seed=seed,
        logging_dir=os.path.join(output_dir,'logs')
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        # tokenizer=tokenizer,
        peft_config=lora_cfg,
        # dataset_text_field="text",
        # max_seq_length=max_seq_length,
        # packing=False,
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return trainer
