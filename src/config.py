from __future__ import annotations
from dataclasses import dataclass, field
from typing import List

@dataclass
class TrainConfig:
    # ==== Data ====
    dataset_name: str = "arxiver"
    base_path: str = "dataset"
    train_file: str = "train.csv"
    val_file: str = "validation.csv"
    test_file: str = "test.csv"
    label_column: str = "label_name"
    # The columns to combine into the input text. If a column is missing, it is ignored.
    text_fields: List[str] = field(default_factory=lambda: ["title", "abstract"])

    # All possible labels (the model will be prompted with these)
    labels: List[str] = field(default_factory=lambda: ['cs.CL', 'cs.CV', 'cs.LG', 'hep-ph', 'quant-ph'])

    # ==== Model / Tokenizer ====
    base_model_name: str = "meta-llama/Llama-3.2-1B"
    output_dir: str = "llama-3.2-1b-arxiver-lora"
    max_seq_length: int = 512
    hf_token_env: str = "HF_TOKEN"  # read from environment

    # ==== LoRA ====
    lora_r: int = 2
    lora_alpha: int = 2
    lora_dropout: float = 0.0

    # ==== Training ====
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-4
    weight_decay: float = 1e-3
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    evaluation_strategy: str = "epoch"   # 'steps' or 'epoch'
    save_strategy: str = "epoch"
    save_total_limit: int = 2
    report_to: str = "tensorboard"
    seed: int = 762_920

    # ==== Quantization (QLoRA defaults) ====
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"  # 'bfloat16' or 'float16'
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    # ==== Inference ====
    gen_max_new_tokens: int = 8
    gen_do_sample: bool = False
    gen_temperature: float = 0.0
