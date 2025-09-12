from __future__ import annotations
from typing import List, Tuple

import pandas as pd
from datasets import Dataset

def seed_everything(seed: int) -> None:
    import torch, numpy as _np
    _np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_splits(base_path: str, train_file: str, val_file: str, test_file: str):
    import pandas as pd
    import os
    train_df = pd.read_csv(os.path.join(base_path, train_file))
    val_df = pd.read_csv(os.path.join(base_path, val_file))
    test_df = pd.read_csv(os.path.join(base_path, test_file))
    return train_df, val_df, test_df

def combine_text_fields(df: pd.DataFrame, fields: List[str]) -> pd.Series:
    def join_row(row: pd.Series) -> str:
        parts = []
        for f in fields:
            val = row.get(f, None)
            if val is None or (isinstance(val, float) and pd.isna(val)):
                continue
            s = str(val).strip()
            if s:
                parts.append(s)
        if not parts:
            return " "
        text = ". ".join(parts)
        if not text.endswith("."):
            text += "."
        return text
    return df.apply(join_row, axis=1)

def generate_supervised_prompt(text: str, labels_str: str, label: str) -> str:
    return (
        f"Classify the text into {labels_str} and return the answer as the exact text label.\n"
        f"text: {text}\n"
        f"label: {label}"
    )

def generate_inference_prompt(text: str, labels_str: str) -> str:
    return (
        f"Classify the text into {labels_str} and return the answer as the exact text label.\n"
        f"text: {text}\n"
        f"label:"
    )

def build_prompts(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                  label_col: str, text_fields: List[str], labels: List[str]):
    labels_str = ", ".join(labels)
    train_text = combine_text_fields(train_df, text_fields)
    val_text = combine_text_fields(val_df, text_fields)
    test_text = combine_text_fields(test_df, text_fields)

    train_prompts = pd.DataFrame({
        "text": [generate_supervised_prompt(t, labels_str, y)
                 for t, y in zip(train_text.tolist(), train_df[label_col].tolist())]
    })
    val_prompts = pd.DataFrame({
        "text": [generate_supervised_prompt(t, labels_str, y)
                 for t, y in zip(val_text.tolist(), val_df[label_col].tolist())]
    })
    test_prompts = pd.DataFrame({
        "text": [generate_inference_prompt(t, labels_str) for t in test_text.tolist()]
    })
    y_true = test_df[label_col].copy()

    return train_prompts, val_prompts, test_prompts, y_true

def to_hf_dataset(df: pd.DataFrame, text_col: str = "text") -> Dataset:
    clean = df[[text_col]].fillna(" ")
    return Dataset.from_pandas(clean, preserve_index=False)
