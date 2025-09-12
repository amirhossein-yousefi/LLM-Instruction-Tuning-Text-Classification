# pip install -U datasets pandas
import os
from collections import Counter
import pandas as pd
from datasets import load_dataset, DatasetDict

# ----------------------------
# Config
# ----------------------------
DATASET_ID  = "real-jiakai/arxiver-with-category"
TEXT_COLUMNS = ["title", "abstract"]         # change to include "markdown" if you want it
LABEL_COLUMN = "primary_category"
OUTPUT_DIR   = "csv_top5_5class"
SEED         = 42
TEST_SIZE    = 0.10      # 10% test
VAL_FROM_TRAIN = 1/9     # -> ~10% of total becomes validation (80/10/10 overall)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# 1) Load dataset (single 'train' split)
# ----------------------------
raw = load_dataset(DATASET_ID)
ds_all = raw["train"]

# Ensure the text columns exist
missing = [c for c in TEXT_COLUMNS if c not in ds_all.column_names]
if missing:
    raise ValueError(f"Missing text columns in dataset: {missing}")

# ----------------------------
# 2) Pick the TOP-5 categories by frequency
# ----------------------------
counts = Counter(ds_all[LABEL_COLUMN])
top5 = [lbl for lbl, _ in counts.most_common(5)]
top5_set = set(top5)

print("Top-5 categories:", top5)

# Keep only examples from those 5 classes
ds_top5 = ds_all.filter(lambda ex: ex[LABEL_COLUMN] in top5_set)

# ----------------------------
# 3) Encode labels to ClassLabel ids and add readable names
# ----------------------------
# After encoding, LABEL_COLUMN becomes an int feature with .names
ds_top5 = ds_top5.class_encode_column(LABEL_COLUMN)

# Rename encoded label column -> 'label_id' (clearer downstream)
ds_top5 = ds_top5.rename_column(LABEL_COLUMN, "label_id")
label_names = ds_top5.features["label_id"].names  # id -> name mapping

def add_label_name(example):
    return {"label_name": label_names[example["label_id"]]}

ds_top5 = ds_top5.map(add_label_name)

# Quick sanity: make sure we still have exactly 5 classes
unique_ids = set(ds_top5["label_id"])
assert len(unique_ids) == 5, f"Expected 5 classes, found {len(unique_ids)}"

# ----------------------------
# 4) Stratified 80/10/10 split
# ----------------------------
split_90_10 = ds_top5.train_test_split(
    test_size=TEST_SIZE, seed=SEED, stratify_by_column="label_id"
)
split_train_val = split_90_10["train"].train_test_split(
    test_size=VAL_FROM_TRAIN, seed=SEED, stratify_by_column="label_id"
)

ds_splits = DatasetDict({
    "train":      split_train_val["train"],
    "validation": split_train_val["test"],
    "test":       split_90_10["test"],
})

# ----------------------------
# 5) Save CSVs (title, abstract, label_id, label_name)
# ----------------------------
cols_to_keep = TEXT_COLUMNS + ["label_id", "label_name"]

def save_split(split_name):
    df = ds_splits[split_name].select_columns(cols_to_keep).to_pandas()
    out_path = os.path.join(OUTPUT_DIR, f"{split_name}.csv")
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved {split_name}: {out_path}  (rows={len(df)})")

for split in ["train", "validation", "test"]:
    save_split(split)

# ----------------------------
# 6) (Optional) Print per-split class counts for a quick check
# ----------------------------
for split in ["train", "validation", "test"]:
    ids = ds_splits[split]["label_id"]
    c = Counter(ids)
    # map id -> human-readable name
    pretty = {f"{i}:{label_names[i]}": c[i] for i in sorted(c)}
    print(f"{split} class counts:", pretty)
