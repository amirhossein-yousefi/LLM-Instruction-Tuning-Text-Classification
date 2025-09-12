import os, json
from collections import Counter
import pandas as pd
from datasets import load_dataset, DatasetDict

DATASET_ID = "real-jiakai/arxiver-with-category"
TEXT_COLUMNS = ["title", "abstract"]
LABEL_COLUMN = "primary_category"
OUTPUT_DIR = "export_csv_arxiver_splits"
os.makedirs(OUTPUT_DIR, exist_ok=True)

raw = load_dataset(DATASET_ID)
ds_all = raw["train"]

# Filter rare classes to avoid stratification errors
MIN_COUNT = 3
counts = Counter(ds_all[LABEL_COLUMN])
keep = {lbl for lbl, cnt in counts.items() if cnt >= MIN_COUNT}
ds_all = ds_all.filter(lambda ex: ex[LABEL_COLUMN] in keep)

# Encode to ClassLabel ids
ds_all = ds_all.class_encode_column(LABEL_COLUMN)

label_names = ds_all.features[LABEL_COLUMN].names
id2label = {i: n for i, n in enumerate(label_names)}
label2id = {n: i for i, n in id2label.items()}

# 80/10/10 via (10% test) + (1/9) val-from-train
split_90_10 = ds_all.train_test_split(test_size=0.10, seed=42, stratify_by_column=LABEL_COLUMN)
split_train_val = split_90_10["train"].train_test_split(test_size=1/9, seed=42, stratify_by_column=LABEL_COLUMN)

ds = DatasetDict({
    "train": split_train_val["train"],
    "validation": split_train_val["test"],
    "test": split_90_10["test"],
})

# Add label_id/label_name for saving
def add_label_cols(example):
    lid = example[LABEL_COLUMN]
    return {"label_id": lid, "label_name": label_names[lid]}

for split in ["train", "validation", "test"]:
    ds[split] = ds[split].map(add_label_cols)
    cols_to_keep = TEXT_COLUMNS + ["label_id", "label_name"]
    df = ds[split].select_columns(cols_to_keep).to_pandas()
    out_path = os.path.join(OUTPUT_DIR, f"{split}.csv")
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved {split} CSV:", out_path)

# Save mapping files (same across splits)
pd.DataFrame({"label_id": list(id2label.keys()), "label_name": list(id2label.values())}) \
  .sort_values("label_id") \
  .to_csv(os.path.join(OUTPUT_DIR, "label_mapping.csv"), index=False, encoding="utf-8")

with open(os.path.join(OUTPUT_DIR, "label_mapping.json"), "w", encoding="utf-8") as f:
    json.dump({"id2label": id2label, "label2id": label2id}, f, ensure_ascii=False, indent=2)
