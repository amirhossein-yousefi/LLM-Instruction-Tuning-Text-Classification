# LLM Instruction‑Tuning for Text Classification (LoRA + QLoRA)

> **Train a decoder‑only LLM to follow *classification* instructions using LoRA adapters and 4‑bit quantization.**  
> Minimal, modular, and fast: one config, two scripts (`train.py`, `predict.py`).

---

## 🚀 TL;DR

This repo instruction‑tunes a decoder‑only LLM (default: **`meta-llama/Llama-3.2-1B`**) to classify short texts (e.g., arXiv titles/abstracts) **by responding with an exact label string**. It uses:

- **PEFT/LoRA** adapters (tiny trainable layers) instead of full fine‑tuning
- **4‑bit (QLoRA‑style)** quantization for memory‑efficient training/inference
- **TRL’s `SFTTrainer`** for supervised instruction tuning
- Simple **prompt templates** that turn classification into a generative task

Out of the box, it ships with an **arXiv‑style demo** (5 labels) and example results (~94% accuracy).

---

## ✨ What’s inside

```
LLM-Instruction-Tuning-Text-Classification/
├── scripts/
│   ├── train.py           # one‑command training with TRL + LoRA
│   └── predict.py         # attach LoRA adapters & run inference/eval
├── src/
│   ├── config.py          # dataclass: all tunables live here (TrainConfig)
│   ├── data.py            # CSV I/O, prompt builders, seeding, HF dataset adapter
│   ├── model.py           # 4‑bit (bitsandbytes) + tokenizer + LoRA configs
│   ├── training.py        # wrapper that builds TRL SFTTrainer
│   ├── infer.py           # generation pipeline & label decoding
│   ├── metrics.py         # accuracy/F1, classification report, confusion matrix
│   ├── prepare_all_arxiver_data.py
│   └── prepare_top_5_arxiver_data.py
├── requirements.txt
├── .gitignore
└── results.txt            # sample evaluation output
```

---

## 🧠 Why instruction‑tune a classifier?

Instead of training a bespoke encoder classifier, we **teach a general‑purpose LLM to follow a classification instruction**. This brings three benefits:

1. **Flexibility:** add/remove labels or extend the schema without changing the head.
2. **Transfer:** one prompting style scales across domains/tasks.
3. **Traceability:** prompts & responses are plain text—easy to debug and audit.

---

## 🧩 Prompt format

The project formulates classification as short **instruction → answer** exchanges.

**Training prompts** (supervised):  

```text
Classify the text into {labels_str} and return the answer as the exact text label.
text: {text}
label: {gold_label}
```

**Inference prompts** (generation):  

```text
Classify the text into {labels_str} and return the answer as the exact text label.
text: {text}
label:
```

> The model is evaluated by string‑matching its generated label to the ground truth.

---

## 📦 Data format

By default, the repo expects three CSVs inside a folder (see `TrainConfig.base_path`):

- `train.csv`
- `validation.csv`
- `test.csv`

Each CSV should include:
- **`label_name`** (or customize via `--label_column`)
- One or more text fields, default: **`title`**, **`abstract`** (customize via `--text_fields`)

**Minimal example (`train.csv`):**

```csv
title,abstract,label_name
"Neural topic models with...", "We propose...", "cs.CL"
"Vision transformer for...", "We revisit...", "cs.CV"
```

If a text column is missing or blank, it’s skipped automatically. Text fields are concatenated into a single sentence with proper punctuation.

> Utilities such as `src/prepare_top_5_arxiver_data.py` can help produce demo CSVs. Feel free to adapt them to your own corpus.

---

## ⚙️ Configuration (everything in one place)

All defaults live in `src/config.py` (`TrainConfig`). Highlights:

- **Base model:** `meta-llama/Llama-3.2-1B`
- **Labels (demo):** `['cs.CL', 'cs.CV', 'cs.LG', 'hep-ph', 'quant-ph']`
- **LoRA:** `r=2`, `alpha=2`, `dropout=0.0` (easy to scale up)
- **Quantization:** 4‑bit NF4 + bfloat16 compute (QLoRA‑style)
- **Sequence length:** 512
- **Trainer:** epochs, batch sizes, eval/save/logging strategies, seed
- **Generation:** deterministic by default (`temperature=0.0`, few tokens)

You can override any config at the CLI (see below).

---

## 🛠️ Setup

> Python ≥ 3.10 recommended.

```bash
# 1) Create & activate a virtual env
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

# 2) Install deps
pip install --upgrade pip
pip install -r requirements.txt

# 3) (If your base HF model requires access) set a token:
# Linux/macOS
export HF_TOKEN=YOUR_HF_ACCESS_TOKEN
# Windows (PowerShell)
# $env:HF_TOKEN="YOUR_HF_ACCESS_TOKEN"
```

> **GPU note:** The project expects an NVIDIA CUDA GPU for 4‑bit (bitsandbytes) training and inference. CPU‑only runs are not supported by the provided scripts.

---

## 🏃‍♀️ Quickstart: Train

The simplest run trains LoRA adapters on your CSVs and saves them to an output folder:

```bash
python scripts/train.py   --base_path dataset   --train_file train.csv   --val_file validation.csv   --test_file test.csv   --label_column label_name   --text_fields title abstract   --base_model_name meta-llama/Llama-3.2-1B   --output_dir llama-3.2-1b-arxiver-lora
```

What happens under the hood:

- Loads CSVs → builds *instruction* prompts for train/val/test
- Loads the base model in **4‑bit** with **bitsandbytes**
- Wraps the model with **LoRA** adapters (PEFT)
- Runs **TRL’s `SFTTrainer`** for one or more epochs
- Saves the LoRA weights + tokenizer to `--output_dir`

---

## 🔮 Inference & evaluation

Attach the saved LoRA adapters to the base model and predict:

```bash
python scripts/predict.py   --base_path dataset   --test_file test.csv   --base_model_name meta-llama/Llama-3.2-1B   --output_dir llama-3.2-1b-arxiver-lora   --save_csv predictions.csv
```

This script:

- Recreates the **inference prompt** for each test row
- Generates the label deterministically (`temperature=0.0` by default)
- Computes **accuracy**, **micro/macro F1**, a **classification report**, and a **confusion matrix**
- Optionally writes `predictions.csv` (prompt, gold, pred)

---

## 📈 Reference results (demo)

The included `results.txt` (arXiv‑style 5‑label demo) reports:

- **Accuracy:** 0.938  
- **F1 (micro):** 0.938  
- **F1 (macro):** 0.950

Per‑class performance and a confusion matrix are also provided in that file. Your mileage will vary with different seeds, tokenization, and label sets.

---

## 🧪 Reproducibility

- Global seeding via `seed_everything()`
- Deterministic generation defaults for evaluation
- All hyperparameters centralized in `TrainConfig` for easy experiment tracking

> Tip: Log to **TensorBoard** by keeping `report_to="tensorboard"` and launching:
>
> ```bash
> tensorboard --logdir .
> ```

---

## 🧯 Troubleshooting

- **`CUDA GPU is required`** — The scripts guard against CPU runs because 4‑bit quantization depends on CUDA (bitsandbytes). Make sure NVIDIA drivers + CUDA toolkit are properly installed for your environment.
- **`bitsandbytes` import errors** — Verify the installed `bitsandbytes` version and CUDA compatibility; reinstall if needed. Some Windows setups may require WSL2 for a smoother experience.
- **Model access** — For gated models (e.g., Llama family), ensure you’ve accepted the license on Hugging Face and set `HF_TOKEN` in your environment.

---

## 🔧 Adapting to your dataset

- Replace/extend **labels** with `--labels` or edit `TrainConfig.labels`
- Point to your **label column** with `--label_column`
- Provide one or more **text fields** with `--text_fields title abstract body ...`
- Consider increasing **LoRA rank** (`lora_r`) and **alpha** for tougher tasks
- Increase **context length** (`max_seq_length`) if your texts are long

---

## 🗺️ Roadmap ideas

- Add support for *multi‑label* classification (comma‑separated labels)
- Add few‑shot exemplars to the prompt template
- Provide ready‑made dataset builders for common corpora
- Export to merged full‑precision checkpoint for deployment

---

## 📚 References (tech used here)

- **LoRA adapters (PEFT)** — Low‑rank adapters for parameter‑efficient fine‑tuning  
- **TRL SFTTrainer** — Supervised fine‑tuning for decoder‑only LLMs  
- **4‑bit quantization (NF4)** — Memory‑efficient training/inference with bitsandbytes  
- **Llama 3.2 models** — 1B/3B multilingual text‑only models; some variants are gated

---

## 🤝 Acknowledgements

Thanks to the open‑source communities behind **Transformers**, **TRL**, **PEFT**, **bitsandbytes**, and the **Llama** model family. This project stands on their shoulders.

---

## 📄 License

No explicit license file is provided in this repository at the time of writing. If you plan to use or redistribute the code, please add an appropriate license file and abide by any upstream model licenses (e.g., Llama).

---

## 🙌 Contributing

Issues and PRs that improve docs, add datasets, or extend training recipes are very welcome!

