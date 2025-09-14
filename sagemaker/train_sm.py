#!/usr/bin/env python3

"""
Launches a SageMaker training job for LLM-Instruction-Tuning-Text-Classification
using the managed Hugging Face PyTorch Training DLC.

It assumes you have cloned the repository and are running this from the repo root,
or that you pass --source_dir to point at the repo root that contains scripts/train.py
and requirements.txt.
"""
import argparse
import json
import os
import sys
from datetime import datetime

import boto3
import sagemaker
from sagemaker.huggingface import HuggingFace

def parse_args():
    p = argparse.ArgumentParser()
    # Repo/source code & dataset
    p.add_argument("--source_dir", default=".", help="Path to repo root (must include scripts/train.py)")
    p.add_argument("--dataset_dir", default="dataset", help="Local folder with train/validation/test CSVs")
    p.add_argument("--train_file", default="train.csv")
    p.add_argument("--val_file", default="validation.csv")
    p.add_argument("--test_file", default="test.csv")
    p.add_argument("--label_column", default="label_name")
    p.add_argument("--text_fields", nargs="+", default=["title", "abstract"])
    # Model & training infra
    p.add_argument("--base_model_id", default="meta-llama/Llama-3.2-1B")
    p.add_argument("--instance_type", default="ml.g5.2xlarge")
    p.add_argument("--instance_count", type=int, default=1)
    p.add_argument("--volume_size", type=int, default=200)
    p.add_argument("--max_run", type=int, default=24*60*60)  # 24h
    p.add_argument("--region", default=None, help="AWS region (defaults to boto3 default)")
    p.add_argument("--role_arn", default=None, help="Execution role ARN. If not set, sagemaker.get_execution_role() is tried.")
    p.add_argument("--bucket", default=None, help="S3 bucket for inputs/outputs. Defaults to Session.default_bucket().")
    p.add_argument("--job_name", default=None, help="Optional training job name. Auto-generated if omitted.")
    # DLC/image
    p.add_argument("--image_uri", default=None,
                   help="Optional explicit DLC image URI (region-specific). If omitted, transformers/pytorch/py versions are used.")
    p.add_argument("--transformers_version", default="4.49.0")
    p.add_argument("--pytorch_version", default="2.5.1")
    p.add_argument("--py_version", default="py311")
    # HF auth & extra hyperparameters
    p.add_argument("--hf_token", default=None, help="HF token if your base model is gated/private.")
    p.add_argument("--extra_hparams", default="{}", help="JSON string of extra CLI overrides to scripts/train.py (e.g., '{\"epochs\":3}')")
    return p.parse_args()

def main():
    args = parse_args()

    # Create session
    boto_sess = boto3.session.Session(region_name=args.region) if args.region else boto3.session.Session()
    region = boto_sess.region_name
    sm_sess = sagemaker.Session(boto_session=boto_sess)

    # Role
    if args.role_arn:
        role = args.role_arn
    else:
        try:
            role = sagemaker.get_execution_role()
        except Exception:
            raise RuntimeError("role_arn not provided and get_execution_role() failed. Provide --role_arn explicitly.")

    # Bucket & S3 prefixes
    bucket = args.bucket or sm_sess.default_bucket()
    prefix = f"llm-instruction-tuning-text-classification/{datetime.utcnow().strftime('%Y-%m-%d')}"

    print(f"[INFO] Using region={region}, bucket=s3://{bucket}, role={role}")

    # Upload dataset directory to S3 and bind to 'training' channel
    if not os.path.isdir(args.dataset_dir):
        raise FileNotFoundError(f"dataset_dir '{args.dataset_dir}' not found")
    s3_data_uri = sm_sess.upload_data(path=args.dataset_dir, bucket=bucket, key_prefix=f"{prefix}/data")
    print(f"[INFO] Uploaded dataset to: {s3_data_uri}")

    # Hyperparameters: these are mapped to scripts/train.py CLI flags
    # NOTE: SageMaker passes them as --key value pairs
    extra = json.loads(args.extra_hparams or "{}")
    hps = {
        "base_path": "/opt/ml/input/data/training",  # bound to our 'training' channel below
        "train_file": args.train_file,
        "val_file": args.val_file,
        "test_file": args.test_file,
        "label_column": args.label_column,
        "text_fields": " ".join(args.text_fields),
        "base_model_name": args.base_model_id,
        # IMPORTANT: ensure the adapters are saved where SageMaker collects model artifacts.
        "output_dir": "/opt/ml/model",
    }
    # Merge any extra overrides
    for k, v in (extra.items() if isinstance(extra, dict) else {}.items()):
        hps[str(k)] = str(v)

    # Create the estimator
    hf_kwargs = dict(
        entry_point="scripts/train.py",
        source_dir=args.source_dir,
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        volume_size=args.volume_size,
        max_run=args.max_run,
        role=role,
        hyperparameters=hps,
        environment={},
        sagemaker_session=sm_sess,
    )

    if args.hf_token:
        hf_kwargs["environment"]["HF_TOKEN"] = args.hf_token

    if args.image_uri:
        hf_kwargs["image_uri"] = args.image_uri
        print(f"[INFO] Using explicit image_uri: {args.image_uri}")
    else:
        # Use versioned DLC selection
        hf_kwargs.update(dict(
            transformers_version=args.transformers_version,
            pytorch_version=args.pytorch_version,
            py_version=args.py_version,
        ))
        print(f"[INFO] Using DLC: transformers={args.transformers_version}, pytorch={args.pytorch_version}, py={args.py_version}")

    estimator = HuggingFace(**hf_kwargs)

    # Name
    job_name = args.job_name or f"lora-tc-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    print(f"[INFO] Starting training job: {job_name}")

    # Launch
    estimator.fit(inputs={"training": s3_data_uri}, job_name=job_name, wait=True)
    print(f"[OK] Training complete. Model artifacts: {estimator.model_data}")

    # Save a small JSON with metadata to reuse in deployment
    meta = {
        "region": region,
        "bucket": bucket,
        "s3_model_artifact": estimator.model_data,
        "job_name": job_name,
        "base_model_id": args.base_model_id,
        "label_column": args.label_column,
        "text_fields": args.text_fields,
    }
    meta_path = os.path.join(os.getcwd(), f"{job_name}-training-metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[OK] Wrote: {meta_path}")

if __name__ == "__main__":
    main()
