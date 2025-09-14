#!/usr/bin/env python3

"""
Simple client to test the SageMaker endpoint created by deploy_sm.py.
"""
import argparse
import json
import boto3

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint_name", required=True)
    p.add_argument("--text", required=True, help="Input text to classify")
    p.add_argument("--labels", required=True, help="JSON list of allowed labels, e.g., '[\"cs.CL\",\"cs.CV\",...]'")
    p.add_argument("--region", default=None)
    return p.parse_args()

def main():
    args = parse_args()
    rt = boto3.client("sagemaker-runtime", region_name=args.region)
    payload = {"text": args.text, "labels": json.loads(args.labels)}
    resp = rt.invoke_endpoint(
        EndpointName=args.endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload).encode("utf-8"),
        Accept="application/json",
    )
    body = resp["Body"].read().decode("utf-8")
    print(body)

if __name__ == "__main__":
    main()
