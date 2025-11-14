#!/bin/bash
set -e

# Get AWS account ID and region
AWS_ACCOUNT_ID=${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}
AWS_REGION=${AWS_REGION:-us-east-1}
IMAGE_TAG=${IMAGE_TAG:-latest}
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

RAW_DATA_PATH=${RAW_DATA_PATH:-"s3://player-churn-data/raw/"}
OUTPUT_PATH=${OUTPUT_PATH:-"s3://player-churn-data/processed/features.parquet"}
DASK_CLUSTER_NAME=${DASK_CLUSTER_NAME:-"preprocessing-cluster"}

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
JOB_TEMPLATE="$SCRIPT_DIR/../jobs/preprocessing-job.yaml"

echo "=========================================="
echo "Submitting Preprocessing Job"
echo "=========================================="
echo "AWS Account: $AWS_ACCOUNT_ID"
echo "Image: ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/player-churn/preprocess:${IMAGE_TAG}"
echo "Raw Data: $RAW_DATA_PATH"
echo "Output: $OUTPUT_PATH"
echo "Dask Cluster: $DASK_CLUSTER_NAME"
echo "=========================================="

# Substitute variables and apply
export AWS_ACCOUNT_ID
export AWS_REGION
export IMAGE_TAG
export TIMESTAMP
export RAW_DATA_PATH
export OUTPUT_PATH
export DASK_CLUSTER_NAME

envsubst < "$JOB_TEMPLATE" | kubectl apply -f -

JOB_NAME="preprocessing-job-${TIMESTAMP}"

echo ""
echo "Job submitted: $JOB_NAME"
echo ""
echo "To follow logs:"
echo "  kubectl logs -n processing job/$JOB_NAME --follow"
echo ""
echo "To check status:"
echo "  kubectl get job -n processing $JOB_NAME"
echo ""
echo "To delete job:"
echo "  kubectl delete job -n processing $JOB_NAME"
