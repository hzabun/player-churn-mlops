#!/bin/bash
set -e

ENV=${1:-dev}
export TIMESTAMP=$(date +%Y%m%d-%H%M%S)

# AWS Configuration
export AWS_ACCOUNT_ID=${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}
export AWS_REGION=${AWS_REGION:-us-east-1}
export IMAGE_TAG=${IMAGE_TAG:-latest}

# Data paths
export LABEL_FILE_PATH=${LABEL_FILE_PATH:-s3://player-churn-bns-mlops/data/processed/train-labels.parquet}
export FEATURE_STORE_PATH=${FEATURE_STORE_PATH:-/feast}

# Training config
export TEST_SIZE=${TEST_SIZE:-0.2}
export VAL_SIZE=${VAL_SIZE:-0.1}
export RANDOM_STATE=${RANDOM_STATE:-42}

# MLFlow config
export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-http://mlflow-server.mlflow.svc.cluster.local:5000}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-player_churn_prediction}
export RUN_NAME=${RUN_NAME:-test_run}

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "=========================================="
echo "Submitting Training Job"
echo "=========================================="
echo "Environment: $ENV"
echo "Timestamp: $TIMESTAMP"
echo "AWS Account: $AWS_ACCOUNT_ID"
echo "AWS Region: $AWS_REGION"
echo "Image Tag: $IMAGE_TAG"
echo ""
echo "Data Configuration:"
echo "  Label file path: $LABEL_FILE_PATH"
echo "  Feast repo: $FEATURE_STORE_PATH"
echo "=========================================="


# Submit the job
echo "Submitting job..."
envsubst < "$SCRIPT_DIR/../jobs/preprocessing-job.yaml" | kubectl apply -f -

echo ""
echo "=========================================="
echo "Job submitted successfully!"
echo "=========================================="
echo "Job name: training-job-${TIMESTAMP}"
echo ""
echo "Monitor job status:"
echo "  kubectl get jobs -n training"
echo "  kubectl get pods -n training"
echo ""
echo "View logs:"
echo "  kubectl logs -n training -l job-name=training-job-${TIMESTAMP} -f"
echo ""
