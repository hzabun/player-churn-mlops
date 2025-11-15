#!/bin/bash
set -e

ENV=${1:-dev}
export TIMESTAMP=$(date +%Y%m%d-%H%M%S)

# AWS Configuration
export AWS_ACCOUNT_ID=${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}
export AWS_REGION=${AWS_REGION:-us-east-1}
export IMAGE_TAG=${IMAGE_TAG:-latest}
export CLUSTER_NAME=${CLUSTER_NAME:-player-churn-${ENV}}

# Data paths
export RAW_DATA_PATH=${RAW_DATA_PATH:-"s3://player-churn-bns-mlops/data/raw-parquet/"}
export OUTPUT_FILE_PATH=${OUTPUT_FILE_PATH:-"s3://player-churn-bns-mlops/data/processed/player-features.parquet"}

# Dask cluster configuration
export DASK_CLUSTER_NAME=${DASK_CLUSTER_NAME:-"preprocessing-cluster-${TIMESTAMP}"}
export DASK_NAMESPACE="processing"

# Environment-specific configurations
case "$ENV" in
  dev)
    export DASK_N_WORKERS=${DASK_N_WORKERS:-2}
    export DASK_WORKER_MEMORY_REQUEST=${DASK_WORKER_MEMORY_REQUEST:-2Gi}
    export DASK_WORKER_MEMORY_LIMIT=${DASK_WORKER_MEMORY_LIMIT:-4Gi}
    export DASK_WORKER_CPU_REQUEST=${DASK_WORKER_CPU_REQUEST:-1000m}
    export DASK_WORKER_CPU_LIMIT=${DASK_WORKER_CPU_LIMIT:-2000m}
    export DASK_WORKER_THREADS=${DASK_WORKER_THREADS:-1}
    export DASK_WORKER_MEMORY_LIMIT_GB=${DASK_WORKER_MEMORY_LIMIT_GB:-2GiB}
    ;;
  staging)
    export DASK_N_WORKERS=${DASK_N_WORKERS:-4}
    export DASK_WORKER_MEMORY_REQUEST=${DASK_WORKER_MEMORY_REQUEST:-4Gi}
    export DASK_WORKER_MEMORY_LIMIT=${DASK_WORKER_MEMORY_LIMIT:-8Gi}
    export DASK_WORKER_CPU_REQUEST=${DASK_WORKER_CPU_REQUEST:-2000m}
    export DASK_WORKER_CPU_LIMIT=${DASK_WORKER_CPU_LIMIT:-4000m}
    export DASK_WORKER_THREADS=${DASK_WORKER_THREADS:-2}
    export DASK_WORKER_MEMORY_LIMIT_GB=${DASK_WORKER_MEMORY_LIMIT_GB:-4GiB}
    ;;
  prod)
    export DASK_N_WORKERS=${DASK_N_WORKERS:-8}
    export DASK_WORKER_MEMORY_REQUEST=${DASK_WORKER_MEMORY_REQUEST:-8Gi}
    export DASK_WORKER_MEMORY_LIMIT=${DASK_WORKER_MEMORY_LIMIT:-16Gi}
    export DASK_WORKER_CPU_REQUEST=${DASK_WORKER_CPU_REQUEST:-4000m}
    export DASK_WORKER_CPU_LIMIT=${DASK_WORKER_CPU_LIMIT:-8000m}
    export DASK_WORKER_THREADS=${DASK_WORKER_THREADS:-4}
    export DASK_WORKER_MEMORY_LIMIT_GB=${DASK_WORKER_MEMORY_LIMIT_GB:-8GiB}
    ;;
  *)
    echo "Usage: $0 [dev|staging|prod]"
    exit 1
    ;;
esac

# Scheduler resources (same across all environments)
export DASK_SCHEDULER_MEMORY_REQUEST=${DASK_SCHEDULER_MEMORY_REQUEST:-1Gi}
export DASK_SCHEDULER_MEMORY_LIMIT=${DASK_SCHEDULER_MEMORY_LIMIT:-2Gi}
export DASK_SCHEDULER_CPU_REQUEST=${DASK_SCHEDULER_CPU_REQUEST:-500m}
export DASK_SCHEDULER_CPU_LIMIT=${DASK_SCHEDULER_CPU_LIMIT:-1000m}

# Dask worker configuration
export DASK_WORKER_DEATH_TIMEOUT=${DASK_WORKER_DEATH_TIMEOUT:-60}
export DASK_SERVICE_ACCOUNT="preprocess-sa"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "=========================================="
echo "Submitting Preprocessing Job"
echo "=========================================="
echo "Environment: $ENV"
echo "Timestamp: $TIMESTAMP"
echo "AWS Account: $AWS_ACCOUNT_ID"
echo "AWS Region: $AWS_REGION"
echo "Image Tag: $IMAGE_TAG"
echo ""
echo "Data Configuration:"
echo "  Raw Data: $RAW_DATA_PATH"
echo "  Output: $OUTPUT_FILE_PATH"
echo ""
echo "Dask Cluster Configuration:"
echo "  Cluster Name: $DASK_CLUSTER_NAME"
echo "  Workers: $DASK_N_WORKERS"
echo "  Worker Resources: ${DASK_WORKER_MEMORY_REQUEST}/${DASK_WORKER_MEMORY_LIMIT} memory, ${DASK_WORKER_CPU_REQUEST}/${DASK_WORKER_CPU_LIMIT} CPU"
echo "  Worker Threads: $DASK_WORKER_THREADS"
echo "=========================================="

# Create namespace and service account if they don't exist
echo "Ensuring namespace and service account exist..."
envsubst < "$SCRIPT_DIR/../base/namespace.yaml" | kubectl apply -f -
envsubst < "$SCRIPT_DIR/../base/service-account.yaml" | kubectl apply -f -

# Submit the job
echo "Submitting job..."
envsubst < "$SCRIPT_DIR/../jobs/preprocessing-job.yaml" | kubectl apply -f -

echo ""
echo "=========================================="
echo "Job submitted successfully!"
echo "=========================================="
echo "Job name: preprocessing-job-${TIMESTAMP}"
echo ""
echo "Monitor job status:"
echo "  kubectl get jobs -n processing"
echo "  kubectl get pods -n processing"
echo ""
echo "View logs:"
echo "  kubectl logs -n processing -l job-name=preprocessing-job-${TIMESTAMP} -f"
echo ""
echo "View Dask dashboard (once cluster is created):"
echo "  kubectl port-forward -n processing svc/${DASK_CLUSTER_NAME}-scheduler 8787:8787"
