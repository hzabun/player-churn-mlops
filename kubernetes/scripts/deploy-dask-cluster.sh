#!/bin/bash
set -e

ENV=${1:-dev}
AWS_ACCOUNT_ID=${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}
AWS_REGION=${AWS_REGION:-us-east-1}
IMAGE_TAG=${IMAGE_TAG:-latest}

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MANIFESTS_DIR="$SCRIPT_DIR/../manifests/dask"

if [[ ! "$ENV" =~ ^(dev|staging|prod)$ ]]; then
  echo "Usage: $0 [dev|staging|prod]"
  exit 1
fi

echo "=========================================="
echo "Deploying Dask cluster for environment: $ENV"
echo "AWS Account: $AWS_ACCOUNT_ID"
echo "AWS Region: $AWS_REGION"
echo "Image Tag: $IMAGE_TAG"
echo "=========================================="

# Apply base manifests with variable substitution
echo "Creating namespace and service account..."
export AWS_ACCOUNT_ID AWS_REGION IMAGE_TAG
envsubst < "$SCRIPT_DIR/../base/namespace.yaml" | kubectl apply -f -
envsubst < "$SCRIPT_DIR/../base/service-account.yaml" | kubectl apply -f -

# Deploy Dask cluster using DaskCluster CRD
echo "Deploying DaskCluster CRD..."
envsubst < "$MANIFESTS_DIR/cluster-${ENV}.yaml" | kubectl apply -f -

echo "Waiting for cluster to be ready..."
kubectl wait --for=condition=ready pod \
  -l dask.org/cluster-name=preprocessing-cluster,dask.org/component=scheduler \
  -n processing \
  --timeout=5m

echo ""
echo "=========================================="
echo "Dask cluster deployed successfully!"
echo "=========================================="
kubectl get daskclusters -n processing
echo ""
kubectl get pods -n processing
echo ""
echo "To access Dask dashboard:"
echo "  kubectl port-forward -n processing svc/preprocessing-cluster-scheduler 8787:8787"
echo "  Then open: http://localhost:8787"
