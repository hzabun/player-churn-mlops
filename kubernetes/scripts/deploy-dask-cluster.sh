#!/bin/bash
set -e

ENV=${1:-dev}
AWS_ACCOUNT_ID=${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}
AWS_REGION=${AWS_REGION:-us-east-1}
IMAGE_TAG=${IMAGE_TAG:-latest}

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
HELM_VALUES_DIR="$SCRIPT_DIR/../helm/values/dask"

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
envsubst < "$SCRIPT_DIR/../base/namespace.yaml" | kubectl apply -f -
envsubst < "$SCRIPT_DIR/../base/service-account.yaml" | kubectl apply -f -

# Deploy Dask cluster with Helm
echo "Deploying Dask cluster..."
helm upgrade --install preprocessing-cluster dask/dask \
  --namespace processing \
  --values "$HELM_VALUES_DIR/values.yaml" \
  --values "$HELM_VALUES_DIR/values-${ENV}.yaml" \
  --set worker.image.repository="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/player-churn/preprocess" \
  --set worker.image.tag="${IMAGE_TAG}" \
  --set scheduler.image.repository="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/player-churn/preprocess" \
  --set scheduler.image.tag="${IMAGE_TAG}" \
  --wait \
  --timeout 5m

echo ""
echo "=========================================="
echo "Dask cluster deployed successfully!"
echo "=========================================="
kubectl get pods -n processing
echo ""
echo "To access Dask dashboard:"
echo "  kubectl port-forward -n processing svc/preprocessing-cluster-scheduler 8787:8787"
echo "  Then open: http://localhost:8787"
