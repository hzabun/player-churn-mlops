# kubernetes/scripts/deploy-mlflow.sh
#!/bin/bash
set -e

ENV=${1:-dev}

# AWS Configuration
export AWS_ACCOUNT_ID=${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}
export AWS_REGION=${AWS_REGION:-us-east-1}
export IMAGE_TAG=${IMAGE_TAG:-latest}
export CLUSTER_NAME=${CLUSTER_NAME:-player-churn-${ENV}}
export S3_BUCKET_NAME=${S3_BUCKET_NAME:-player-churn-bns-mlops}

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "=========================================="
echo "Deploying MLflow Tracking Server"
echo "=========================================="
echo "Environment: $ENV"
echo "AWS Account: $AWS_ACCOUNT_ID"
echo "AWS Region: $AWS_REGION"
echo "Cluster Name: $CLUSTER_NAME"
echo "S3 Bucket: $S3_BUCKET_NAME"
echo "Image Tag: $IMAGE_TAG"
echo "=========================================="

# Deploy MLflow server
echo "Deploying MLflow server..."
envsubst < "$SCRIPT_DIR/../deployments/mlflow-server.yaml" | kubectl apply -f -

echo ""
echo "=========================================="
echo "MLflow Deployment Complete!"
echo "=========================================="
echo ""
echo "Check deployment status:"
echo "  kubectl get pods -n mlflow"
echo "  kubectl get svc -n mlflow"
echo ""
echo "View logs:"
echo "  kubectl logs -n mlflow -l app=mlflow-server -f"
echo ""
echo "Port forward to access locally:"
echo "  kubectl port-forward -n mlflow svc/mlflow-server 5000:5000"
echo ""
echo "Then access at: http://localhost:5000"
echo ""
echo "Internal cluster URL: http://mlflow-server.mlflow.svc.cluster.local:5000"
echo ""
