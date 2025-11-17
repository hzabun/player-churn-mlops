#!/bin/bash
set -e

# Get connection string directly from Terraform output
CONNECTION_STRING=$(terraform -chdir=terraform/environments/dev output -raw rds_connection_string)

# Create Kubernetes secret
kubectl create secret generic mlflow-rds-secret \
  --from-literal=connection_string="$CONNECTION_STRING" \
  --namespace=mlflow \
  --dry-run=client -o yaml | kubectl apply -f -

echo "✅ MLflow RDS secret created successfully!"
