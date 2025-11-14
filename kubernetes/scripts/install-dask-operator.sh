#!/bin/bash
set -e

echo "Installing Dask Kubernetes Operator..."

helm repo add dask https://helm.dask.org/
helm repo update

helm upgrade --install dask-operator dask/dask-kubernetes-operator \
  --namespace dask-operator \
  --create-namespace \
  --wait

echo "Dask operator installed successfully!"
kubectl get pods -n dask-operator
