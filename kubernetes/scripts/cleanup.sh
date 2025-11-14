#!/bin/bash
set -e

echo "Cleaning up Dask resources..."

# Delete Helm releases
helm uninstall preprocessing-cluster -n processing || true
helm uninstall dask-operator -n dask-operator || true

# Delete namespaces
kubectl delete namespace processing --ignore-not-found=true
kubectl delete namespace dask-operator --ignore-not-found=true

echo "Cleanup complete!"
