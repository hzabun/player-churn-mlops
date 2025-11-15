#!/bin/bash
set -e

echo "Cleaning up Dask resources..."

# Delete Helm release
helm uninstall dask-operator -n dask-operator || true

# Delete namespace
kubectl delete namespace dask-operator --ignore-not-found=true

echo "Cleanup complete!"
