#!/bin/bash

echo "Forwarding Dask dashboard to localhost:8787..."
kubectl port-forward -n processing svc/preprocessing-cluster-scheduler 8787:8787
