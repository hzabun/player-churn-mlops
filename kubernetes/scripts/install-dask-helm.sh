helm repo add dask https://helm.dask.org/
helm repo update
helm install dask-operator dask/dask-kubernetes-operator --namespace dask-operator --create-namespace
