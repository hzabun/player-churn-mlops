# Kubernetes Deployment Guide

This directory contains Kubernetes manifests and Helm configurations for deploying the player churn MLOps pipeline.

## Directory Structure

```
kubernetes/
├── base/                          # Base K8s resources
│   ├── namespace.yaml            # Processing namespace
│   └── service-account.yaml      # ServiceAccount with IAM role for S3 access
├── helm/
│   └── values/
│       └── dask/                 # Dask Helm values
│           ├── values.yaml       # Base configuration
│           ├── values-dev.yaml   # Development environment
│           ├── values-staging.yaml # Staging environment
│           └── values-prod.yaml  # Production environment
├── jobs/
│   └── preprocessing-job.yaml    # Kubernetes Job for preprocessing
└── scripts/
    ├── install-dask-operator.sh   # Install Dask Kubernetes Operator
    ├── deploy-dask-cluster.sh     # Deploy Dask cluster
    ├── submit-preprocessing-job.sh # Submit preprocessing job
    ├── cleanup.sh                 # Clean up all resources
    └── port-forward-dashboard.sh  # Access Dask dashboard
```

## Prerequisites

1. **EKS cluster** is running and configured
2. **kubectl** is configured to access your cluster
3. **Helm 3** is installed
4. **IAM Role** for S3 access is created and ARN is set in `service-account.yaml`
5. **Container images** are pushed to ECR

## Quick Start

### 1. Set Environment Variables

```bash
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export AWS_REGION=us-east-1
```

### 2. Install Dask Operator (One-time setup)

```bash
./kubernetes/scripts/install-dask-operator.sh
```

### 3. Deploy Dask Cluster

For development:

```bash
./kubernetes/scripts/deploy-dask-cluster.sh dev
```

For staging:

```bash
./kubernetes/scripts/deploy-dask-cluster.sh staging
```

For production:

```bash
./kubernetes/scripts/deploy-dask-cluster.sh prod
```

### 4. Access Dask Dashboard

```bash
./kubernetes/scripts/port-forward-dashboard.sh
```

Then open: http://localhost:8787

### 5. Submit Preprocessing Job

```bash
# Use defaults
./kubernetes/scripts/submit-preprocessing-job.sh

# Or customize
export RAW_DATA_PATH="s3://your-bucket/raw/"
export OUTPUT_PATH="s3://your-bucket/processed/features.parquet"
export IMAGE_TAG="v1.0.0"
./kubernetes/scripts/submit-preprocessing-job.sh
```

### 6. Monitor Job

```bash
# List jobs
kubectl get jobs -n processing

# Follow logs
kubectl logs -n processing job/preprocessing-job-<timestamp> --follow

# Check job status
kubectl describe job -n processing preprocessing-job-<timestamp>
```

## Environment Configuration

### Development

- **Workers**: 2
- **Memory**: 2-4Gi per worker
- **CPU**: 1000m per worker
- **Jupyter**: Enabled for debugging

### Staging

- **Workers**: 4
- **Memory**: 4-8Gi per worker
- **CPU**: 2000m per worker
- **Jupyter**: Disabled

### Production

- **Workers**: 10 (with HA scheduler)
- **Memory**: 8-16Gi per worker
- **CPU**: 4000m per worker
- **Node Affinity**: compute-optimized instances

## IAM Configuration

The ServiceAccount requires an IAM role with the following permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::player-churn-data/*",
        "arn:aws:s3:::player-churn-data"
      ]
    }
  ]
}
```

Trust relationship:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::<ACCOUNT_ID>:oidc-provider/oidc.eks.<REGION>.amazonaws.com/id/<CLUSTER_ID>"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "oidc.eks.<REGION>.amazonaws.com/id/<CLUSTER_ID>:sub": "system:serviceaccount:processing:preprocess-sa"
        }
      }
    }
  ]
}
```

## Scaling

### Scale Workers Up/Down

```bash
# Scale to 8 workers
kubectl scale daskcluster preprocessing-cluster -n processing --replicas=8

# Or update Helm values and redeploy
helm upgrade preprocessing-cluster dask/dask \
  --namespace processing \
  --set worker.replicas=8 \
  --reuse-values
```

### Autoscaling (Optional)

Add to `values.yaml`:

```yaml
worker:
  autoscaling:
    enabled: true
    minReplicas: 2
    maxReplicas: 20
    targetCPUUtilizationPercentage: 70
```

## Troubleshooting

### Cluster Won't Start

```bash
# Check operator logs
kubectl logs -n dask-operator deployment/dask-operator

# Check cluster status
kubectl get daskcluster -n processing
kubectl describe daskcluster preprocessing-cluster -n processing
```

### Workers Can't Access S3

```bash
# Verify ServiceAccount
kubectl get sa preprocess-sa -n processing -o yaml

# Check IAM role annotation
kubectl get sa preprocess-sa -n processing -o jsonpath='{.metadata.annotations.eks\.amazonaws\.com/role-arn}'

# Test from within a pod
kubectl run -it --rm debug --image=amazon/aws-cli --serviceaccount=preprocess-sa -n processing -- s3 ls s3://player-churn-data/
```

### Job Fails

```bash
# Get job details
kubectl describe job -n processing <job-name>

# Check pod logs
kubectl logs -n processing <pod-name>

# Get pod events
kubectl get events -n processing --sort-by='.lastTimestamp'
```

## Cleanup

Remove all resources:

```bash
./kubernetes/scripts/cleanup.sh
```

Remove only the cluster (keep operator):

```bash
helm uninstall preprocessing-cluster -n processing
```

## Best Practices

1. **Always use Helm for cluster management** - Don't hardcode K8s configs in Python
2. **Separate cluster lifecycle from jobs** - Cluster is persistent, jobs are ephemeral
3. **Use environment-specific values** - Different resources for dev/staging/prod
4. **Monitor resource usage** - Adjust requests/limits based on actual usage
5. **Use IAM roles** - Never hardcode AWS credentials
6. **Enable TTL for jobs** - Auto-cleanup completed jobs after 5 minutes
7. **Use specific image tags** - Avoid `:latest` in production

## Next Steps

- [ ] Set up monitoring with Prometheus/Grafana
- [ ] Configure autoscaling for workers
- [ ] Add alerting for failed jobs
- [ ] Implement CI/CD pipeline
- [ ] Add cost monitoring and optimization
