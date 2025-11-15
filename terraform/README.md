# Terraform Project Structure

This directory contains Terraform infrastructure as code for the Player Churn MLOps project.

## Directory Structure

```
terraform/
├── environments/
│   └── dev/
│       ├── main.tf           # Main configuration
│       ├── variables.tf      # Variable definitions
│       ├── terraform.tfvars  # Variable values
│       └── outputs.tf        # Output values
│
├── modules/
│   ├── vpc/                  # VPC module
│   ├── eks-cluster/          # EKS cluster with Fargate
│   ├── iam/                  # IAM roles and policies
│   ├── ecr/                  # Container registries
│   └── s3/                   # S3 bucket for data
│
└── scripts/
    ├── init-env.sh           # Initialize Terraform
    ├── plan-env.sh           # Plan changes
    ├── apply-env.sh          # Apply changes
    └── destroy-env.sh        # Destroy infrastructure
```

## Prerequisites

1. **AWS CLI** configured with appropriate credentials
2. **Terraform** >= 1.5.0 installed
3. **S3 bucket** for Terraform state (update `backend` in `main.tf`)

## Quick Start

### 1. Update Configuration

Edit `environments/dev/terraform.tfvars`:

```hcl
s3_bucket_name = "your-unique-bucket-name"
```

### 2. Initialize Terraform

```bash
./scripts/init-env.sh dev
```

### 3. Plan Changes

```bash
./scripts/plan-env.sh dev
```

### 4. Apply Changes

```bash
./scripts/apply-env.sh dev
```

### 5. Configure kubectl

After deployment, run the command from the output:

```bash
aws eks update-kubeconfig --region us-east-1 --name player-churn-dev
```

## Resources Created

- **VPC**: Isolated network with public and private subnets
- **EKS Cluster**: Kubernetes cluster (v1.33) with Fargate
- **Fargate Profiles**: For kube-system, default, dask-operator, and processing namespaces
- **ECR Repositories**: player-churn/preprocess and player-churn/train
- **S3 Bucket**: For raw and processed data
- **IAM Roles**:
  - EKS cluster role
  - Fargate pod execution role
  - Service account role (IRSA) for S3 access
- **CloudWatch Log Group**: For Fargate container logs

## Environment Variables

You can override variables using environment variables:

```bash
export TF_VAR_eks_cluster_version="1.33"
export TF_VAR_log_retention_days=14
./scripts/plan-env.sh dev
```

## Adding New Environments

1. Copy `environments/dev` to `environments/prod`:

   ```bash
   cp -r environments/dev environments/prod
   ```

2. Update `terraform.tfvars` with production values

3. Update backend configuration in `main.tf`:

   ```hcl
   backend "s3" {
     bucket = "player-churn-tfstate-prod"
     key    = "terraform.tfstate"
     region = "us-east-1"
   }
   ```

4. Deploy:
   ```bash
   ./scripts/init-env.sh prod
   ./scripts/plan-env.sh prod
   ./scripts/apply-env.sh prod
   ```

## State Management

### Remote State (Recommended)

The backend is configured to use S3 for remote state storage. Make sure the S3 bucket exists:

```bash
aws s3 mb s3://player-churn-tfstate-dev --region us-east-1
```

### State Locking (Optional)

To enable state locking, create a DynamoDB table and uncomment the `dynamodb_table` line in `main.tf`:

```bash
aws dynamodb create-table \
  --table-name terraform-state-lock \
  --attribute-definitions AttributeName=LockID,AttributeType=S \
  --key-schema AttributeName=LockID,KeyType=HASH \
  --provisioned-throughput ReadCapacityUnits=5,WriteCapacityUnits=5
```

## Outputs

After applying, Terraform will output:

- VPC and subnet IDs
- EKS cluster endpoint and ARN
- ECR repository URLs
- S3 bucket name
- IAM role ARNs
- kubectl configuration command

View outputs anytime:

```bash
cd environments/dev
terraform output
```

## Troubleshooting

### EKS Cluster Not Accessible

```bash
# Update kubeconfig
aws eks update-kubeconfig --region us-east-1 --name player-churn-dev

# Verify connection
kubectl get nodes
```

### Fargate Pods Not Starting

```bash
# Check Fargate profiles
aws eks list-fargate-profiles --cluster-name player-churn-dev

# Check pod execution role
kubectl describe pod <pod-name> -n <namespace>
```

### State Lock Issues

```bash
# Force unlock (use with caution)
cd environments/dev
terraform force-unlock <lock-id>
```

## Cost Optimization

- **Dev Environment**: Uses smaller resources, 7-day log retention
- **Production**: Adjust instance types and log retention as needed
- **Clean up**: Run `./scripts/destroy-env.sh dev` when not in use

## Security Best Practices

1. ✅ S3 backend encryption enabled
2. ✅ S3 bucket public access blocked
3. ✅ IAM roles follow least privilege principle
4. ✅ EKS uses IRSA (IAM Roles for Service Accounts)
5. ✅ Private subnets for EKS workloads
6. ⚠️ **TODO**: Restrict EKS public access CIDRs in production

## Next Steps

- [ ] Set up multiple environments (staging, prod)
- [ ] Enable Terraform state locking with DynamoDB
- [ ] Configure VPN/bastion for private EKS access
- [ ] Add autoscaling for Fargate
- [ ] Implement monitoring and alerting
- [ ] Set up automated backups for S3
