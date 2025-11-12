# VPC Outputs
output "vpc_id" {
  description = "The ID of the VPC"
  value       = module.vpc.vpc_id
}

output "private_subnet_ids" {
  description = "List of private subnet IDs"
  value       = module.vpc.private_subnets
}

output "public_subnet_ids" {
  description = "List of public subnet IDs"
  value       = module.vpc.public_subnets
}

# EKS Cluster Outputs
output "eks_cluster_id" {
  description = "The name of the EKS cluster"
  value       = aws_eks_cluster.player_churn_prediction.id
}

output "eks_cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = aws_eks_cluster.player_churn_prediction.endpoint
}

output "eks_cluster_security_group_id" {
  description = "Security group ID attached to the EKS cluster"
  value       = aws_eks_cluster.player_churn_prediction.vpc_config[0].cluster_security_group_id
}

output "eks_cluster_arn" {
  description = "The ARN of the EKS cluster"
  value       = aws_eks_cluster.player_churn_prediction.arn
}

output "eks_oidc_provider_arn" {
  description = "ARN of the OIDC Provider for EKS"
  value       = aws_iam_openid_connect_provider.eks_oidc.arn
}

# ECR Outputs
output "ecr_preprocess_repository_url" {
  description = "URL of the ECR repository for preprocessing"
  value       = aws_ecr_repository.player_churn_preprocess_repo.repository_url
}

output "ecr_train_repository_url" {
  description = "URL of the ECR repository for training"
  value       = aws_ecr_repository.player_churn_train_repo.repository_url
}

# S3 Outputs
output "s3_bucket_name" {
  description = "The name of the S3 bucket"
  value       = aws_s3_bucket.player_churn_bucket.id
}

output "s3_bucket_arn" {
  description = "The ARN of the S3 bucket"
  value       = aws_s3_bucket.player_churn_bucket.arn
}

# CloudWatch Logs
output "fargate_log_group_name" {
  description = "CloudWatch log group name for Fargate logs"
  value       = aws_cloudwatch_log_group.fargate_logs.name
}

# Region
output "aws_region" {
  description = "AWS region"
  value       = "us-east-1"
}
