output "eks_cluster_role_arn" {
  description = "ARN of the EKS cluster IAM role"
  value       = aws_iam_role.eks_cluster_role.arn
}

output "fargate_pod_execution_role_arn" {
  description = "ARN of the Fargate pod execution role"
  value       = aws_iam_role.fargate_pod_execution_role.arn
}

output "preprocess_sa_role_arn" {
  description = "ARN of the IAM role for preprocess service account"
  value       = aws_iam_role.preprocess_sa_role.arn
}

output "oidc_provider_arn" {
  description = "ARN of the OIDC Provider for EKS"
  value       = aws_iam_openid_connect_provider.eks_oidc.arn
}

output "oidc_provider_url" {
  description = "URL of the OIDC Provider for EKS"
  value       = aws_iam_openid_connect_provider.eks_oidc.url
}
