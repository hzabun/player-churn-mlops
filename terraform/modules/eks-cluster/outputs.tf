output "cluster_id" {
  description = "The name of the EKS cluster"
  value       = aws_eks_cluster.this.id
}

output "cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = aws_eks_cluster.this.endpoint
}

output "cluster_security_group_id" {
  description = "Security group ID attached to the EKS cluster"
  value       = aws_eks_cluster.this.vpc_config[0].cluster_security_group_id
}

output "cluster_arn" {
  description = "The ARN of the EKS cluster"
  value       = aws_eks_cluster.this.arn
}

output "cluster_certificate_authority_data" {
  description = "Base64 encoded certificate data required to communicate with the cluster"
  value       = aws_eks_cluster.this.certificate_authority[0].data
  sensitive   = true
}

output "oidc_issuer_url" {
  description = "The URL of the OpenID Connect identity provider"
  value       = aws_eks_cluster.this.identity[0].oidc[0].issuer
}

output "fargate_log_group_arn" {
  description = "ARN of the CloudWatch log group for Fargate logs"
  value       = aws_cloudwatch_log_group.fargate_logs.arn
}

output "fargate_log_group_name" {
  description = "Name of the CloudWatch log group for Fargate logs"
  value       = aws_cloudwatch_log_group.fargate_logs.name
}
