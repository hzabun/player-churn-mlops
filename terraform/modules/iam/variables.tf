variable "cluster_name" {
  description = "Name of the EKS cluster"
  type        = string
}

variable "oidc_thumbprint" {
  description = "Thumbprint of the OIDC provider certificate"
  type        = string
}

variable "oidc_issuer_url" {
  description = "URL of the OIDC issuer"
  type        = string
}

variable "fargate_log_group_arn" {
  description = "ARN of the CloudWatch log group for Fargate logs"
  type        = string
}

variable "s3_bucket_arn" {
  description = "ARN of the S3 bucket for data storage"
  type        = string
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}
