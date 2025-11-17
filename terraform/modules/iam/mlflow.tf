# IAM Role for MLflow Service Account
resource "aws_iam_role" "mlflow_sa" {
  name = "${var.cluster_name}-mlflow-sa-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Federated = aws_iam_openid_connect_provider.eks_oidc.arn
        }
        Action = "sts:AssumeRoleWithWebIdentity"
        Condition = {
          StringEquals = {
            "${replace(var.oidc_issuer_url, "https://", "")}:sub" : "system:serviceaccount:mlflow:mlflow-sa"
            "${replace(var.oidc_issuer_url, "https://", "")}:aud" : "sts.amazonaws.com"
          }
        }
      }
    ]
  })

  tags = merge(
    var.tags,
    {
      Name = "${var.cluster_name}-mlflow-sa-role"
    }
  )
}

# IAM Policy for MLflow S3 Access
resource "aws_iam_policy" "mlflow_s3" {
  name        = "${var.cluster_name}-mlflow-s3-policy"
  description = "Policy for MLflow to access S3 for artifacts and backend storage"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:ListBucket",
          "s3:GetBucketLocation"
        ]
        Resource = var.s3_bucket_arn
      },
      {
        Effect = "Allow"
        Action = [
          "s3:PutObject",
          "s3:GetObject",
          "s3:DeleteObject",
          "s3:ListMultipartUploadParts",
          "s3:AbortMultipartUpload"
        ]
        Resource = "${var.s3_bucket_arn}/mlflow/*"
      }
    ]
  })

  tags = var.tags
}

# Attach policy to MLflow role
resource "aws_iam_role_policy_attachment" "mlflow_s3" {
  role       = aws_iam_role.mlflow_sa.name
  policy_arn = aws_iam_policy.mlflow_s3.arn
}

# Add policy for Secrets Manager access
resource "aws_iam_policy" "mlflow_secrets_access" {
  name        = "${var.cluster_name}-mlflow-secrets-access"
  description = "Allow MLflow to read RDS credentials from Secrets Manager"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue",
          "secretsmanager:DescribeSecret"
        ]
        Resource = var.rds_secret_arn
      }
    ]
  })

  tags = var.tags
}

# Attach secrets policy to MLflow role
resource "aws_iam_role_policy_attachment" "mlflow_secrets_access" {
  role       = aws_iam_role.mlflow_sa.name
  policy_arn = aws_iam_policy.mlflow_secrets_access.arn
}
