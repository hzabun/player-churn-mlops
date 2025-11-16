# IAM Role for training Service Account
resource "aws_iam_role" "training_sa" {
  name = "${var.cluster_name}-training-sa-role"

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
            "${replace(var.oidc_issuer_url, "https://", "")}:sub" : "system:serviceaccount:training:training-sa"
            "${replace(var.oidc_issuer_url, "https://", "")}:aud" : "sts.amazonaws.com"
          }
        }
      }
    ]
  })

  tags = merge(
    var.tags,
    {
      Name = "${var.cluster_name}-training-sa-role"
    }
  )
}

# IAM Policy for training S3 Access
resource "aws_iam_policy" "training_s3" {
  name        = "${var.cluster_name}-training-s3-policy"
  description = "Policy for feast training to access S3 for registry and offline store"

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
        Resource = "${var.s3_bucket_arn}/feature-store/*"
      }
    ]
  })

  tags = var.tags
}

# Attach policy to training role
resource "aws_iam_role_policy_attachment" "training_s3" {
  role       = aws_iam_role.training_sa.name
  policy_arn = aws_iam_policy.training_s3.arn
}
