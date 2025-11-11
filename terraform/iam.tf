resource "aws_iam_role" "eks_cluster_role" {
  name        = "AmazonEKSClusterRole"
  description = "Allows access to other AWS service resources that are required to operate clusters managed by EKS."

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "eks.amazonaws.com"
        }
        Action = [
          "sts:AssumeRole",
          "sts:TagSession"
        ]
      }
    ]
  })

  tags = local.tags
}

resource "aws_iam_role_policy_attachment" "eks_cluster_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
  role       = aws_iam_role.eks_cluster_role.name
}

data "aws_caller_identity" "current" {}

data "tls_certificate" "eks_oidc" {
  url = aws_eks_cluster.player_churn_prediction.identity[0].oidc[0].issuer
}

resource "aws_iam_openid_connect_provider" "eks_oidc" {
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = [data.tls_certificate.eks_oidc.certificates[0].sha1_fingerprint]
  url             = aws_eks_cluster.player_churn_prediction.identity[0].oidc[0].issuer

  tags = local.tags
}
