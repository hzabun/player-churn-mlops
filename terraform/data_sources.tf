data "aws_caller_identity" "current" {}

data "tls_certificate" "eks_oidc" {
  url = aws_eks_cluster.player_churn_prediction.identity[0].oidc[0].issuer
}
