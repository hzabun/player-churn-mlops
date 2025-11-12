resource "aws_eks_cluster" "player_churn_prediction" {
  name    = "player-churn-prediction"
  version = "1.33"

  role_arn = aws_iam_role.eks_cluster_role.arn

  bootstrap_self_managed_addons = false

  upgrade_policy {
    support_type = "STANDARD"
  }

  vpc_config {
    subnet_ids              = module.vpc.private_subnets
    endpoint_public_access  = true
    endpoint_private_access = true
    public_access_cidrs     = ["0.0.0.0/0"]
  }

  tags = local.tags

  zonal_shift_config {
    enabled = false
  }

  depends_on = [module.vpc]
}

resource "aws_eks_addon" "coredns" {
  cluster_name = aws_eks_cluster.player_churn_prediction.name
  addon_name   = "coredns"
  # addon_version               = "v1.11.3-eksbuild.2"
  resolve_conflicts_on_update = "OVERWRITE"

  configuration_values = jsonencode({
    computeType = "Fargate"
  })

  depends_on = [
    aws_eks_fargate_profile.kube_system
  ]

  tags = local.tags
}
