resource "aws_eks_cluster" "this" {
  name    = var.cluster_name
  version = var.cluster_version

  role_arn = var.cluster_role_arn

  bootstrap_self_managed_addons = false

  upgrade_policy {
    support_type = "STANDARD"
  }

  vpc_config {
    subnet_ids              = var.subnet_ids
    endpoint_public_access  = var.endpoint_public_access
    endpoint_private_access = var.endpoint_private_access
    public_access_cidrs     = var.public_access_cidrs
  }

  tags = var.tags

  zonal_shift_config {
    enabled = false
  }
}

resource "aws_eks_addon" "coredns" {
  cluster_name                = aws_eks_cluster.this.name
  addon_name                  = "coredns"
  resolve_conflicts_on_update = "OVERWRITE"

  configuration_values = jsonencode({
    computeType = "Fargate"
  })

  depends_on = [
    aws_eks_fargate_profile.kube_system
  ]

  tags = var.tags
}
