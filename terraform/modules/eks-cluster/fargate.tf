# Fargate profile for kube-system namespace
resource "aws_eks_fargate_profile" "kube_system" {
  cluster_name           = aws_eks_cluster.this.name
  fargate_profile_name   = "kube-system"
  pod_execution_role_arn = var.fargate_pod_execution_role_arn
  subnet_ids             = var.subnet_ids

  selector {
    namespace = "kube-system"
    labels = {
      "k8s-app" = "kube-dns"
    }
  }

  tags = var.tags
}

# Fargate profile for default namespace
resource "aws_eks_fargate_profile" "default" {
  cluster_name           = aws_eks_cluster.this.name
  fargate_profile_name   = "default"
  pod_execution_role_arn = var.fargate_pod_execution_role_arn
  subnet_ids             = var.subnet_ids

  selector {
    namespace = "default"
  }

  tags = var.tags
}

# Fargate profile for dask-operator namespace
resource "aws_eks_fargate_profile" "dask_operator" {
  cluster_name           = aws_eks_cluster.this.name
  fargate_profile_name   = "dask-operator"
  pod_execution_role_arn = var.fargate_pod_execution_role_arn
  subnet_ids             = var.subnet_ids

  selector {
    namespace = "dask-operator"
  }

  tags = var.tags
}

# Fargate profile for processing namespace (Dask workers)
resource "aws_eks_fargate_profile" "processing" {
  cluster_name           = aws_eks_cluster.this.name
  fargate_profile_name   = "processing"
  pod_execution_role_arn = var.fargate_pod_execution_role_arn
  subnet_ids             = var.subnet_ids

  selector {
    namespace = "processing"
  }

  tags = var.tags
}
