resource "aws_eks_fargate_profile" "kube_system" {
  cluster_name           = aws_eks_cluster.player_churn_prediction.name
  fargate_profile_name   = "kube-system"
  pod_execution_role_arn = aws_iam_role.fargate_pod_execution_role.arn
  subnet_ids             = module.vpc.private_subnets

  selector {
    namespace = "kube-system"
    labels = {
      "k8s-app" = "kube-dns"
    }
  }

  tags = local.tags
}

# Fargate profile for default namespace
resource "aws_eks_fargate_profile" "default" {
  cluster_name           = aws_eks_cluster.player_churn_prediction.name
  fargate_profile_name   = "default"
  pod_execution_role_arn = aws_iam_role.fargate_pod_execution_role.arn
  subnet_ids             = module.vpc.private_subnets

  selector {
    namespace = "default"
  }

  tags = local.tags
}

# Create aws-observability namespace
resource "kubernetes_namespace" "aws_observability" {
  metadata {
    name = "aws-observability"
    labels = {
      "aws-observability" = "enabled"
    }
  }
}

# Configure Fluent Bit to send logs to CloudWatch
resource "kubernetes_manifest" "fargate_logging" {
  manifest = yamldecode(templatefile(
    "${path.module}/fargate-logging-configmap.yaml",
    {
      log_group_name     = aws_cloudwatch_log_group.fargate_logs.name
      log_retention_days = aws_cloudwatch_log_group.fargate_logs.retention_in_days
    }
  ))

  depends_on = [
    kubernetes_namespace.aws_observability
  ]
}

# IAM Role for Fargate Pod Execution used to pull container images and write logs to CloudWatch
resource "aws_iam_role" "fargate_pod_execution_role" {
  name        = "fargate-pod-execution-role"
  description = "Fargate pod execution role for EKS cluster"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "eks-fargate-pods.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = local.tags
}

# CloudWatch Logs acces for Fargate pod execution role
resource "aws_iam_role_policy" "fargate_cloudwatch_logs" {
  name = "fargate-cloudwatch-logs"
  role = aws_iam_role.fargate_pod_execution_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogStream",
          "logs:CreateLogGroup",
          "logs:DescribeLogStreams",
          "logs:PutLogEvents"
        ]
        Resource = "${aws_cloudwatch_log_group.fargate_logs.arn}:*"
      }
    ]
  })
}

# ECR access policy for Fargate pod execution role
resource "aws_iam_role_policy" "fargate_ecr_access" {
  name = "fargate-ecr-access"
  role = aws_iam_role.fargate_pod_execution_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ]
        Resource = "*"
      }
    ]
  })
}
