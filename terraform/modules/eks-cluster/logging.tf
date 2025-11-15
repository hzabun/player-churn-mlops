resource "aws_cloudwatch_log_group" "fargate_logs" {
  name              = "/aws/eks/${var.cluster_name}/fargate"
  retention_in_days = var.log_retention_days

  tags = var.tags
}
