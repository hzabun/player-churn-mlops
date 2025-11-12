resource "aws_cloudwatch_log_group" "fargate_logs" {
  name              = "/aws/eks/${aws_eks_cluster.player_churn_prediction.name}/fargate"
  retention_in_days = 7

  tags = local.tags
}
