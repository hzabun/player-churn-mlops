resource "aws_s3_bucket" "player_churn_bucket" {
  bucket = var.bucket_name

  tags = local.tags
}
