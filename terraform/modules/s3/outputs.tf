output "bucket_id" {
  description = "The name of the bucket"
  value       = aws_s3_bucket.this.id
}

output "bucket_arn" {
  description = "The ARN of the bucket"
  value       = aws_s3_bucket.this.arn
}

output "bucket_region" {
  description = "The AWS region the bucket resides in"
  value       = aws_s3_bucket.this.region
}
