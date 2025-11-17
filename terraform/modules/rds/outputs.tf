output "cluster_endpoint" {
  description = "RDS instance endpoint"
  value       = aws_db_instance.postgres.address
}

output "cluster_reader_endpoint" {
  description = "RDS instance endpoint (same as cluster_endpoint for single instance)"
  value       = aws_db_instance.postgres.address
}

output "cluster_id" {
  description = "RDS instance identifier"
  value       = aws_db_instance.postgres.id
}

output "cluster_arn" {
  description = "RDS instance ARN"
  value       = aws_db_instance.postgres.arn
}

output "database_name" {
  description = "Name of the database"
  value       = aws_db_instance.postgres.db_name
}

output "master_username" {
  description = "Master username"
  value       = aws_db_instance.postgres.username
  sensitive   = true
}

output "security_group_id" {
  description = "Security group ID for RDS instance"
  value       = aws_security_group.postgres.id
}

output "secret_arn" {
  description = "ARN of Secrets Manager secret containing credentials"
  value       = aws_secretsmanager_secret.postgres_credentials.arn
}

output "connection_string" {
  description = "PostgreSQL connection string for MLflow"
  value       = "postgresql://${var.master_username}:${random_password.master_password.result}@${aws_db_instance.postgres.address}:5432/${var.database_name}"
  sensitive   = true
}
