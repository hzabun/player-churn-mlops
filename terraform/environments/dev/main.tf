terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source = "hashicorp/aws"
      # version = "~> 5.0"
    }
    tls = {
      source = "hashicorp/tls"
      # version = "~> 4.0"
    }
  }

  backend "s3" {
    bucket = "player-churn-tfstate-dev"
    key    = "terraform.tfstate"
    region = "us-east-1"
    # dynamodb_table = "terraform-state-lock"
    encrypt = true
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Environment = var.environment
      Project     = "player-churn-mlops"
      ManagedBy   = "terraform"
    }
  }
}

# Data sources
data "aws_caller_identity" "current" {}

data "tls_certificate" "eks_oidc" {
  url = module.eks.oidc_issuer_url
}

# Local values
locals {
  cluster_name = "player-churn-${var.environment}"

  tags = {
    Environment = var.environment
    Project     = "player-churn-mlops"
  }
}

# VPC Module
module "vpc" {
  source = "../../modules/vpc"

  vpc_name             = "${local.cluster_name}-vpc"
  vpc_cidr             = var.vpc_cidr
  availability_zones   = var.availability_zones
  private_subnet_cidrs = var.private_subnet_cidrs
  public_subnet_cidrs  = var.public_subnet_cidrs
  enable_nat_gateway   = var.enable_nat_gateway

  tags = local.tags
}

# S3 Module
module "s3" {
  source = "../../modules/s3"

  bucket_name       = var.s3_bucket_name
  enable_versioning = var.enable_s3_versioning

  tags = local.tags
}

# ECR Module
module "ecr" {
  source = "../../modules/ecr"

  repository_names = var.ecr_repository_names

  tags = local.tags
}

# EKS Module
module "eks" {
  source = "../../modules/eks-cluster"

  cluster_name     = local.cluster_name
  cluster_version  = var.eks_cluster_version
  cluster_role_arn = module.iam.eks_cluster_role_arn

  subnet_ids                     = module.vpc.private_subnets
  endpoint_public_access         = var.eks_endpoint_public_access
  endpoint_private_access        = var.eks_endpoint_private_access
  public_access_cidrs            = var.eks_public_access_cidrs
  fargate_pod_execution_role_arn = module.iam.fargate_pod_execution_role_arn
  log_retention_days             = var.log_retention_days

  tags = local.tags
}

# IAM Module
module "iam" {
  source = "../../modules/iam"

  cluster_name          = local.cluster_name
  oidc_thumbprint       = data.tls_certificate.eks_oidc.certificates[0].sha1_fingerprint
  oidc_issuer_url       = module.eks.oidc_issuer_url
  fargate_log_group_arn = module.eks.fargate_log_group_arn
  s3_bucket_arn         = module.s3.bucket_arn

  tags = local.tags
}
