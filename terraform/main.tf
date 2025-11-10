provider "aws" {
  region = "us-east-1"
}

resource "aws_s3_bucket" "player_churn_bucket" {
  bucket = var.bucket_name

  tags = local.tags
}

module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "6.5.0"

  name = "main-mlops-vpc"
  cidr = "10.0.0.0/24"

  azs             = ["us-east-1a", "us-east-1b"]
  private_subnets = ["10.0.0.0/26", "10.0.0.64/26"]
  public_subnets  = ["10.0.0.128/25"]


  enable_nat_gateway = true

  tags = local.tags
}

resource "aws_ecr_repository" "player_churn_preprocess_repo" {
  name                 = "player-churn-preprocess"
  image_tag_mutability = "IMMUTABLE_WITH_EXCLUSION"

  image_tag_mutability_exclusion_filter {
    filter      = "latest*"
    filter_type = "WILDCARD"
  }

  image_tag_mutability_exclusion_filter {
    filter      = "dev-*"
    filter_type = "WILDCARD"
  }
}

resource "aws_ecr_repository" "player_churn_train_repo" {
  name                 = "player-churn-train"
  image_tag_mutability = "IMMUTABLE_WITH_EXCLUSION"

  image_tag_mutability_exclusion_filter {
    filter      = "latest*"
    filter_type = "WILDCARD"
  }

  image_tag_mutability_exclusion_filter {
    filter      = "dev-*"
    filter_type = "WILDCARD"
  }
}

resource "aws_eks_cluster" "player_churn_prediction" {
  name    = "player-churn-prediction"
  version = "1.33"

  role_arn = aws_iam_role.eks_cluster_role.arn

  bootstrap_self_managed_addons = false

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

  # Prevent recreation on certain attribute changes
  lifecycle {
    ignore_changes = [
      access_config[0].bootstrap_cluster_creator_admin_permissions,
    ]
  }
}
