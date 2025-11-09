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

  azs             = ["us-east-1a"]
  private_subnets = ["10.0.0.0/25"]
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

# module "eks" {
#   source  = "terraform-aws-modules/eks/aws"
#   version = "21.8.0"
# }
