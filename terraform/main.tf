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

  name = "my-vpc"
  cidr = "10.0.0.0/24"

  azs             = ["us-east-1a"]
  private_subnets = ["10.0.0.0/25"]
  public_subnets  = ["10.0.0.128/25"]


  enable_nat_gateway = true

  tags = local.tags
}

module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "21.8.0"
}
