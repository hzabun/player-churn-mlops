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
