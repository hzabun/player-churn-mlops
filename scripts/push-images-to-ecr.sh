#!/bin/bash
# Source environment variables
source "$(dirname "$0")/ecr_env.sh"

aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$ECR_URL"

docker build -t "$ECR_REPO_PREPROCESS" -f "$DOCKERFILE_PREPROCESS" .
docker build -t "$ECR_REPO_TRAIN" -f "$DOCKERFILE_TRAIN" .

docker tag "$ECR_REPO_PREPROCESS:latest" "$ECR_URL/$ECR_REPO_PREPROCESS:latest"
docker tag "$ECR_REPO_TRAIN:latest" "$ECR_URL/$ECR_REPO_TRAIN:latest"

docker push "$ECR_URL/$ECR_REPO_PREPROCESS:latest"
docker push "$ECR_URL/$ECR_REPO_TRAIN:latest"
