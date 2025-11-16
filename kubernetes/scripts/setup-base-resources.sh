AWS_ACCOUNT_ID=${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}
ENV=${1:-dev}
CLUSTER_NAME=${CLUSTER_NAME:-player-churn-${ENV}}

export AWS_ACCOUNT_ID CLUSTER_NAME

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
envsubst < "$SCRIPT_DIR/../base/namespace.yaml" | kubectl apply -f -
envsubst < "$SCRIPT_DIR/../base/service-account.yaml" | kubectl apply -f -
envsubst < "$SCRIPT_DIR/../base/aws_cloudwatch_fargate_setup.yaml" | kubectl apply -f -
