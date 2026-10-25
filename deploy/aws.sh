#!/usr/bin/env bash
# Build and deploy WoundTrace to AWS Lambda (arm64 / Graviton) with a public Function URL.
#
#   ./deploy/aws.sh            build the package, create or update everything, print the URL
#   ./deploy/aws.sh seed       also (re)create the two demo patients in S3
#
# Needs AWS CLI v2 credentials (aws login). Idempotent: safe to run again.
# Creates: one S3 bucket (private), one IAM role, one Lambda function + URL, a log group.
set -euo pipefail
cd "$(dirname "$0")/.."
unset PYTHONPATH

REGION=${AWS_REGION:-us-east-1}
export AWS_REGION=$REGION AWS_DEFAULT_REGION=$REGION
NAME=woundtrace
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
BUCKET=${BUCKET:-$NAME-$ACCOUNT}
ROLE=$NAME-lambda
MODEL=${BEDROCK_MODEL:-us.amazon.nova-2-lite-v1:0}
BUILD=build/lambda

echo "== bucket s3://$BUCKET"
if ! aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  aws s3api create-bucket --bucket "$BUCKET" >/dev/null
  aws s3api put-public-access-block --bucket "$BUCKET" --public-access-block-configuration \
    BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
fi

echo "== package (linux arm64, python 3.12)"
rm -rf "$BUILD" && mkdir -p "$BUILD"
# boto3 is already in the Lambda runtime; leave it out to keep the package small.
grep -vE '^(boto3|botocore)' requirements.txt > build/requirements-lambda.txt
./venv/bin/pip install -q --platform manylinux2014_aarch64 --platform manylinux_2_28_aarch64 \
  --only-binary=:all: --python-version 3.12 --implementation cp --target "$BUILD" \
  -r build/requirements-lambda.txt
cp -r app web "$BUILD/"
rm -rf "$BUILD"/app/__pycache__ "$BUILD"/*.dist-info/RECORD
cat > "$BUILD/lambda_function.py" <<'PY'
import cv2
from mangum import Mangum

from app.main import app

cv2.setNumThreads(2)
handler = Mangum(app, lifespan="off")
PY
(cd "$BUILD" && rm -f ../woundtrace.zip && zip -qr9 ../woundtrace.zip .)
du -sh "$BUILD" build/woundtrace.zip
aws s3 cp --quiet build/woundtrace.zip "s3://$BUCKET/deploy/woundtrace.zip"

echo "== IAM role $ROLE"
if ! aws iam get-role --role-name "$ROLE" >/dev/null 2>&1; then
  aws iam create-role --role-name "$ROLE" --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{"Effect": "Allow", "Principal": {"Service": "lambda.amazonaws.com"},
                   "Action": "sts:AssumeRole",
                   "Condition": {"StringEquals": {"aws:SourceAccount": "'"$ACCOUNT"'"}}}]}' >/dev/null
  NEW_ROLE=1
fi
# Least privilege: its own log group, its own bucket's data prefixes, one model family.
aws iam put-role-policy --role-name "$ROLE" --policy-name "$NAME" --policy-document '{
  "Version": "2012-10-17",
  "Statement": [
    {"Effect": "Allow", "Action": ["logs:CreateLogStream", "logs:PutLogEvents"],
     "Resource": "arn:aws:logs:'"$REGION:$ACCOUNT"':log-group:/aws/lambda/'"$NAME"':*"},
    {"Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject"],
     "Resource": ["arn:aws:s3:::'"$BUCKET"'/photos/*", "arn:aws:s3:::'"$BUCKET"'/annotated/*",
                  "arn:aws:s3:::'"$BUCKET"'/patients/*", "arn:aws:s3:::'"$BUCKET"'/alerts/*",
                  "arn:aws:s3:::'"$BUCKET"'/runs/*", "arn:aws:s3:::'"$BUCKET"'/usage/*"]},
    {"Effect": "Allow", "Action": "s3:ListBucket", "Resource": "arn:aws:s3:::'"$BUCKET"'"},
    {"Effect": "Allow", "Action": ["bedrock:InvokeModel"],
     "Resource": ["arn:aws:bedrock:*::foundation-model/amazon.nova-2-lite-v1:0",
                  "arn:aws:bedrock:*:'"$ACCOUNT"':inference-profile/us.amazon.nova-2-lite-v1:0"]}
  ]}'
ROLE_ARN=$(aws iam get-role --role-name "$ROLE" --query Role.Arn --output text)
[ -n "${NEW_ROLE:-}" ] && sleep 10  # new roles take a moment to become assumable

echo "== log group (14-day retention)"
aws logs create-log-group --log-group-name "/aws/lambda/$NAME" 2>/dev/null || true
aws logs put-retention-policy --log-group-name "/aws/lambda/$NAME" --retention-in-days 14

ENV="Variables={BUCKET=$BUCKET,BEDROCK_MODEL=$MODEL,AGENT_ENGINE=bedrock,DAILY_LLM_RUNS=${DAILY_LLM_RUNS:-150}}"
echo "== function $NAME"
if aws lambda get-function --function-name "$NAME" >/dev/null 2>&1; then
  aws lambda update-function-code --function-name "$NAME" --s3-bucket "$BUCKET" \
    --s3-key deploy/woundtrace.zip --architectures arm64 >/dev/null
  aws lambda wait function-updated --function-name "$NAME"
  aws lambda update-function-configuration --function-name "$NAME" --environment "$ENV" \
    --memory-size 2048 --timeout 60 >/dev/null
else
  aws lambda create-function --function-name "$NAME" --runtime python3.12 --architectures arm64 \
    --role "$ROLE_ARN" --handler lambda_function.handler --memory-size 2048 --timeout 60 \
    --code "S3Bucket=$BUCKET,S3Key=deploy/woundtrace.zip" --environment "$ENV" >/dev/null
fi
aws lambda wait function-updated --function-name "$NAME"

if ! aws lambda get-function-url-config --function-name "$NAME" >/dev/null 2>&1; then
  aws lambda create-function-url-config --function-name "$NAME" --auth-type NONE >/dev/null
  aws lambda add-permission --function-name "$NAME" --statement-id public-url \
    --action lambda:InvokeFunctionUrl --principal "*" --function-url-auth-type NONE >/dev/null
  aws lambda add-permission --function-name "$NAME" --statement-id public-url-invoke \
    --action lambda:InvokeFunction --principal "*" --invoked-via-function-url >/dev/null 2>&1 || true
fi
URL=$(aws lambda get-function-url-config --function-name "$NAME" --query FunctionUrl --output text)

if [ "${1:-}" = seed ]; then
  echo "== demo patients"
  BUCKET=$BUCKET ./venv/bin/python -m scripts.demo_data seed
fi

echo "== smoke test"
for i in 1 2 3 4 5; do curl -sf "${URL}api/health" | head -c 200 && break; sleep 3; done; echo
echo "live: $URL"
