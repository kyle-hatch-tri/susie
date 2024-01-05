# Build the docker image
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com
# make docker_build



# Upload the docker image

account=$(aws sts get-caller-identity --query Account --output text)

# Ensure this is 124224456861; if not, see FAQ below.
echo "account: $account"

algorithm_name=susie

# Currently only us-east-1 is available to us
region=us-east-1

# Full name for the docker image
fullname="${account}.dkr.ecr.${region}.amazonaws.com/${algorithm_name}:latest"
echo "fullname: $fullname"

aws ecr get-login-password --region ${region} | docker login --username AWS --password-stdin ${fullname}

# # Create the registry
# # Note: If you get a permission error here, you probably are using your
# # individual account, instead of the poweruser ML-R account; see FAQ below.
# aws ecr create-repository --repository-name "${algorithm_name}" > /dev/null

docker tag ${algorithm_name} ${fullname}
docker push ${fullname}

