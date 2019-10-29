#!/bin/bash
#
# Start a local docker container.

set -euo pipefail

source scripts/shared.sh

parse_std_args "$@"

if [ "$arch" == 'gpu' ]; then
    docker_command='nvidia-docker'
else
    docker_command='docker'
fi


MODEL_DIR="$(cd "test/resources/models" > /dev/null && pwd)"
$docker_command run \
    -v "$MODEL_DIR":/opt/ml/model:ro \
    -e "SAGEMAKER_TFS_DEFAULT_MODEL_NAME=huiying" \
    -e "SAGEMAKER_TFS_NGINX_LOGLEVEL=error" \
    -e "SAGEMAKER_BIND_TO_PORT=8080" \
    -e "SAGEMAKER_SAFE_PORT_RANGE=9000-9999" \
    --rm -it \
    $repository:$full_version-$device /bin/bash 
