#!/usr/bin/env bash
# Authenticate interactively with the SDK already installed in the training image.
set -euo pipefail

IMAGE="${1:-scicomp26:cu121-train-v1}"
docker info >/dev/null 2>&1 || {
  printf 'Cannot access Docker. Refresh your docker group with: newgrp docker\n' >&2
  exit 1
}
docker image inspect "$IMAGE" >/dev/null

umask 077
WANDB_NETRC_PATH="$HOME/.netrc"
touch -- "$WANDB_NETRC_PATH"
chmod 600 -- "$WANDB_NETRC_PATH"

docker run --rm -it --pull=never \
  --network host \
  --user "$(id -u):$(id -g)" \
  --env HOME=/tmp \
  --env NETRC=/tmp/.netrc \
  --env HTTP_PROXY --env HTTPS_PROXY --env ALL_PROXY --env NO_PROXY \
  --env http_proxy --env https_proxy --env all_proxy --env no_proxy \
  --workdir /tmp \
  --mount "type=bind,src=$WANDB_NETRC_PATH,dst=/tmp/.netrc" \
  "$IMAGE" wandb login --cloud --relogin --verify
