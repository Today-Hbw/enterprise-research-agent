#!/usr/bin/env bash
set -Eeuo pipefail

base=/mnt/enterprise-research-agent
release_root="$base/docker/releases"
shared_root="$base/shared"
env_file="$shared_root/config/app.env"
state_root="$shared_root/docker"

release_dir="${1:?release directory is required}"
image="${2:?image tag is required}"

resolved_release=$(realpath "$release_dir")
case "$resolved_release" in
  "$release_root"/*) ;;
  *) echo "Refusing release outside $release_root" >&2; exit 2 ;;
esac

compose_file="$resolved_release/docker-compose.server.yml"
test -f "$compose_file"
test -f "$env_file"
mkdir -p "$state_root"

compose() {
  env \
    AGENT_IMAGE="$1" \
    APP_REVISION="${1##*:}" \
    APP_ENV_FILE="$env_file" \
    docker compose --project-name enterprise-research-agent --file "$compose_file" "${@:2}"
}

echo "Building $image"
compose "$image" build agent

echo "Validating production configuration inside the image"
docker run --rm \
  --network host \
  --env-file "$env_file" \
  --env PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
  --env LD_LIBRARY_PATH= \
  "$image" \
  python -c 'from app.main import runtime; print(runtime.provider.name); assert not runtime.provider.is_demo'

old_image=$(docker inspect --format '{{.Config.Image}}' enterprise-research-agent 2>/dev/null || true)
old_source=$(readlink -f "$state_root/current-source" 2>/dev/null || true)
legacy_was_active=0

if systemctl --user is-active --quiet enterprise-research-agent.service; then
  legacy_was_active=1
  echo "Stopping legacy systemd service"
  systemctl --user stop enterprise-research-agent.service
fi

rollback() {
  echo "Docker deployment failed; rolling back" >&2
  compose "$image" logs --no-color --tail 80 agent >&2 || true
  if test -n "$old_image"; then
    compose "$old_image" up -d --no-build --wait --wait-timeout 60 agent || true
  else
    compose "$image" down --remove-orphans || true
    if test "$legacy_was_active" -eq 1; then
      systemctl --user start enterprise-research-agent.service || true
    fi
  fi
}
trap rollback ERR

echo "Starting $image"
compose "$image" up -d --no-build --remove-orphans --wait --wait-timeout 90 agent
curl --fail --silent --show-error --max-time 10 http://127.0.0.1:8000/api/health >/dev/null

trap - ERR
if test -n "$old_source" && test "$old_source" != "$resolved_release"; then
  ln -sfn "$old_source" "$state_root/previous-source"
fi
ln -sfn "$resolved_release" "$state_root/current-source"
printf '%s\n' "$old_image" >"$state_root/previous-image"
printf '%s\n' "$image" >"$state_root/current-image"
systemctl --user disable enterprise-research-agent.service >/dev/null 2>&1 || true

echo "Deployment complete"
echo "image=$image"
echo "source=$resolved_release"
docker ps --filter name=^/enterprise-research-agent$ --format 'container={{.Names}} status={{.Status}} image={{.Image}}'
curl --fail --silent --show-error --max-time 10 http://127.0.0.1:8000/api/health
