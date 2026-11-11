#!/usr/bin/env bash
set -euo pipefail

# Build one self-contained agent runtime directory for SFORGE_AGENT_RUNTIME_DIR.
#
# A runtime is a host directory holding Node plus exactly one agent CLI under
# a single prefix, with a marker and an env.sh.  The docker backend mounts it
# read-only at /opt/agent-runtime inside work containers, so task images stay
# agent-free and the agent kind/version is chosen per launch instead of per
# image build.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

NODE_VERSION_DEFAULT="v20.18.0"
MOUNT_TARGET="/opt/agent-runtime"

usage() {
  cat <<'USAGE'
Usage:
  build-agent-runtime.sh --agent codex|claude-code [options]

Options:
  --agent NAME      SForge agent name: codex or claude-code (required)
  --version V       CLI version (default: launch/common.sh Codex pin for
                    codex; required for claude-code)
  --node vX.Y.Z     Node version to bundle (default: v20.18.0)
  --root DIR        Runtime store root
                    (default: <repo>/../agent_runtimes)
  --force           Replace an existing runtime directory

The result is <root>/<agent>-<version>/ with bin/node, bin/npm, the agent
CLI, env.sh, and .sforge-agent-runtime.json (including a tree SHA-256 for
run manifests).  Downloads honor the ambient proxy environment.
USAGE
}

agent=""
version=""
node_version="${NODE_VERSION_DEFAULT}"
root=""
force=0
while (( $# )); do
  case "$1" in
    --agent)   [[ $# -ge 2 ]] || launch_die "--agent requires a value"; agent="$2"; shift 2 ;;
    --version) [[ $# -ge 2 ]] || launch_die "--version requires a value"; version="$2"; shift 2 ;;
    --node)    [[ $# -ge 2 ]] || launch_die "--node requires a value"; node_version="$2"; shift 2 ;;
    --root)    [[ $# -ge 2 ]] || launch_die "--root requires a value"; root="$2"; shift 2 ;;
    --force)   force=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *)         usage >&2; launch_die "unknown argument: $1" ;;
  esac
done

case "${agent}" in
  codex)
    package="@openai/codex"
    cli="codex"
    [[ -n "${version}" ]] || version="${CODEX_CLI_VERSION_PIN}"
    ;;
  claude-code)
    package="@anthropic-ai/claude-code"
    cli="claude"
    [[ -n "${version}" ]] || launch_die "--version is required for claude-code"
    ;;
  *)
    launch_die "unsupported --agent: ${agent:-<missing>} (use codex or claude-code)"
    ;;
esac
[[ "${version}" =~ ^[0-9A-Za-z][0-9A-Za-z._-]*$ ]] || launch_die "unsafe version: ${version}"
[[ "${node_version}" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]] || launch_die "unsafe node version: ${node_version}"

REPO_ROOT="$(launch_repo_root)"
[[ -n "${root}" ]] || root="$(cd -- "${REPO_ROOT}/.." && pwd)/agent_runtimes"
runtime_dir="${root}/${agent}-${version}"
if [[ -e "${runtime_dir}" ]]; then
  (( force )) || launch_die "runtime already exists (use --force): ${runtime_dir}"
  chmod -R u+w -- "${runtime_dir}"
  rm -rf -- "${runtime_dir}"
fi
mkdir -p "${root}"
stage="$(mktemp -d "${root}/.build-${agent}-${version}.XXXXXX")"
trap 'rm -rf -- "${stage}"' EXIT

node_mirror="${SFORGE_NODEJS_MIRROR_URL:-https://nodejs.org/dist}"
echo "Fetching Node ${node_version} from ${node_mirror}"
curl -fsSL "${node_mirror}/${node_version}/node-${node_version}-linux-x64.tar.xz" \
  | tar -xJ -C "${stage}" --strip-components=1
"${stage}/bin/node" --version | grep -Fx "${node_version}" \
  || launch_die "bundled node reports an unexpected version"

echo "Installing ${package}@${version}"
PATH="${stage}/bin:${PATH}" npm_config_update_notifier=false \
  "${stage}/bin/npm" install -g --prefix "${stage}" "${package}@${version}"
PATH="${stage}/bin:${PATH}" "${stage}/bin/${cli}" --version | grep -F "${version}" \
  || launch_die "installed CLI does not report version ${version}"

cat >"${stage}/env.sh" <<EOF
# Sourced via BASH_ENV in SForge work containers.
export PATH="${MOUNT_TARGET}/bin:\${PATH}"
EOF

echo "Hashing runtime tree"
tree_sha256="$(
  cd "${stage}" \
    && find . -type f ! -name .sforge-agent-runtime.json -print0 \
      | LC_ALL=C sort -z \
      | xargs -0 sha256sum \
      | sha256sum \
      | awk '{print $1}'
)"

python3 - "${stage}" "${agent}" "${cli}" "${version}" "${node_version}" "${tree_sha256}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

stage, agent, cli, version, node_version, tree_sha256 = sys.argv[1:7]
marker = {
    "schema_version": 1,
    "kind": "sforge-agent-runtime",
    "agent": agent,
    "cli": cli,
    "version": version,
    "node_version": node_version,
    "tree_sha256": tree_sha256,
    "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}
Path(stage, ".sforge-agent-runtime.json").write_text(
    json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

# World-readable and immutable: the container-side agent user has a
# different uid from the host builder, and mktemp starts the stage at 0700.
chmod -R a+rX,a-w "${stage}"
chmod u+w "${stage}"
mv -- "${stage}" "${runtime_dir}"
trap - EXIT
echo "Built ${runtime_dir}"
echo "  agent: ${agent} (${cli} ${version}, node ${node_version})"
echo "  tree sha256: ${tree_sha256}"
echo "  use: SFORGE_AGENT_RUNTIME_DIR=${runtime_dir}"
