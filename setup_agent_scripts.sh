#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_AGENT_SCRIPTS_DIR="/home/trana/Development/agent-scripts"
AGENT_SCRIPTS_DIR="${AGENT_SCRIPTS_DIR:-$DEFAULT_AGENT_SCRIPTS_DIR}"
LOCAL_BIN_DIR="${LOCAL_BIN_DIR:-$HOME/.local/bin}"

echo "Project root: ${PROJECT_ROOT}"
echo "Using agent-scripts dir: ${AGENT_SCRIPTS_DIR}"

if [[ ! -d "${AGENT_SCRIPTS_DIR}" ]]; then
  echo "agent-scripts directory not found: ${AGENT_SCRIPTS_DIR}" >&2
  echo "Set AGENT_SCRIPTS_DIR to an existing checkout and rerun." >&2
  exit 1
fi

if [[ -d "${AGENT_SCRIPTS_DIR}/.git" ]]; then
  echo "Refreshing agent-scripts (git pull --ff-only)..."
  git -C "${AGENT_SCRIPTS_DIR}" pull --ff-only || true
fi

mkdir -p "${LOCAL_BIN_DIR}"

if [[ -x "${AGENT_SCRIPTS_DIR}/bin/browser-tools" ]]; then
  ln -sfn "${AGENT_SCRIPTS_DIR}/bin/browser-tools" "${LOCAL_BIN_DIR}/browser-tools"
  echo "Linked browser-tools -> ${LOCAL_BIN_DIR}/browser-tools"
else
  echo "browser-tools not found at ${AGENT_SCRIPTS_DIR}/bin/browser-tools" >&2
fi

if [[ -x "${AGENT_SCRIPTS_DIR}/scripts/committer" ]]; then
  ln -sfn "${AGENT_SCRIPTS_DIR}/scripts/committer" "${LOCAL_BIN_DIR}/committer"
  echo "Linked committer -> ${LOCAL_BIN_DIR}/committer"
else
  echo "committer not found at ${AGENT_SCRIPTS_DIR}/scripts/committer" >&2
fi

if [[ -x "${AGENT_SCRIPTS_DIR}/scripts/trash.ts" ]]; then
  ln -sfn "${AGENT_SCRIPTS_DIR}/scripts/trash.ts" "${LOCAL_BIN_DIR}/trash.ts"
  echo "Linked trash.ts -> ${LOCAL_BIN_DIR}/trash.ts"
fi

echo
echo "Done."
echo "If needed, add this to your shell profile:"
echo "  export PATH=\"${LOCAL_BIN_DIR}:\$PATH\""

