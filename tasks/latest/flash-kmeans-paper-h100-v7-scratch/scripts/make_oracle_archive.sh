#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Compatibility entry point. In v7-scratch "oracle" means the official repository
# implementation, not the former clean-room implementation.
exec "${SCRIPT_DIR}/make_host_calibration_archives.sh"
