#!/usr/bin/env bash
set -euo pipefail

# Cooperative, Work-side cleanup before an interactive submission.
#
# This script can stop same-UID processes which currently hold NVIDIA device
# file descriptors.  It is intentionally not a lease or idle attestation:
# only the host coordinator can pause the exact Work container, rule out a
# post-check reacquire, and attest the scheduler allocation before Judge runs.

PROC_ROOT="${SFORGE_GPU_PROC_ROOT:-/proc}"
TERM_WAIT_SECONDS="${SFORGE_GPU_DRAIN_TERM_WAIT_SECONDS:-10}"

case "${TERM_WAIT_SECONDS}" in
  ''|*[!0-9]*)
    echo "ERROR: SFORGE_GPU_DRAIN_TERM_WAIT_SECONDS must be a non-negative integer" >&2
    exit 64
    ;;
esac

declare -A protected=()
cursor="$$"
while [[ "${cursor}" =~ ^[0-9]+$ && "${cursor}" != "0" ]]; do
  protected["${cursor}"]=1
  status="${PROC_ROOT}/${cursor}/status"
  [[ -r "${status}" ]] || break
  parent="$(awk '$1 == "PPid:" {print $2; exit}' "${status}" 2>/dev/null || true)"
  [[ "${parent}" =~ ^[0-9]+$ && "${parent}" != "${cursor}" ]] || break
  cursor="${parent}"
done

gpu_holders() {
  local process_dir pid status owner fd target
  local current_uid
  current_uid="$(id -u)"
  for process_dir in "${PROC_ROOT}"/[0-9]*; do
    [[ -d "${process_dir}" ]] || continue
    pid="${process_dir##*/}"
    [[ -z "${protected[${pid}]:-}" ]] || continue
    status="${process_dir}/status"
    [[ -r "${status}" ]] || continue
    owner="$(awk '$1 == "Uid:" {print $2; exit}' "${status}" 2>/dev/null || true)"
    [[ "${owner}" == "${current_uid}" ]] || continue
    for fd in "${process_dir}"/fd/*; do
      [[ -e "${fd}" || -L "${fd}" ]] || continue
      target="$(readlink "${fd}" 2>/dev/null || true)"
      case "${target}" in
        /dev/nvidia[0-9]*|/dev/nvidiactl|/dev/nvidia-uvm|/dev/nvidia-uvm-tools|/dev/nvidia-caps/*)
          printf '%s\n' "${pid}"
          break
          ;;
      esac
    done
  done | sort -nu
}

mapfile -t holders < <(gpu_holders)
if (( ${#holders[@]} )); then
  echo "Cooperative GPU drain: asking same-UID holders to exit: ${holders[*]}" >&2
  kill -TERM -- "${holders[@]}" 2>/dev/null || true
  deadline="$((SECONDS + TERM_WAIT_SECONDS))"
  while (( SECONDS < deadline )); do
    mapfile -t holders < <(gpu_holders)
    (( ${#holders[@]} )) || break
    sleep 1
  done
fi

mapfile -t holders < <(gpu_holders)
if (( ${#holders[@]} )); then
  echo "Cooperative GPU drain: force-stopping remaining holders: ${holders[*]}" >&2
  kill -KILL -- "${holders[@]}" 2>/dev/null || true
  sleep 1
fi

sync
mapfile -t holders < <(gpu_holders)
if (( ${#holders[@]} )); then
  echo "ERROR: Work still has same-UID NVIDIA device holders: ${holders[*]}" >&2
  echo "Do not submit; stop the listed public-test/compiler/worker processes and retry." >&2
  exit 75
fi

echo "Cooperative Work GPU drain complete."
echo "This is not host idle attestation; the handoff coordinator must still pause Work and attest the exact allocation."
