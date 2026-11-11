#!/usr/bin/env bash
# Fetch this task's Infrathon assets into ${INFRATHON_ASSETS_ROOT} and verify them
# (scripts/infrathon/assets.py; extra arguments are passed to it), then stage the
# SWE-bench corpus: the 192 digest-locked swebench/sweb.eval.x86_64.* images (about
# 230 GB of Docker image store; images already local are reused) and manifest.json under
# KVBENCH_CORPUS_ROOT (scripts/stage_swebench_corpus.py).  INFRATHON_SKIP_CORPUS=1 skips
# the corpus; it can be staged later with that script.
set -euo pipefail
task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repository_root="$(cd "${task_root}/../../.." && pwd)"
python="${PYTHON_BIN:-${repository_root}/.venv/bin/python}"
"${python}" "${repository_root}/scripts/infrathon/assets.py" fetch "$(basename "${task_root}")" "$@"
if [[ "${INFRATHON_SKIP_CORPUS:-0}" == 1 ]]; then
  echo "INFRATHON_SKIP_CORPUS=1: SWE-bench corpus not staged (scripts/stage_swebench_corpus.py)"
  exit 0
fi
echo "Staging the SWE-bench corpus: 192 Docker images, about 230 GB"
exec "${python}" "${task_root}/scripts/stage_swebench_corpus.py"
