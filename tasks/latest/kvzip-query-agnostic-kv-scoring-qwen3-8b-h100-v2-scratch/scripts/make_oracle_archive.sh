#!/usr/bin/env bash
# Writes artifacts/oracle.tar.gz for oracle/FINAL_VERSION.  ORACLE_VERSION=N packs
# candidate N instead (FINAL_VERSION N inside) as artifacts/oracle-vNNN.tar.gz.  A
# candidate with assets.ref (only to scorer_v001) ships with the scorer_v001 it names.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"; source "${SCRIPT_DIR}/lib.sh"
final_version="$(tr -d '[:space:]' < "${MIGRATION_ROOT}/oracle/FINAL_VERSION")"; version="${ORACLE_VERSION:-${final_version}}"; [[ "$version" =~ ^[0-9]{1,3}$ && "$final_version" =~ ^[0-9]{1,3}$ ]] || die "bad Oracle version"
printf -v selected_dir 'candidates/scorer_v%03d' "$((10#$version))"; selected="${selected_dir}/submission.py"; [[ -f "${MIGRATION_ROOT}/oracle/$selected" ]] || die "Oracle source missing"
asset_dir="$selected_dir"
if [[ -f "${MIGRATION_ROOT}/oracle/${selected_dir}/assets.ref" ]]; then
  asset_dir="candidates/$(tr -d '[:space:]' < "${MIGRATION_ROOT}/oracle/${selected_dir}/assets.ref")"
  [[ "$asset_dir" == candidates/scorer_v001 && -f "${MIGRATION_ROOT}/oracle/${asset_dir}/submission.py" ]] || die "Oracle assets.ref must name scorer_v001"
fi
oracle_asset_python="${ORACLE_ASSET_PYTHON:-${PYTHON_BIN}}"
[[ -f "${MIGRATION_ROOT}/oracle/${asset_dir}/assets.pt" ]] || "${oracle_asset_python}" "${MIGRATION_ROOT}/scripts/prepare_oracle_assets.py"
mkdir -p "$ARTIFACTS_DIR" "${MIGRATION_ROOT}/build"
archive="$ARTIFACTS_DIR/oracle.tar.gz"; [[ "$((10#$version))" == "$((10#$final_version))" ]] || printf -v archive '%s/oracle-v%03d.tar.gz' "$ARTIFACTS_DIR" "$((10#$version))"
temporary="$archive.tmp.$$"
staging="$(mktemp -d "${MIGRATION_ROOT}/build/oracle-archive.XXXXXX")"
trap 'rm -rf -- "$staging" "$temporary"' EXIT
mkdir -p "$staging/$selected_dir" "$staging/$asset_dir"
printf '%d\n' "$((10#$version))" > "$staging/FINAL_VERSION"; chmod 0644 "$staging/FINAL_VERSION"
install -m 0644 "${MIGRATION_ROOT}/oracle/notes.md" "$staging/"
install -m 0644 "${MIGRATION_ROOT}/oracle/$selected" "$staging/$selected"
if [[ "$asset_dir" != "$selected_dir" ]]; then
  install -m 0644 "${MIGRATION_ROOT}/oracle/${selected_dir}/assets.ref" "$staging/${selected_dir}/assets.ref"
  install -m 0644 "${MIGRATION_ROOT}/oracle/${asset_dir}/submission.py" "$staging/${asset_dir}/submission.py"
fi
"${oracle_asset_python}" "${MIGRATION_ROOT}/scripts/prepare_oracle_assets.py" \
  --transport-output "$staging/$asset_dir/assets.pt" >/dev/null
entries=(FINAL_VERSION notes.md "$selected_dir"); [[ "$asset_dir" == "$selected_dir" ]] || entries+=("$asset_dir")
tar -C "$staging" --sort=name --mtime='UTC 1970-01-01' --mode=u=rwX,go=rX --owner=0 --group=0 --numeric-owner -cf - "${entries[@]}" | gzip -9n > "$temporary"
[[ "$(stat -c %s "$temporary")" -le $((128 * 1024 * 1024)) ]] || die "Oracle transport archive exceeds SForge's 128 MiB compressed limit"
mv -- "$temporary" "$archive"; echo "$archive"; echo "sha256=$(sha256sum "$archive" | awk '{print $1}')" >&2
