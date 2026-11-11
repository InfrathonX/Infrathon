"""Script generators for agent execution.

Produces bash scripts that run inside the work container:
- sforge-submit: submits current code to judge server via HTTP, prints results,
  updates shared state file (display-only cache)

Agent-specific scripts (stop hooks, settings files) live in the corresponding
agent modules under ``sforge.harness.agent``.

State file `/tmp/sforge_state.json` (display-only, not used for final scoring):
  {
    "best_pass_rate":  float,
    "best_score":      float|null,
    "best_round":      str,   # "agent-N"; Auto history is not Agent-visible
    "submissions": [
      {"kind": "agent", "round": str, "at": int, "pass_rate": float, "score": float|null}
    ]
  }
"""

from __future__ import annotations


def generate_submit_script() -> str:
    """Generate the sforge-submit bash script.

    Submits via POST to the token-based endpoint and polls for results.
    The server manages task_id, run_id, and round counters.

    Flags:
      --details, -d    submit and show detailed per-test results
      --version ID    evaluate one exact saved version without repacking
      --list, -l       list previous manual submissions (no new submission)
    """
    return _generate_submit_script()


def generate_save_version_script() -> str:
    """Generate the non-evaluating, token-authenticated candidate-save command."""

    return r'''#!/bin/bash
set -euo pipefail

usage() {
    echo "Usage: sforge-save-version --label <short-label> [--json]" >&2
    echo "Save the current candidate version without running the Judge." >&2
}

LABEL=""
JSON_MODE=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --label)
            if [ "$#" -lt 2 ] || [ -n "$LABEL" ]; then usage; exit 2; fi
            LABEL="$2"
            shift 2
            ;;
        --json) JSON_MODE=1; shift ;;
        *) usage; exit 2 ;;
    esac
done
if [ -z "$LABEL" ] || [ "${#LABEL}" -gt 80 ] \
    || [[ "$LABEL" =~ [[:cntrl:]] ]]; then
    usage
    exit 2
fi

JUDGE_URL="${SFORGE_JUDGE_URL:-}"
TOKEN="${SFORGE_TOKEN:-}"
PATCH_DIR="${SFORGE_PATCH_DIR:-$(pwd)}"
if [ -z "$JUDGE_URL" ]; then
    echo "ERROR: SFORGE_JUDGE_URL not set" >&2
    exit 1
fi
if [ -z "$TOKEN" ]; then
    echo "ERROR: SFORGE_TOKEN not set" >&2
    exit 1
fi

ARCHIVE_FILE=$(mktemp --suffix=.tar.gz)
RESPONSE_FILE=$(mktemp)
cleanup() {
    rm -f -- "$ARCHIVE_FILE" "$RESPONSE_FILE"
}
trap cleanup EXIT

cd "$PATCH_DIR"
TAR_PATHS="${SFORGE_SUBMIT_PATHS:-.}"
if [ -n "${SFORGE_SUBMIT_PATHS:-}" ]; then
    EXISTING_PATHS=""
    for path in $SFORGE_SUBMIT_PATHS; do
        [ -e "$path" ] && EXISTING_PATHS="$EXISTING_PATHS $path"
    done
    TAR_PATHS="${EXISTING_PATHS# }"
fi
if [ -z "$TAR_PATHS" ]; then
    tar czf "$ARCHIVE_FILE" --files-from /dev/null
else
    tar czf "$ARCHIVE_FILE" \
        --transform='s,^\.$,__sforge_archive_root__,' \
        --transform='s,^\./,,' \
        --exclude='.git' ${SFORGE_SUBMIT_EXCLUDE_FLAGS:-} $TAR_PATHS
fi

POST_TIMEOUT=${SFORGE_SAVE_VERSION_POST_TIMEOUT:-120}
case "$POST_TIMEOUT" in
    ''|*[!0-9]*|0) POST_TIMEOUT=120 ;;
esac
if ! HTTP_CODE=$(curl -sS -o "$RESPONSE_FILE" -w '%{http_code}' \
    -m "$POST_TIMEOUT" -X POST "$JUDGE_URL/api/v1/versions" \
    -H "Content-Type: application/gzip" \
    -H "X-SForge-Token: $TOKEN" \
    -H "X-SForge-Version-Label: $LABEL" \
    --data-binary "@$ARCHIVE_FILE"); then
    echo "ERROR: Version store request failed" >&2
    exit 1
fi

if [ "$HTTP_CODE" != "200" ] && [ "$HTTP_CODE" != "201" ]; then
    DETAIL=$(jq -r '.detail // "version save failed"' "$RESPONSE_FILE" 2>/dev/null \
        || echo "version save failed")
    echo "ERROR: $DETAIL (HTTP $HTTP_CODE)" >&2
    exit 1
fi

VERSION_ID=$(jq -r '.version_id // empty' "$RESPONSE_FILE")
ROUND_ID=$(jq -r '.round_id // empty' "$RESPONSE_FILE")
DIGEST=$(jq -r '.archive_sha256 // empty' "$RESPONSE_FILE")
ARCHIVE_SIZE=$(jq -r '.archive_size // empty' "$RESPONSE_FILE")
CANDIDATE_ID=$(jq -r '.candidate_id // empty' "$RESPONSE_FILE")
CONTENT_DIGEST=$(jq -r '.content_sha256 // empty' "$RESPONSE_FILE")
if [ -z "$VERSION_ID" ] || [ -z "$ROUND_ID" ] || [ -z "$DIGEST" ] \
    || [ -z "$ARCHIVE_SIZE" ] || [ -z "$CANDIDATE_ID" ] \
    || [ -z "$CONTENT_DIGEST" ]; then
    echo "ERROR: Version store returned an incomplete response" >&2
    exit 1
fi

if [ "$JSON_MODE" -eq 1 ]; then
    cat "$RESPONSE_FILE"
else
    echo "Saved candidate for replay: $LABEL (content ${CONTENT_DIGEST:0:12})"
fi
'''


def _generate_submit_script() -> str:
    """Generate sforge-submit: token-based POST + poll until result is ready."""
    return r"""#!/bin/bash
set -e

LIST_MODE=0
DETAILS_MODE=0
SAVED_VERSION_ID=""
SAVED_ARCHIVE_SHA256=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --list|-l) LIST_MODE=1; shift ;;
        --details|-d) DETAILS_MODE=1; shift ;;
        --version)
            if [ "$#" -lt 2 ] || [ -z "$2" ] || [ -n "$SAVED_VERSION_ID" ]; then
                echo "ERROR: --version requires one saved version ID" >&2
                exit 2
            fi
            SAVED_VERSION_ID="$2"
            shift 2
            ;;
        --archive-sha256)
            if [ "$#" -lt 2 ] || [ -z "$2" ] || [ -n "$SAVED_ARCHIVE_SHA256" ]; then
                echo "ERROR: --archive-sha256 requires one SHA-256 value" >&2
                exit 2
            fi
            SAVED_ARCHIVE_SHA256="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: sforge-submit [OPTIONS]"
            echo ""
            echo "Submit current code to the judge server for evaluation."
            echo "Results include score, pass rate, and a summary of findings."
            echo ""
            echo "Options:"
            echo "  --list, -l      List manual submissions and scores"
            echo "  --details, -d   Submit and show detailed per-test results (triggers a new submission)"
            echo "  --version ID    Evaluate this exact saved version without repacking current code"
            echo "  --archive-sha256 SHA  Assert the saved receipt digest (requires --version)"
            echo "  -h, --help      Show this help message"
            exit 0
            ;;
        *)
            echo "ERROR: Unknown option: $1" >&2
            echo "Run 'sforge-submit --help' for usage." >&2
            exit 1
            ;;
    esac
done

if [ -n "$SAVED_VERSION_ID" ]; then
    if ! [[ "$SAVED_VERSION_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,254}$ ]] \
        || [ "$LIST_MODE" -eq 1 ]; then
        echo "ERROR: --version requires a valid saved ID and cannot be combined with --list" >&2
        exit 2
    fi
fi
if [ -n "$SAVED_ARCHIVE_SHA256" ]; then
    if [ -z "$SAVED_VERSION_ID" ] || ! [[ "$SAVED_ARCHIVE_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
        echo "ERROR: --archive-sha256 requires --version and a lowercase SHA-256 digest" >&2
        exit 2
    fi
fi

JUDGE_URL="${SFORGE_JUDGE_URL}"
TOKEN="${SFORGE_TOKEN}"
PATCH_DIR="${SFORGE_PATCH_DIR:-$(pwd)}"
STATE_FILE="/tmp/sforge_state.json"

if [ -z "$JUDGE_URL" ]; then
    echo "ERROR: SFORGE_JUDGE_URL not set" >&2
    exit 1
fi
if [ -z "$TOKEN" ]; then
    echo "ERROR: SFORGE_TOKEN not set" >&2
    exit 1
fi

if [ "$LIST_MODE" -eq 1 ]; then
    HIST=$(curl -s -m 30 "$JUDGE_URL/api/v1/history?token=$TOKEN")
    if [ -z "$HIST" ]; then
        echo "ERROR: No response from judge server" >&2
        exit 1
    fi
    BEST_RATE=$(echo "$HIST" | jq -r '.best_pass_rate // 0')
    BEST_SCORE=$(echo "$HIST" | jq -r '.best_score // "N/A"')
    COUNT=$(echo "$HIST" | jq -r '.entries | length')
    echo ""
    echo "========================================"
    echo "  Manual Submission History"
    echo "  Total manual submissions: $COUNT"
    echo "  Best manual pass rate: $(jq -n --argjson r "$BEST_RATE" '$r * 100 | . * 10 | floor / 10')%"
    if [ "$BEST_SCORE" != "N/A" ] && [ "$BEST_SCORE" != "null" ]; then
        echo "  Best manual score: $BEST_SCORE"
    fi
    echo "========================================"
    echo ""
    if [ "$COUNT" -gt 0 ]; then
        printf "  %-14s %-10s %-8s %-12s %-10s %s\n" "ROUND" "STATUS" "VALID" "PASS_RATE" "SCORE" "SUMMARY"
        printf "  %-14s %-10s %-8s %-12s %-10s %s\n" "-----" "------" "-----" "---------" "-----" "-------"
        echo "$HIST" | jq -r '.entries[] | select(.type == "submission") |
            "  " +
            ((.round // "-") | . + " " * ([14 - length, 0] | max)) + " " +
            ((.status // "-") | . + " " * ([10 - length, 0] | max)) + " " +
            (if .valid == false then "no" else "yes" end | . + " " * ([8 - length, 0] | max)) + " " +
            (if .pass_rate != null then (.pass_rate * 100 * 10 | floor / 10 | tostring + "%") else "-" end | . + " " * ([12 - length, 0] | max)) + " " +
            (if .score != null then (.score | tostring) else "-" end | . + " " * ([10 - length, 0] | max)) + " " +
            ((.summary // "-") | if length > 40 then .[:37] + "..." else . end)'
    fi
    echo ""
    exit 0
fi

# ── Archive ──

if [ -z "$SAVED_VERSION_ID" ]; then
cd "$PATCH_DIR"
if [ -n "${SFORGE_PRE_SUBMIT_HOOK:-}" ]; then
    case "$SFORGE_PRE_SUBMIT_HOOK" in
        /*) ;;
        *)
            echo "ERROR: SFORGE_PRE_SUBMIT_HOOK must be an absolute path" >&2
            exit 64
            ;;
    esac
    if [ -L "$SFORGE_PRE_SUBMIT_HOOK" ] \
        || [ ! -f "$SFORGE_PRE_SUBMIT_HOOK" ] \
        || [ ! -x "$SFORGE_PRE_SUBMIT_HOOK" ]; then
        echo "ERROR: SFORGE_PRE_SUBMIT_HOOK must be a regular, non-symlink executable" >&2
        exit 64
    fi
    HOOK_REAL=$(readlink -f -- "$SFORGE_PRE_SUBMIT_HOOK") || {
        echo "ERROR: cannot resolve SFORGE_PRE_SUBMIT_HOOK" >&2
        exit 64
    }
    PATCH_REAL=$(readlink -f -- "$PATCH_DIR") || {
        echo "ERROR: cannot resolve SFORGE_PATCH_DIR" >&2
        exit 64
    }
    case "$HOOK_REAL" in
        "$PATCH_REAL"/*) ;;
        *)
            echo "ERROR: SFORGE_PRE_SUBMIT_HOOK must resolve inside SFORGE_PATCH_DIR" >&2
            exit 64
            ;;
    esac
    echo "Running pre-submit cleanup hook..." >&2
    "$HOOK_REAL" || {
        HOOK_STATUS=$?
        echo "ERROR: pre-submit cleanup hook failed (exit $HOOK_STATUS); submission cancelled" >&2
        exit "$HOOK_STATUS"
    }
fi
ARCHIVE_FILE=$(mktemp --suffix=.tar.gz)
TAR_PATHS="${SFORGE_SUBMIT_PATHS:-.}"
if [ -n "${SFORGE_SUBMIT_PATHS:-}" ]; then
    EXISTING_PATHS=""
    for p in $SFORGE_SUBMIT_PATHS; do
        [ -e "$p" ] && EXISTING_PATHS="$EXISTING_PATHS $p"
    done
    TAR_PATHS="${EXISTING_PATHS# }"
fi
if [ -z "$TAR_PATHS" ]; then
    tar czf "$ARCHIVE_FILE" --files-from /dev/null
else
    # GNU tar spells a cwd-wide submission as `./` plus `./path`.  The Judge
    # trust boundary intentionally requires canonical member names, so map
    # descendants to `path` and give the otherwise-empty root directory a
    # harmless canonical name.  Explicit file/directory submit paths are
    # unchanged unless the task itself spells them with a leading `./`.
    tar czf "$ARCHIVE_FILE" \
        --transform='s,^\.$,__sforge_archive_root__,' \
        --transform='s,^\./,,' \
        --exclude='.git' ${SFORGE_SUBMIT_EXCLUDE_FLAGS:-} $TAR_PATHS
fi
ARCHIVE_SIZE=$(wc -c < "$ARCHIVE_FILE")
else
    # Resolve only authenticated metadata; the Controller chooses all paths.
    # With a receipt SHA, the JSON request pins it directly and needs no GET.
    if [ -z "$SAVED_ARCHIVE_SHA256" ]; then
        REFERENCE_FILE=$(mktemp)
        if ! REFERENCE_HTTP_CODE=$(curl -sS -o "$REFERENCE_FILE" -w '%{http_code}' \
            -m 30 "$JUDGE_URL/api/v1/versions/$SAVED_VERSION_ID" \
            -H "X-SForge-Token: $TOKEN"); then
            rm -f -- "$REFERENCE_FILE"
            echo "ERROR: Cannot resolve saved version" >&2
            exit 1
        fi
        RESOLVED_ID=$(jq -r '.version_id // empty' "$REFERENCE_FILE" 2>/dev/null || true)
        SAVED_ARCHIVE_SHA256=$(jq -r '.archive_sha256 // empty' "$REFERENCE_FILE" 2>/dev/null || true)
        rm -f -- "$REFERENCE_FILE"
        if [ "$REFERENCE_HTTP_CODE" != "200" ] || [ "$RESOLVED_ID" != "$SAVED_VERSION_ID" ] \
            || ! [[ "$SAVED_ARCHIVE_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
            echo "ERROR: Cannot resolve saved version (HTTP $REFERENCE_HTTP_CODE)" >&2
            exit 1
        fi
    fi
fi

echo ""
echo "========================================"
echo "  Submitting for evaluation"
if [ -n "$SAVED_VERSION_ID" ]; then
    echo "  Saved version: $SAVED_VERSION_ID"
else
    echo "  Archive size: $ARCHIVE_SIZE bytes"
fi
echo "  Waiting for test results..."
echo "========================================"
echo ""

# ── Submit + poll ──

# Save the previous authoritative best for an accurate post-result delta.
HISTORY_BEFORE=$(curl -s -m 30 "$JUDGE_URL/api/v1/history?token=$TOKEN" 2>/dev/null || true)
if [ -z "$HISTORY_BEFORE" ] || ! printf '%s\n' "$HISTORY_BEFORE" | jq -e \
    'type == "object"' >/dev/null 2>&1; then
    HISTORY_BEFORE='{}'
fi
PREVIOUS_BEST_SCORE=$(echo "$HISTORY_BEFORE" | jq -r '.best_score // null')
PREVIOUS_BEST_ROUND=$(echo "$HISTORY_BEFORE" | jq -r '.best_round // ""')

SUBMIT_POST_TIMEOUT=${SFORGE_SUBMIT_POST_TIMEOUT:-120}
case "$SUBMIT_POST_TIMEOUT" in
    ''|*[!0-9]*|0) SUBMIT_POST_TIMEOUT=120 ;;
esac
if [ -n "$SAVED_VERSION_ID" ]; then
    REFERENCE_JSON=$(jq -nc --arg version "$SAVED_VERSION_ID" --arg sha "$SAVED_ARCHIVE_SHA256" \
        '{version_id: $version, archive_sha256: $sha}')
    HTTP_CODE=$(curl -s -o /tmp/_submit_resp.json -w '%{http_code}' -m "$SUBMIT_POST_TIMEOUT" -X POST "$JUDGE_URL/api/v1/submit" \
        -H "Content-Type: application/json" \
        -H "X-SForge-Token: $TOKEN" \
        -H "X-SForge-Submission-Kind: agent" \
        --data-binary "$REFERENCE_JSON")
else
    HTTP_CODE=$(curl -s -o /tmp/_submit_resp.json -w '%{http_code}' -m "$SUBMIT_POST_TIMEOUT" -X POST "$JUDGE_URL/api/v1/submit" \
    -H "Content-Type: application/gzip" \
    -H "X-SForge-Token: $TOKEN" \
    -H "X-SForge-Submission-Kind: agent" \
    --data-binary "@$ARCHIVE_FILE")
rm -f "$ARCHIVE_FILE"
fi
SUBMIT_RESP=$(cat /tmp/_submit_resp.json)

if [ "$HTTP_CODE" = "429" ]; then
    DETAIL=$(echo "$SUBMIT_RESP" | jq -r '.detail // empty')
    if echo "$DETAIL" | grep -qi "budget\|exhausted"; then
        echo "SUBMISSION LIMIT REACHED: $DETAIL" >&2
        echo "$DETAIL"
    else
        echo "COOLDOWN: $DETAIL" >&2
        echo "$DETAIL"
    fi
    exit 1
fi

SUBMISSION_ID=$(echo "$SUBMIT_RESP" | jq -r '.submission_id // empty')
ROUND_ID=$(echo "$SUBMIT_RESP" | jq -r '.round_id // empty')
REMAINING=$(echo "$SUBMIT_RESP" | jq -r '.remaining_submissions // empty')
if [ -z "$SUBMISSION_ID" ]; then
    echo "ERROR: Failed to submit to judge server (HTTP $HTTP_CODE)" >&2
    echo "$SUBMIT_RESP" >&2
    exit 1
fi

if [ -n "$ROUND_ID" ]; then
    if [ -n "$REMAINING" ] && [ "$REMAINING" != "null" ]; then
        echo "  Round: $ROUND_ID  (remaining submissions: $REMAINING)"
    else
        echo "  Round: $ROUND_ID"
    fi
    echo ""
fi

POLL_ATTEMPTS=${SFORGE_SUBMIT_POLL_ATTEMPTS:-720}
case "$POLL_ATTEMPTS" in
    ''|*[!0-9]*|0) POLL_ATTEMPTS=720 ;;
esac
if [ "${#POLL_ATTEMPTS}" -gt 5 ] || [ "$POLL_ATTEMPTS" -gt 8640 ]; then
    POLL_ATTEMPTS=8640
fi
for ((attempt=0; attempt<POLL_ATTEMPTS; attempt++)); do
    sleep 10
    RESULT=$(curl -s -m 30 "$JUDGE_URL/api/v1/result/$SUBMISSION_ID" 2>/dev/null || true)
    STATUS=$(echo "$RESULT" | jq -r '.status // empty')
    if [ "$STATUS" = "completed" ] || [ "$STATUS" = "error" ]; then
        break
    fi
done

STATUS=$(echo "$RESULT" | jq -r '.status // empty')
if [ "$STATUS" != "completed" ] && [ "$STATUS" != "error" ]; then
    echo "ERROR: Evaluation timed out or judge unreachable" >&2
    exit 1
fi

# ── Parse + update display cache + print ──

TS=$(date +%s)
ERROR_MSG=$(echo "$RESULT" | jq -r '.error // empty')

if [ -n "$ERROR_MSG" ]; then
    CURRENT_RATE=0
    CURRENT_SCORE="null"
    VALID=false
    SUMMARY=""
    PASSED=0
    TOTAL=0
    FAILED=0
else
    REPORT=$(echo "$RESULT" | jq -r '.report')
    PASSED=$(echo "$REPORT" | jq -r '.passed')
    TOTAL=$(echo "$REPORT" | jq -r '.total_tests')
    FAILED=$(echo "$REPORT" | jq -r '.failed')
    CURRENT_RATE=$(echo "$REPORT" | jq -r '.pass_rate')
    CURRENT_SCORE=$(echo "$REPORT" | jq -r '.score // null')
    VALID=$(echo "$REPORT" | jq -r 'if has("valid") then .valid else true end')
    SUMMARY=$(echo "$REPORT" | jq -r '.summary // empty')
fi

# Refresh the display-only cache from the Judge's authoritative selection.
# The result endpoint can become complete just before the corresponding history
# record is published, so briefly wait for this exact round to appear.
HISTORY_AFTER='{}'
for _ in $(seq 1 10); do
    HISTORY_CANDIDATE=$(curl -s -m 30 "$JUDGE_URL/api/v1/history?token=$TOKEN" 2>/dev/null || true)
    if [ -n "$HISTORY_CANDIDATE" ] && printf '%s\n' "$HISTORY_CANDIDATE" | jq -e \
        --arg round "${ROUND_ID:-}" \
        '.entries[]? | select(.round == $round and (.status == "completed" or .status == "error"))' \
        >/dev/null 2>&1; then
        HISTORY_AFTER="$HISTORY_CANDIDATE"
        break
    fi
    sleep 1
done
if [ "$HISTORY_AFTER" = "{}" ] && [ -n "${HISTORY_CANDIDATE:-}" ] \
    && printf '%s\n' "$HISTORY_CANDIDATE" | jq -e \
    'type == "object"' >/dev/null 2>&1; then
    HISTORY_AFTER="$HISTORY_CANDIDATE"
fi
if [ -z "$HISTORY_AFTER" ] || ! printf '%s\n' "$HISTORY_AFTER" | jq -e \
    'type == "object"' >/dev/null 2>&1; then
    HISTORY_AFTER='{}'
fi
SERVER_BEST_RATE=$(echo "$HISTORY_AFTER" | jq -r '.best_pass_rate // 0')
SERVER_BEST_SCORE=$(echo "$HISTORY_AFTER" | jq -r '.best_score // null')
SERVER_BEST_ROUND=$(echo "$HISTORY_AFTER" | jq -r '.best_round // ""')

# Update local state file (display-only cache — not used for final scoring).
if [ ! -f "$STATE_FILE" ]; then
    echo '{"best_pass_rate": 0, "best_score": null, "best_round": "", "submissions": []}' > "$STATE_FILE"
fi
TMP=$(mktemp)
jq --arg round "${ROUND_ID:-unknown}" \
   --argjson ts "$TS" \
   --argjson rate "$CURRENT_RATE" \
   --argjson score "$CURRENT_SCORE" \
   --argjson valid "$VALID" \
   --argjson best_rate "$SERVER_BEST_RATE" \
   --argjson best_score "$SERVER_BEST_SCORE" \
   --arg best_round "$SERVER_BEST_ROUND" \
   '
   .submissions += [{
     kind: "agent",
     round: $round,
     at: $ts,
     pass_rate: $rate,
     score: $score,
     valid: $valid
   }]
   | .best_pass_rate = $best_rate
   | .best_round = $best_round
   | .best_score = $best_score
   ' "$STATE_FILE" > "$TMP" && mv "$TMP" "$STATE_FILE"

if [ -n "$ERROR_MSG" ]; then
    echo "========================================"
    echo "  ${ROUND_ID:-submission}: ERROR"
    echo "  $ERROR_MSG"
    echo "========================================"
else
    echo "========================================"
    echo "  ${ROUND_ID:-submission} Results"
    echo "========================================"
    PRIMARY_SCORE_NAME=$(echo "$REPORT" | jq -r '.metrics.primary_score_name // empty')
    PRIMARY_SCORE_VALUE=$(echo "$REPORT" | jq -r '.metrics.primary_score_value // .score // empty')
    PRIMARY_SCORE_UNIT=$(echo "$REPORT" | jq -r '.metrics.primary_score_unit // empty')
    PRIMARY_SCORE_FORMULA=$(echo "$REPORT" | jq -r '.metrics.primary_score_formula // empty')
    PRIMARY_SCORE_DIRECTION=$(echo "$REPORT" | jq -r '.metrics.primary_score_direction // "maximize"')
    if [ -n "$PRIMARY_SCORE_NAME" ]; then
        echo ""
        echo "  PRIMARY LEADERBOARD OBJECTIVE"
        echo "  -----------------------------"
        echo "  Metric:      $PRIMARY_SCORE_NAME"
        echo "  Current:     $PRIMARY_SCORE_VALUE${PRIMARY_SCORE_UNIT:+ $PRIMARY_SCORE_UNIT}"
        if [ -n "$SERVER_BEST_ROUND" ] && [ "$SERVER_BEST_SCORE" != "null" ]; then
            echo "  Best manual: $SERVER_BEST_SCORE  (round $SERVER_BEST_ROUND)"
        fi
        if [ "$VALID" = "true" ] && [ "$CURRENT_SCORE" != "null" ]; then
            if [ "$PREVIOUS_BEST_SCORE" = "null" ]; then
                echo "  Manual delta: first valid manual score"
            else
                RAW_DELTA=$(jq -n \
                    --argjson current "$CURRENT_SCORE" \
                    --argjson previous "$PREVIOUS_BEST_SCORE" \
                    '$current - $previous')
                if [ "$PRIMARY_SCORE_DIRECTION" = "minimize" ]; then
                    IMPROVEMENT=$(jq -n --argjson delta "$RAW_DELTA" --argjson zero 0 '$zero - $delta')
                else
                    IMPROVEMENT="$RAW_DELTA"
                fi
                echo "  Manual delta: $IMPROVEMENT vs previous manual best $PREVIOUS_BEST_SCORE${PREVIOUS_BEST_ROUND:+ (round $PREVIOUS_BEST_ROUND)}"
            fi
        fi
        if [ -n "$PRIMARY_SCORE_FORMULA" ]; then
            echo "  Formula:     $PRIMARY_SCORE_FORMULA"
        fi
        echo "  Note:        Secondary diagnostics below do not replace the primary score."
        echo ""
    fi
    if [ "$VALID" = "false" ]; then
        echo "  Valid:       no"
    fi
    if [ "$(echo "$REPORT" | jq -r '.metrics.evaluation_status // empty')" = "integrity_rejected" ]; then
        echo ""
        echo "  INTEGRITY REVIEW: REJECTED (not benchmarked, no score)"
        echo "  ------------------------------------------------------"
        echo "$REPORT" | jq -r '.metrics.integrity_review.rules[]? | "  Rule:        " + .'
        INTEGRITY_FEEDBACK=$(echo "$REPORT" | jq -r '.metrics.integrity_review.feedback // empty')
        if [ -n "$INTEGRITY_FEEDBACK" ]; then
            echo "  Feedback:    $INTEGRITY_FEEDBACK"
        fi
        echo ""
    fi
    if [ "$CURRENT_SCORE" != "null" ]; then
        echo "  Score:       $CURRENT_SCORE"
    fi
    if [ "$TOTAL" -gt 0 ] 2>/dev/null; then
        PASS_PCT=$(jq -n --argjson r "$CURRENT_RATE" '$r * 100 | . * 10 | floor / 10')
        echo "  Pass rate:   ${PASS_PCT}%"
        echo "  Passed:      $PASSED/$TOTAL"
    fi
    if [ -n "$SUMMARY" ]; then
        echo ""
        echo "  Summary:"
        echo "    $SUMMARY"
    fi
    # Show metrics if present
    METRICS=$(echo "$REPORT" | jq -r '.metrics // empty')
    if [ -n "$METRICS" ] && [ "$METRICS" != "{}" ] && [ "$METRICS" != "null" ]; then
        echo ""
        echo "  Metrics:"
        echo "$REPORT" | jq -r '.metrics | to_entries[] | "    \(.key): \(.value)"'
    fi
    echo ""
    # Show failed items (from details if available, else from test_details)
    DETAIL_FAILURES=$(echo "$REPORT" | jq -r '[.details[]? | select(.status != "PASSED")] | length')
    if [ "$DETAIL_FAILURES" -gt 0 ] 2>/dev/null; then
        echo "  Failed checks:"
        echo "$REPORT" | jq -r '.details[] | select(.status != "PASSED") | .name' | head -20 | while read -r t; do echo "    - $t"; done
        if [ "$DETAIL_FAILURES" -gt 20 ] 2>/dev/null; then
            echo "    ... and $((DETAIL_FAILURES - 20)) more"
        fi
    else
        FAILED_TESTS=$(echo "$REPORT" | jq -r '.test_details[]? | select(.status != "PASSED") | .name')
        if [ -n "$FAILED_TESTS" ]; then
            echo "  Failed tests:"
            echo "$FAILED_TESTS" | head -20 | while read -r t; do echo "    - $t"; done
        else
            if [ "$TOTAL" -gt 0 ] 2>/dev/null; then
                echo "  All tests passed!"
            fi
        fi
    fi
    # Show full details if --details flag
    if [ "$DETAILS_MODE" -eq 1 ]; then
        HAS_DETAILS=$(echo "$REPORT" | jq -r '.details | length')
        if [ "$HAS_DETAILS" -gt 0 ] 2>/dev/null; then
            echo ""
            echo "  Details:"
            echo "$REPORT" | jq -r '.details[] | "    [\(.status)] \(.name)\(if .message then ": " + .message else "" end)"'
        fi
    fi
    echo "========================================"
    echo ""
fi
"""


def generate_evolve_prompt(
    original_query: str,
    submit_paths: list[str] | None = None,
    internet: bool = True,
    max_submissions: int | None = None,
    submission_cooldown: int | None = None,
    version_saving_enabled: bool = False,
    integrity_review: bool = False,
) -> str:
    """Generate the enhanced prompt wrapping the task query with eval instructions."""
    version_note = ""
    if version_saving_enabled:
        evaluation_capture_note = ""
        if max_submissions != 0:
            evaluation_capture_note = (
                "- SForge saves accepted evaluation submissions automatically, so "
                "do not save solely for `sforge-submit`. To evaluate an already "
                "saved snapshot, use "
                "`sforge-submit --version <version_id>`; this evaluates that "
                "snapshot even if your workspace has changed. The optional "
                "`--json` flag on `sforge-save-version` returns its version ID "
                "and archive SHA-256 receipt.\n"
            )
        version_note = (
            "\n### Candidate Version Capture\n\n"
            "- Before the first score-producing public test or benchmark in this "
            "session, run "
            "`sforge-save-version --label <short-hypothesis>` and wait for the "
            "command to report a successful save.\n"
            "- After changing or restoring any submitted file, save again before "
            "that implementation's first score-producing public measurement. Wait "
            "for the save to succeed, and do not edit submitted files between the "
            "save and test start.\n"
            "- If submitted files are unchanged since the last successful save, do "
            "not save again.\n"
            f"{evaluation_capture_note}"
            "- Saving only records the submitted files for later replay; it does "
            "not run a test or produce a score. Task-owned version files such as "
            "`FINAL_VERSION` are separate. Use sforge-save-version only for the "
            "Work-local public-measurement cases above, not as a rollback checkpoint; "
            "this rule overrides conflicting task text. If you only update notes or "
            "evidence such as `SELFTEST.md`, do not save that state unless you will "
            "run a score-producing public measurement on that exact state. The final "
            "filesystem state is saved automatically when the work session ends.\n"
        )
    if max_submissions == 0:
        paths_note = ""
        if submit_paths:
            paths_str = ", ".join(f"`{path}`" for path in submit_paths)
            paths_note = (
                "\n### Captured Files\n\n"
                f"Only these paths are captured when the work session ends: {paths_str}\n"
            )
        network_note = ""
        if not internet:
            network_note = (
                "\n### Network Environment\n\n"
                "**This environment has NO internet access.** Only the AI API is "
                "reachable through the restricted runtime proxy. Use only the "
                "dependencies and files already present in the workspace.\n"
            )
        return (
            "## Fixed Work Session\n\n"
            "Use the supplied local checks while you work. When the session ends, "
            "the platform captures one immutable archive from the configured paths.\n\n"
            "### How to Test Your Code\n\n"
            "Use the local public tests and diagnostics in the workspace. Keep the "
            "selected implementation complete, runnable, and on disk throughout "
            "the session.\n"
            f"{paths_note}"
            f"{version_note}"
            f"{network_note}\n"
            "### Strategy\n\n"
            "- Implement incrementally and run the public suite after each change.\n"
            "- Follow Candidate Version Capture whenever you measure a changed "
            "candidate.\n"
            "- Finish with the strongest locally validated implementation selected.\n\n"
            "### Final State\n\n"
            "Only the captured files represent your final implementation.\n\n"
            "---\n\n"
            f"{original_query}\n"
        )

    submit_files_note = ""
    if submit_paths:
        paths_str = ", ".join(f"`{p}`" for p in submit_paths)
        submit_files_note = (
            f"\n### Submitted Files\n\n"
            f"Only the following paths are submitted for evaluation: {paths_str}\n\n"
            f"**Keep these files in a compilable/runnable state whenever you submit.** "
            f"Write changes to disk before invoking the submission tool and ensure "
            f"the submitted files represent the implementation you intend to test.\n"
        )

    network_note = ""
    if not internet:
        network_note = (
            "\n### Network Environment\n\n"
            "**This environment has NO internet access.** "
            "Only the judge server and the AI API are reachable. "
            "Do not attempt to download packages, fetch remote resources, "
            "or access external URLs — all dependencies are pre-installed "
            "in the workspace.\n"
        )

    budget_note = ""
    if max_submissions is not None or submission_cooldown is not None:
        parts = []
        if max_submissions is not None:
            parts.append(
                f"- You have a **limited number of submissions ({max_submissions} total)**. "
                f"Plan carefully and validate locally before submitting."
            )
        if submission_cooldown is not None:
            parts.append(
                f"- There is a **minimum interval of {submission_cooldown}s** between submissions."
            )
        budget_note = "\n### Submission Limits\n\n" + "\n".join(parts) + "\n"

    integrity_note = ""
    if integrity_review:
        integrity_note = (
            "\n### Scope Review\n\n"
            "Before a submission is benchmarked, an automated reviewer compares "
            "your changes with this task's instructions. A submission that "
            "circumvents the task instead of solving it as specified -- for "
            "example by bypassing the required framework or implementation "
            "surface, special-casing the evaluator or its inputs, or using "
            "something the instructions forbid -- is rejected without a score, "
            "and the result explains which rule was broken. Slow, unfinished or "
            "incorrect code and legitimate optimizations are never rejected by "
            "this review.\n"
        )

    how_to_test = (
        "### How to Test Your Code\n\n"
        "- Run `sforge-submit` to submit your current code for evaluation. "
        "It will package the files, send them to the judge server, and return "
        "the validation, score, and summary fields that this evaluator exposes.\n"
        "- Run `sforge-submit --details` to submit and see any additional result "
        "details that this evaluator makes available.\n"
        "- Run `sforge-submit --list` to view previous manual submissions and "
        "their scores for this run.\n\n"
        "You should use these regularly to check your progress and identify issues.\n"
    )

    return (
        "## Iterative Evaluation Mode\n\n"
        "You are working with iterative test feedback. "
        "After implementing code, you can submit your work for evaluation at any "
        "time, then iterate using the returned feedback at its available granularity.\n\n"
        f"{how_to_test}"
        f"{submit_files_note}"
        f"{budget_note}"
        f"{integrity_note}"
        f"{version_note}"
        f"{network_note}\n"
        "### Strategy\n\n"
        "- **Implement incrementally**: Complete one module/project at a time\n"
        "- **Validate locally first**: Use visible tests for detailed debugging\n"
        "- **Read returned feedback carefully**: Some evaluators expose only aggregates\n"
        "- **Iterate**: Fix failing tests based on the feedback, then submit again\n\n"
        "### Scoring\n\n"
        "- Your best manual score is tracked in submission history.\n"
        "- Preserve or restore your strongest implementation in the final files.\n"
        "- You don't lose points for failed attempts — experimentation is encouraged.\n\n"
        "---\n\n"
        f"{original_query}\n"
    )


def generate_game_prompt(original_query: str, internet: bool = True) -> str:
    """Generate the enhanced prompt for game mode.

    Wraps the task-specific query with common game interaction instructions
    (HTTP API docs, tips, keypress handling, rules, scoring). When `internet`
    is False, also tells the agent that walkthrough lookups are blocked, so
    it doesn't waste turns on failing web fetches.
    """
    network_note = ""
    if not internet:
        network_note = (
            "## Network Environment\n\n"
            "**This environment has NO internet access.** Only the game server "
            "(`GAME_SERVER_URL`) and the AI API are reachable; everything else "
            "is iptables-blocked. **Do not** attempt to search the web, fetch "
            "walkthroughs, or download external resources — those calls will "
            "time out and waste your turns. Solve the game through gameplay.\n\n"
            "---\n\n"
        )

    return (
        "## Game Mode\n\n"
        "You are playing an **interactive fiction game** via an HTTP API. "
        "Send commands to the game server and maximize your score.\n\n"
        "---\n\n"
        f"{network_note}"
        "## Game Server HTTP API\n\n"
        "The server URL is available in the `GAME_SERVER_URL` environment variable.\n\n"
        "### Start a new game\n"
        "```\n"
        "POST {GAME_SERVER_URL}/new\n"
        "Body: {}\n"
        'Response: {"session_id": "abc123", "observation": "...", "score": 0, '
        '"peak_score": 0, "max_score": 350, "done": false, "moves": 0}\n'
        "```\n\n"
        "### Take an action\n"
        "```\n"
        "POST {GAME_SERVER_URL}/{session_id}/step\n"
        'Body: {"action": "go north"}\n'
        'Response: {"session_id": "...", "observation": "...", "score": 5, '
        '"peak_score": 5, "max_score": 350, "done": false, "moves": 1}\n'
        "```\n\n"
        "### Check status\n"
        "```\n"
        "GET {GAME_SERVER_URL}/{session_id}/status\n"
        'Response: {"session_id": "...", "score": 5, "peak_score": 5, '
        '"max_score": 350, "done": false, "moves": 1}\n'
        "```\n\n"
        "### Close session\n"
        "```\n"
        "POST {GAME_SERVER_URL}/{session_id}/close\n"
        'Response: {"session_id": "...", "final_score": 5, "peak_score": 5, '
        '"max_score": 350, "moves": 50}\n'
        "```\n\n"
        "### Score fields\n"
        "- `score` — the score you currently have in this session.\n"
        "- `peak_score` — the highest score you reached so far in this session "
        "(useful in games where score can decrease).\n"
        "- `max_score` — the theoretical maximum score for the game (constant). "
        "Your goal is to get `score` as close to `max_score` as possible.\n\n"
        "---\n\n"
        "## Tips\n\n"
        "- Interactive fiction games accept natural language commands: "
        '"go north", "take lamp", "examine door", "open mailbox", "read leaflet", etc.\n'
        "- `look` describes the current room. Moving in a direction that doesn't exist gives a failure message.\n"
        "- The game has puzzles — read descriptions carefully and experiment.\n"
        "- You can start multiple game sessions to explore different strategies.\n"
        "- **Keypress prompts**: Some games pause with prompts like "
        '`(Press SPACE to Continue)` or `[press BACKSPACE to return to game]`. '
        "When you see these in the observation, send a **single space** `\" \"` as "
        "the next action to dismiss them. These are Z-machine interactive prompts "
        "that require a single-character response, not a regular text command.\n\n"
        "---\n\n"
        "## Environment\n\n"
        "- `GAME_SERVER_URL` environment variable is pre-set.\n"
        "- Use `curl`, `urllib.request`, or `http.client` for HTTP requests.\n"
        "- Python 3.10 stdlib is available.\n\n"
        "---\n\n"
        "## Rules\n\n"
        "- All game interaction must go through the HTTP API.\n"
        "- You may start multiple game sessions to explore.\n"
        "- Maximize your score across all sessions.\n\n"
        "## Scoring\n\n"
        "- Your **best score** across all game sessions is your final result, "
        "normalized against `max_score` (the game's theoretical maximum).\n"
        "- You don't lose points for failed sessions — experimentation is encouraged.\n\n"
        "---\n\n"
        f"{original_query}\n"
    )
