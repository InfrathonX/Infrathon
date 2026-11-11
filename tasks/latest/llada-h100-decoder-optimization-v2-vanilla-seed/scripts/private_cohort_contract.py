#!/usr/bin/env python3
"""Print the contract fields of the experiment manifest's private_cohort (JSON).

scripts/run_fixed_budget.sh writes them into experiment-manifest.json, and
scripts/validate_formal_artifacts.py pre checks them against the same contract.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from h100_policy_contract import interactive_private_cohort  # noqa: E402

print(json.dumps(interactive_private_cohort(), sort_keys=True))
