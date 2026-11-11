"""PUBLIC TOY TESTS ONLY. Never replace these with private benchmark tests."""
import importlib.util
import json
import sys

spec = importlib.util.spec_from_file_location(
    "candidate", sys.argv[1] if len(sys.argv) > 1 else "/workspace/project/solution.py"
)
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)
cases = [(1, 2), (-4, 9), (0, 0), (10, -3)]
passed = sum(candidate.add(a, b) == a + b for a, b in cases)
print(json.dumps({"valid": True, "score": passed / len(cases),
                  "pass_rate": passed / len(cases),
                  "summary": f"{passed}/{len(cases)} public toy cases passed"}))
