# Host-only reference policies

Qualification references for the final holdout. `expected_attention` is byte-identical to
the Oracle's `candidates/policy_v002` (selected by `oracle/FINAL_VERSION`). Like `oracle/`, they never enter
a build context and are not shown to participants. Each directory is a lab
root that packs into an archive:

```bash
tar -C oracle/references/expected_attention --sort=name --owner=0 --group=0 --numeric-owner \
  -czf /tmp/expected_attention.tar.gz FINAL_VERSION notes.md candidates
JUDGE_GPUS=<i,j> scripts/eval_archive.sh /tmp/expected_attention.tar.gz
```
