# Host-only reference scorers

Floor references for the final holdout (the seed is an empty file). Like
`oracle/`, they never enter a build context and are not shown to
participants. Pack and evaluate:

```bash
tar -C oracle/references/knorm --sort=name --mtime='UTC 1970-01-01' --mode=u=rwX,go=rX \
  --owner=0 --group=0 --numeric-owner -czf /tmp/knorm.tar.gz FINAL_VERSION notes.md candidates
SFORGE_LLM_JUDGE_MODE=off JUDGE_GPUS=<i,j> scripts/eval_archive.sh /tmp/knorm.tar.gz
```

`SFORGE_LLM_JUDGE_MODE=off` skips the LLM integrity review, which is for agent
submissions; without it `sforge eval` needs the reviewer settings.
