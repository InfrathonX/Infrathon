# runtime_v001

Pristine seed: a static 15/15 two-rank NCCL lag ring with per-chunk/per-step noise
identity, complete matched drain, decode overlap, and the exact upstream rolling-KV
policy.

`Runtime` receives a `StreamPlanMeta` (chunk descriptors, indexed noise, cache
geometry; no input video). Frames arrive only through `start` and `push`, and the
controller, not the runtime, records when each chunk was offered.

Runtime accepts the keyword-only master_port=None. None selects a free loopback TCP port at construction; an explicit integer selects that rendezvous port. The controller passes a port for each case.
