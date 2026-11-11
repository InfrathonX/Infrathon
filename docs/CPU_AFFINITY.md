# CPU affinity for Work and Judge containers (opt-in)

SForge can pin every Work and Judge container it starts for a run, an eval or
a replay to the CPUs local to that container's own GPU(s), and bind the
container's memory to the GPU's NUMA node. Pinning is **off by default**. With
no policy set, container creation, the records, `run_config.json` and the
replay evaluator descriptor's contents carry no affinity information at all.

## Why

Without pinning, a single-GPU container sees every host CPU but gets a CFS
quota of only 8 CPUs (`--cpus 8`). torch sizes its thread pool from the CPUs it
can see, so it starts far more threads than the quota allows. The quota runs
out early in each 100 ms period, and the container is throttled for the rest
of it, which can make millisecond kernels read an order of magnitude slower.
Unpinned threads can also land on the other socket, and host memory can end up
far from the GPU.

A GPU-local cpuset of the same size as the quota fixes both problems:

- torch sees 8 CPUs and opens 8 threads, so it does not oversubscribe the
  quota and the heavy throttling is gone;
- the threads and the memory stay on the GPU's NUMA node.

The quota is kept, so a slice that is kept fully busy still meets it at the
edge of many periods. On the reference host with 8 saturated CPUs, 55–70% of
the 100 ms periods count as throttled, but the throttled time is only about
0.1% of the CPU time (unpinned, 32 threads under the same quota are throttled
in every period, for about three times the CPU time), and an 8-thread GEMM's
p99 rises by 2–6%. A container that leaves a CPU idle is not throttled at all.

## Enabling it

The policy travels from the launch side to SForge through the environment.
The following all set the same variable:

| Where | How |
| --- | --- |
| One launch (preferred) | `launch/<task>/run.sh ... --cpuset auto` |
| One replay batch | `launch/<task>/replay.sh --cpuset auto ...` |
| Every launch on a host | `SFORGE_CPUSET=auto` in `launch/mlsysforge.env` (or `.json`); see the caveat below |
| A task script run directly | `export SFORGE_CPUSET=auto` before `eval_archive.sh`, `smoke_test.sh`, `replay_candidates.sh`, ... |

The last row works for a simple reason: every Work and Judge container is
started by the SForge harness (`sforge serve` / `run` / `eval` / `replay-*`),
and the harness applies the policy to each container from that container's own
GPU. The GPU comes from its device request, or from `NVIDIA_VISIBLE_DEVICES`.
So an operator path that never sees `--device-uuid`, such as `eval_archive.sh`
(which picks the Judge GPU from `PHYSICAL_GPU` and passes its UUID), is pinned
correctly with no task edit. The task scripts pass GPU UUIDs, which SForge
maps to PCI addresses through procfs; a bare NVML index is only a fallback
that needs NVML or `nvidia-smi`.

**Host-wide caveat.** A policy in the launch config reaches every `run.sh` and
`replay.sh` that loads that config, not only the tasks you had in mind. For
example, concurrent-serving stays unpinned, but its `run_config.json`, its
`cpu_affinity*.json` records and its replay evaluator descriptor change, so a
replay campaign begun without the policy cannot be continued with it.

Opt a launch out with `--cpuset off`. Where there is no flag
(concurrent-serving, a task script run directly), `export SFORGE_CPUSET=off`
works, because the environment beats the config; if the config also sets a
per-role value, export that role's variable as `off` too.

Precedence holds per variable: `--cpuset` > environment > launch config > off.
`--cpuset` sets only `SFORGE_CPUSET`, so a `SFORGE_WORK_CPUSET` or
`SFORGE_JUDGE_CPUSET` from the environment or config still wins for its role;
the launcher prints a warning when that happens. `--cpuset off` also sets both
per-role variables to `off`, so it disables pinning outright.

`--cpuset` is available on the single-GPU launchers, on the multi-GPU-Judge
launchers (kvpress, kvzip, quant-videogen), on the dual-GPU launchers
(sdv2-pipeline, qwen3-gkd, qwen3-orpo), on the eight-GPU Flux launcher
(flux-moe-ag), and on the `replay.sh` of each of those. The concurrent-serving, MInference v3 and Miles launchers have
no such flag (the Miles supervisors read the policy from the environment, see
[Task notes](#task-notes)). A pinned replay is a separate measurement set from
an unpinned one, so give every replay into one `--measurement-set` the same
policy.

### Variables

| Variable | Values | Meaning |
| --- | --- | --- |
| `SFORGE_CPUSET` | `off` \| `auto` \| `CPULIST[@NODELIST]` | Policy for every Work and Judge container |
| `SFORGE_WORK_CPUSET` | same | Overrides `SFORGE_CPUSET` for Work (including `off`) |
| `SFORGE_JUDGE_CPUSET` | same | Overrides `SFORGE_CPUSET` for Judge (including `off`) |
| `SFORGE_CPUSET_MEMS` | `bind` (default) \| `none` | `bind` sets `cpuset_mems` to the slice's NUMA nodes; `none` pins CPUs only |
| `SFORGE_CPUSET_RESERVE` | `CPULIST` (default empty) | CPUs removed before the automatic partition, for host daemons |
| `SFORGE_CPUSET_UNDERSIZE` | `refuse` (default) \| `allow` | What happens when a container's CPU budget exceeds its slice |

- **Syntax.** `CPULIST` is the kernel cpulist syntax (`8-11,40-43`), and
  `NODELIST` is the same syntax over NUMA node IDs.
- **What each value does.**
  - `auto` pins each container to the union of its GPUs' slices, with memory on
    those GPUs' nodes.
  - An explicit list without `@` binds memory to the nodes of the listed CPUs.
- **Checks.** The launcher checks the grammar, and refuses an `@NODELIST`
  together with `SFORGE_CPUSET_MEMS=none`; it never reads a GPU. SForge checks
  everything else against the host topology when it pins a container. An
  explicit list must name online CPUs, and in `bind` mode every CPU must lie on
  a listed node. An explicit list is never compared with the container's GPU:
  choosing CPUs near that GPU, and apart from other runs, is up to you.
- **Separate GPUs.** The default of the single-GPU launchers (also
  `--legacy-dual-gpu`) and of the dual-GPU launchers runs Work and Judge at the
  same time on GPUs of their own: one each, or a pair each. The launcher
  therefore rejects an explicit global list there, and per-role lists that
  share a CPU. `auto`, or two disjoint per-role lists, work. An explicit list
  for one role is not compared with the other role's `auto` slice. A task
  script started directly (a single-GPU task's `run_fixed_budget.sh`, or a
  dual-GPU task's fixed-budget script with a Judge pair) runs in this
  concurrent mode without the launcher's check, so set disjoint per-role lists
  (or `auto`) there, never one explicit global list. The shared modes
  (`--shared-gpu`, `--shared-gpus`) run the two roles one after the other on
  the same GPUs, so both get the same slice.

## How the slices are derived (`gpu-local-l3-v1`)

The slices are computed from sysfs every time a container is pinned. No host
constant is involved, and the result is the same on every run on the same
host.

1. **GPUs.** The algorithm finds every NVIDIA display or 3D controller PCI
   function (class `0x0300`/`0x0302`; NVSwitches and bridges are skipped).
   - It reads each GPU's `numa_node`.
   - Every GPU on the node counts, even one this operator does not use, so the
     partition is static per host.
   - The GPUs of a node are ordered by PCI address.
2. **CPUs of the node.** Online CPUs on the GPU's node (limited by the cgroup
   root's effective cpuset when that is readable), minus
   `SFORGE_CPUSET_RESERVE`.
   - The CPUs are grouped into whole cores, keeping SMT siblings together.
   - The cores are then grouped into L3 groups, ranked by lowest CPU (never by
     the cache `id`).
3. **Whole-L3 mode.** This applies when the node's L3 groups divide evenly
   among its GPUs.
   - GPU *i* owns the *i*-th run of `groups / GPUs` L3 groups.
   - Every GPU on the node gets the same number of cores. Any surplus is left
     spare.
4. **Otherwise**, the node's cores are cut into equal contiguous whole-core
   chunks in L3 order. The remainder is spare.
5. **Container slice.** A container's slice is the union of its GPUs' slices,
   with mems set to their nodes. So a dual-GPU container gets both slices.
6. **Budget.**
   - The container's CPU budget (`nano_cpus`) is kept.
   - A slice larger than the budget is trimmed to it: whole cores first, in L3
     order.
   - If the budget is larger than the slice, the container is refused unless
     `SFORGE_CPUSET_UNDERSIZE=allow`.

**Hosts without NUMA information.**
- A single-node host uses all online CPUs as the domain.
- On a multi-node host a GPU with `numa_node = -1` is refused in `auto` mode;
  give an explicit `CPULIST@NODES` instead.

To print the table for the host you are on (sysfs only, no GPU access):

```bash
PYTHONPATH=. python3 -m sforge.harness.host_topology table
```

Every launcher that has `--cpuset` prints the same table to the launching
terminal before delegating, whenever a policy is active. With `--tmux` the
table is not repeated in the tmux run log; the per-container records below
are the durable evidence. `host_topology check --gpu <UUID>` validates the
environment's policy and prints that GPU's slice.

### Example: an 8-GPU H100 NVL host

This host has 2 sockets × 16 cores with SMT (64 logical CPUs), 8 L3 groups of
4 cores, and 8 GPUs (4 per NUMA node). Its node 0 holds CPUs 0-15 and 32-47;
node 1 holds 16-31 and 48-63. The L3 groups divide evenly (one per GPU), so
each GPU gets one whole L3 group: 4 cores plus their SMT siblings, which is 8
logical CPUs and equals the single-GPU tasks' 8-CPU budget.

| Index (PCI order) | PCI address | NUMA node | CPU slice | `cpuset_mems` |
| --- | --- | --- | --- | --- |
| 0 | `0000:46:00.0` | 0 | `0-3,32-35` | `0` |
| 1 | `0000:49:00.0` | 0 | `4-7,36-39` | `0` |
| 2 | `0000:4e:00.0` | 0 | `8-11,40-43` | `0` |
| 3 | `0000:4f:00.0` | 0 | `12-15,44-47` | `0` |
| 4 | `0000:c5:00.0` | 1 | `16-19,48-51` | `1` |
| 5 | `0000:c6:00.0` | 1 | `20-23,52-55` | `1` |
| 6 | `0000:c9:00.0` | 1 | `24-27,56-59` | `1` |
| 7 | `0000:d0:00.0` | 1 | `28-31,60-63` | `1` |

The NVLink pairs are all within one node, so the dual-GPU tasks get two whole
L3 groups each. That is 16 logical CPUs, which equals their 16-CPU budget:

| Pair | Union slice | `cpuset_mems` |
| --- | --- | --- |
| 0,1 | `0-7,32-39` | `0` |
| 2,3 | `8-15,40-47` | `0` |
| 4,5 | `16-23,48-55` | `1` |
| 6,7 | `24-31,56-63` | `1` |

With the dual-GPU launchers' default pairs, Work on 0,1 gets `0-7,32-39` and
the Judge on 2,3 gets `8-15,40-47`: both on node 0, sharing no core or L3.

With `SFORGE_CPUSET_RESERVE=0,32`, the node-0 GPUs drop to 3 cores each. All
four stay L3-confined, and CPUs 7, 11, 15, 39, 43 and 47 become spare. Their
6-CPU slices are then below the 8-CPU (single) and 16-CPU (pair) budgets, so
every node-0 container is refused unless `SFORGE_CPUSET_UNDERSIZE=allow`.

Keep in mind:
- **Node-level locality only.** sysfs reports which node a GPU belongs to, not
  which L3 group is nearest to it.
- **What the per-GPU L3 split buys.** Locality comes from the right node plus
  `cpuset_mems`. The one-L3-group-per-GPU split adds isolation: runs on
  different GPUs share neither an L3 nor a physical core.
- **What it does not isolate.** A busy neighbour on the other socket is not
  measurable. A busy neighbour on another CCD of the same socket still slows
  a 4-thread GEMM by about 6% (and raises a Python loop's p99 by about 6%),
  presumably through socket-level clock and power sharing.

## What gets recorded

Records are written only when a policy is set; unpinned runs keep their
records and digests unchanged.

- **Per container** (host side only, never visible to the Judge):
  - Work writes `cpu_affinity_work.json` in the run's task log directory.
  - Each Judge writes `cpu_affinity.json` in its submission log directory.
  - Each file holds the policy, the GPUs, the slice, the applied
    `cpuset_cpus`/`cpuset_mems`, the CPU budget and whether the slice was
    trimmed.
- **`run_config.json`** gains `"cpu_affinity": {"work": <policy_id>, "judge":
  <policy_id>}`.
- **Replay evaluator descriptor.** It gains `"judge_cpu_affinity":
  <policy_id>`, so a pinned replay cannot continue an unpinned measurement set.

`policy_id` is normalized and host-independent. It never contains raw CPU
IDs, so replays on different GPUs of the same host stay comparable:

- `auto:gpu-local-l3-v1:mems=bind|none`, plus `:reserve=<n>` when a reserve
  is set
- `explicit:cpus=<n>:nodes=<n>|local:mems=bind|none`, where `local` means an
  explicit list without `@NODELIST`
- either form gets `:undersize=allow` appended when
  `SFORGE_CPUSET_UNDERSIZE=allow`

Pinned numbers must not be compared with unpinned numbers (kernel times,
tokens/s, GB/s anchors).

## Failure behaviour (fail closed)

If a policy is set and the pin cannot be applied or verified, the container
does not start:

- **Judge.** The submission gets an invalid `infrastructure_error` report and
  the created container is destroyed. In a shared mode, where the Judge
  pauses Work, the Work container is then resumed, as after any failed Judge.
- **Work.** The run fails to start, and the created container is removed.

In `auto` mode a container whose GPUs cannot be named exactly is refused the
same way: a GPU request by count or `all`, a MIG device, or a device request
that is not recognizably an NVIDIA GPU request (a CDI request, for example).
Give such a container an explicit `CPULIST@NODES`.

The pin is a `docker update` of the created, not-yet-started container,
followed by a re-read that must show the requested `CpusetCpus`/`CpusetMems`.
The backend must be Docker; any other backend with a policy set is rejected
when the config is loaded.

## Task notes

- **Single-GPU tasks** (attn-qat, eagle3, flash-kmeans, lmcache, standalone-*,
  tilelang): by default Work and Judge run on a GPU each and each container is
  pinned to its own GPU's slice; with `--shared-gpu` they share one GPU serially
  and that GPU's slice. On an 8-GPU H100 NVL host the budget and the slice are both 8.
- **Dual-GPU tasks** (sdv2-pipeline, qwen3-gkd, qwen3-orpo): each
  container gets the union of its pair's two slices (16 = 16 on an 8-GPU
  H100 NVL host). By default Work is pinned to the Work pair's slice and every
  Judge, the final evaluation included, to the Judge pair's slice; the two run
  at the same time, so the launcher refuses one explicit global list (see
  Separate GPUs above). In shared mode (`--shared-gpus` or `--gpus`) both roles get the one
  pair's slice. Index-based pairs work with `auto`, because the task adapters
  attach both GPUs of a role's pair to its containers. `replay.sh` runs its
  Judges on one pair and pins them to that pair's slice.
- **Flux** (flux-moe-ag): Work and the Judge share the run's eight GPUs one
  after the other (Work is paused while the Judge measures), so both roles get
  the same slice: with `auto`, the union of the eight GPUs' slices. `replay.sh`
  runs its Judges on the same eight GPUs.
- **kvpress, kvzip** (several Judge GPUs): the task adapter pins the Judge to
  its GPUs' CPUs even with pinning off. Each NUMA node's whole cores are split
  evenly across that node's GPUs, and the Judge gets the blocks of its own GPUs,
  so Judges on different GPUs never share a CPU. The dry run prints the set.
  Unpinned containers of other runs can still be scheduled on those CPUs.
- **llada-h100** has the flag like every other single-GPU launcher, but its
  quality reference is calibrated unpinned: pinning LLaDA needs a new
  calibration under pinning. Keep it off; under a host-wide policy, launch it
  with `--cpuset off` and export `SFORGE_CPUSET=off` before its
  `eval_archive.sh`.
- **Miles family.** Their launchers have no `--cpuset`. Their supervisors
  rebuild the child environment from an allowlist and pass on the
  `SFORGE_CPUSET*` values from the environment, else from the host config file
  (`launch/mlsysforge.env` or `$MLSYSFORGE_CONFIG`), never from the Miles
  `--config` file. A formal run
  refuses a Judge policy other than the one its qualification record names
  (`auto` for the H100 records).
  - OPCD, OPD and ALFWorld: the harness pins Work to the Work pair's slice,
    and every Judge, the final evaluation included, to the Judge pair's slice
    (both to the one pair's slice with `JUDGE_GPU_INDICES=shared`). The
    containers keep their 24-CPU quota, more than a pair's 16-CPU slice on an
    8-GPU H100 NVL host, so the supervisor sets `SFORGE_CPUSET_UNDERSIZE=allow`
    unless you set it yourself, and records that it did. With a separate Judge
    pair, a policy that gives the two roles a common CPU is refused.
  - SD3.5-M and Wan2.2 set their containers' cpuset themselves. Under a policy
    each role gets its own pair's slice. SD3.5-M keeps a quota of at least 24
    CPUs; Wan2.2's quota is the slice's CPU count.
    `CPUSET`, `JUDGE_CPUSET` and `MDOPD_CPU_CORES` / `MDWAN_CPU_CORES` must then
    be unset. Without a policy Work takes `CPUSET` from the config, else the
    first `MDOPD_CPU_CORES` / `MDWAN_CPU_CORES` physical cores local to its
    pair; a separate Judge takes `JUDGE_CPUSET`, else as many cores local to
    the Judge pair that Work does not use; a shared Judge takes Work's. A
    separate Judge's cores may not overlap Work's, and neither set may overlap
    the pinned cores of another running container.
- **concurrent-serving** has no `--cpuset`. Its Work and Judge containers are
  GPU-free (`NVIDIA_VISIBLE_DEVICES=void`), so `auto` leaves them unpinned
  (outcome `unpinned_no_gpu`). A set policy is still recorded, though: its
  `run_config.json`, records and replay evaluator descriptor change. Run it
  with `SFORGE_CPUSET=off` exported (and any per-role value from the config
  set to `off`) so its identity stays that of an unpinned run. The host vLLM
  backend is outside SForge.

### Resource profiles above the slice

Local Docker containers take their task's deployment resource profile size by
default (`SFORGE_RESOURCE_PROFILE`; `native` uses the task JSON's own size).
Some profiles give a role more CPU quota than its GPU-local slice holds, for
example the Miles tasks' 24-CPU quota on an H100 NVL pair with 16 local CPUs. With `auto` and `SFORGE_CPUSET_UNDERSIZE` unset, SForge then
pins such a container to its slice and keeps the larger quota, so it runs on
the slice's CPUs. It does so only when the slice still holds the task's own CPU
count (8 per H100, 16 per pair); otherwise `auto` stays refused. An explicit
`SFORGE_CPUSET_UNDERSIZE` always wins. The decision is recorded under
`cpuset_undersize` in the run's `resource_profile` record.

### Not pinned by this feature

These float across all host CPUs:

- game-session containers that the Judge server starts itself (no H100 task
  uses them), even though they use the Judge image;
- restricted-proxy sidecars and curl probes;
- image builds and fixture containers;
- raw `docker run` timing sites in task scripts, such as the public-suite
  containers some `smoke_test.sh` scripts start directly (only the Judge those
  scripts start through `sforge eval` is pinned);
- host processes (`sforge serve`/`run`, dockerd), GPU interrupts, and other
  users' containers.

Pinning confines our containers. It does not reserve CPUs for them.

`os.cpu_count()` still reports every host CPU inside a container. A pool sized
from it oversubscribes the slice: its threads time-share the slice's CPUs,
and the container sees only the small edge throttling described under
[Why](#why), not the unpinned kind. GPU clocks and power limits are not
touched.

## Verifying a pinned container

On the host:

```bash
docker inspect -f '{{.HostConfig.CpusetCpus}} {{.HostConfig.CpusetMems}} {{.HostConfig.NanoCpus}}' <container>
```

Inside the container:

```bash
docker exec <container> cat /sys/fs/cgroup/cpuset.cpus.effective /sys/fs/cgroup/cpuset.mems.effective
docker exec <container> nproc                                  # slice size, e.g. 8
docker exec <container> python3 -c 'import os; print(len(os.sched_getaffinity(0)))'
docker exec <container> python3 -c 'import torch; print(torch.get_num_threads())'
docker exec <container> cat /sys/fs/cgroup/cpu.stat           # see below
```

In `cpu.stat`, compare `throttled_usec` with `usage_usec`: it should stay a
tiny fraction (about 0.1% even with every CPU of the slice busy).
`nr_throttled` may count many periods while the slice is saturated; that alone
does not mean the pin failed.

The host-side `cpu_affinity*.json` record must show `"outcome": "pinned"` for
every GPU container, or `"task_owned"` for one whose task sets its own cpuset. Its `policy_id` must equal the one in the run's
`run_config.json`.
