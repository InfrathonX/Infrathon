# Miles base images from public pins

The Miles tasks build their Work and Judge images on three base images. This directory holds
their complete recipes, so that any
host can rebuild them:

```bash
scripts/miles_base_images/build.sh model-flat   # miles-model-dependencies-flat:local   OLMo3 v2, OPCD v2, OPD
scripts/miles_base_images/build.sh alfworld     # miles-alfworld-base:local            ALFWorld (on top of model-flat)
scripts/miles_base_images/build.sh diffusion    # miles-diffusion-base:sgl859a278e-md29fde6ee   SD3.5-M MOPD, Wan2.2
```

| Image | Chain | Pins |
|---|---|---|
| `miles-model-dependencies-flat:local` | `radixark/miles@sha256:45ae7833…` → `numpy-scipy.Dockerfile` (NumPy 1.26.4 / SciPy 1.15.3) → `public_runtime.Dockerfile` (removes the bundled Miles, makes one Megatron helper import optional) → `clean.Dockerfile` (tools, `agent` user, no VCS metadata) → `docker export` / `docker import`, cut into ≤4 GiB layers | parent by digest; wheels by exact version; `public_runtime.Dockerfile` asserts the Megatron file hash it edits |
| `miles-alfworld-base:local` | model-flat + the 22 packages `pip install alfworld==0.4.2` adds (`alfworld/requirements.txt`, installed `--no-deps` in a builder stage) | exact versions |
| `miles-diffusion-base:sgl859a278e-md29fde6ee` | `lmsysorg/sglang@sha256:9e02c8e1…` → `diffusion/Dockerfile` (sglang-miles-h3 at `859a278e`, FA3 wheel by sha256, miles_diffusion 29fde6ee requirements, hpsv2 by commit) → flattened, cut into ≤4 GiB layers | parent by digest, source by commit, wheel by sha256 |

Each build writes `outputs/miles-base-images/<model-flat|alfworld|diffusion>.json` with the parent
digest, the recipe digests and the resulting image ID (`MILES_BASE_IMAGE_RECORDS` moves that
directory). `diffusion/Dockerfile` and `diffusion/requirements.txt` are the qualified recipe
(sha256 `a7fdfa23…` and `9973e37f…`).

**Layers.** The flattened rootfs is cut into parts of at most 4 GiB by
`scripts/infrathon/split_layers.py` (same files, owners, modes and config), because one 50-65 GB layer is
fragile to push to and pull from a registry. Bases built before this step are one layer; their
task images are unchanged in content.

**Image identity.** A rebuild does not reproduce the qualified image IDs (`docker import` and `apt-get`
timestamps differ), so the task build scripts accept, in order: `MILES_<KIND>_BASE_ID` from the
environment, the `image_id` recorded here by `build.sh`, or the ID pinned by the qualified
build (`scripts/miles_common/host_config.py base-id`). A new base image is a new image identity:
re-qualify the task on it before comparing scores.
