# engine_overlay/

Your versions of the whitelisted sglang-diffusion files go here, at their path relative to
`/sgl-workspace/sglang/python/sglang/multimodal_gen/` (the clean engine, `sglang-miles-h3@859a278e`):

- `configs/post_training/rl_rollout.py`
- `configs/sample/sampling_params.py` (new fields only)
- `runtime/post_training/scheduler_rl_mixin.py`
- `runtime/post_training/rl_dataclasses.py`
- `runtime/pipelines_core/stages/latent_preparation.py` (RNG source only)
- `runtime/entrypoints/post_training/io_struct.py`, `runtime/entrypoints/post_training/rollout_api.py` (request-field pass-through only)

Any other file here fails the Judge (J0). The existing `sde` / `cps` / `ode` behaviour must stay bitwise identical. Added
code may not import `sys`, `importlib`, `inspect`, `ctypes`, `gc`, `builtins`, and may not call `__import__`, `setattr`,
`exec` or `eval`, or write `os.environ`. For the `run` phase the Judge copies the clean engine, lays these files over it,
and passes the result as the contract's `engine_python_path`. To test locally, do the same:
copy `/sgl-workspace/sglang/python`, overlay, and put the copy first on `PYTHONPATH` of your engine processes.
