# Agent Instructions: test-unimer

## Scope and source of truth

- This repository is the active Kaggle training and research project.
- The active notebook is `uni-mumer-kaggle-dagshub-banmoi.ipynb`, as confirmed by the owner. Do not switch to v5, v8, or an exported snapshot based on its filename or timestamp.
- Read `docs/README.md`, `docs/IMPLEMENTATION_PLAN.md`, and `docs/KAGGLE_RUNBOOK.md` before changing the pipeline.
- Runtime configuration is the base `train/Uni-MuMER-train.yaml` plus notebook overrides. Inspect both; report the final values actually passed to training.
- Dataset registration is `train/dataset_info.json` because the current YAML sets `dataset_dir: train`. Do not edit the vendored framework's example registry by mistake.
- `/home/nhat/Uni-MuMER-project` contains previous models, results, and archives. Leave it untouched unless the user explicitly includes it in the task.

## Permissions and preservation

- Explain findings and progress to the owner in simple Vietnamese.
- A request to review, plan, or write documentation does not authorize implementation changes or training.
- Do not install dependencies, download model/dataset artifacts, launch GPU jobs, or write to DagsHub without authorization for that work. A DagsHub connection check can create/delete a remote test run; it is not read-only.
- Never delete, overwrite, reset, or clean existing models, checkpoints, results, notebooks, or unrelated local changes to make a command pass.
- Inspect `git status` and relevant diffs before editing. Use narrow, explicit file lists when staging. Do not commit/push without the user's request.
- Never print or commit tokens, passwords, `.env`, authenticated URLs, or notebook outputs containing secrets.
- Preserve legacy results. The completeness audit found all expected image IDs in the 24 saved prediction files checked; a risky code path is not evidence that old runs lost samples.

## Required Kaggle environment

- All training, inference, evaluation, MLflow, dataset processing, and runtime YAML helpers must use a project-owned Python environment. Do not rely on Kaggle's preinstalled ML packages.
- Keep the current Python 3.10 / Conda approach unless a separately tested migration is authorized. Expected interpreter: `/kaggle/working/miniconda/envs/unimumer/bin/python`.
- Use the absolute interpreter for commands and `"$PYTHON" -m pip` for installs. Ensure subprocess CLI tools and `torchrun` also resolve inside that environment.
- The notebook kernel is an orchestration surface only. Its platform bridge (`kaggle_secrets`) may retrieve credentials, but do not import project ML/YAML helpers in that kernel. Run those helpers in the isolated interpreter and return JSON or paths.
- Disable user-site imports and do not inject Kaggle/system site-packages into `PYTHONPATH`. Allow only project source/editable framework paths in addition to the environment's own packages.
- Install declared dependencies in a clean Kaggle session, then require `pip check`, import-path checks, and CUDA checks to pass. Do not suppress dependency conflicts or bypass them with an unexplained `--no-deps`.
- Current requirements are a starting point, not a validated lockfile. Do not claim compatibility based on an old notebook's outputs or a successful import alone.

## Hardware and experiment invariants

- Target profile is Kaggle 2 x NVIDIA T4, nominally 16 GB per GPU. Verify actual devices, VRAM, driver, CUDA, and distributed world size every session. Never infer them from a notebook heading.
- Use FP16 for the T4 profile: `bf16: false`, `fp16: true`; inference compute dtype must agree. Do not copy BF16 from the current base YAML.
- Two DDP GPUs do not provide one pooled 32 GB allocation. Each rank must fit on its own GPU.
- Preserve the baseline backbone, NF4 4-bit quantization, LoRA rank 64, alpha 16, and dropout 0.05 unless the experiment explicitly changes them.
- Effective batch = per-device batch x accumulation x actual world size. Record changes, including OOM mitigations. Never silently change image resolution or sequence length to fit memory.
- `max_samples` caps each dataset, not the entire mixture. Count actual rows and unique source images. Keep dataset revision, split manifest, seed, and final runtime YAML with each run.

## Work order and acceptance gates

1. Inventory: read the active notebook, invoked scripts, YAML, dependencies, Git state, and existing artifacts. Record facts separately from assumptions.
2. Stabilize execution: isolate all Python work, validate T4 precision/dependencies, identify the actual notebook and Git revision, and separate output/sync state per run. Add focused CPU tests before GPU tests.
3. Secure checkpoints: distinguish a new run, full resume, and an adapter warm start. Validate configuration, weights, trainer/optimizer/scheduler state and applicable RNG/scaler state. Resolve the real MLflow run ID; an application UUID is only a tag.
4. Validate data/evaluation: split by source image before creating task variants; assert image/label coverage; use deterministic generation and actual sample counts. Keep test sets outside adaptation and checkpoint selection.
5. Authorized smoke test: run only a short, explicitly bounded job, verify checkpoint upload/download/resume and 1-3 inference samples. Save evidence; do not treat smoke metrics as a benchmark.
6. Baseline and task ablations: use the same starting weights, clean split and matched budgets. Report tokens/GPU time as well as steps. Preserve data-source proportions where task effects are being compared.
7. Adaptive learning: only after the previous gates pass, add structure features, task-affinity estimates and phase-wise sampling. Save sampler/RNG/phase state with checkpoints and test resume with the changed data stream.

## Stop conditions

- Stop before expensive work if environment, GPU profile, precision, resolved YAML, paths, or checkpoint provenance do not match the declared run.
- Stop a full benchmark on missing/duplicate/unexpected image IDs, missing labels, or runtime failures. A model-generated empty answer counts as an error; a missing image or CUDA failure must not be silently scored as a normal answer.
- Do not silently replace a failed resume with training from scratch or reuse an old result directory as a new success.
- Require verified remote upload before allowing retention logic to remove the only local resumable checkpoint. Handle interrupted uploads and concurrent final sync explicitly.
- Never claim "Kaggle-ready" until a fresh Kaggle environment has passed the required installation, GPU, checkpoint, and smoke checks. Local static tests cannot establish that result.

## Editing and handoff

- Prefer existing helpers and keep changes outside `train/LLaMA-Factory` when practical. Do not upgrade the vendored framework as an unrelated cleanup.
- Keep the notebook small; put reusable logic in scripts and test it independently. Preserve valid notebook JSON and cell boundaries.
- Add regression tests for each changed contract: overrides, run identity, checkpoint state, sample coverage, metric denominators, and resume behavior.
- Update the checklist only with evidence. Report changed files, tests actually run, tests not run, remaining risks, and the next permitted step.
