# LOG.md - Decision Record

Purpose: chronological decision-and-reasoning record for protocol, implementation, and documentation changes.

Last updated: 2026-03-18 21:35 UTC

## Entry Template
- Date (UTC): YYYY-MM-DD HH:MM UTC
- Decision ID: DEC-XXXX
- Area: data | splits | baselines | eval | infra | docs
- Context: Why this decision is needed now.
- Options considered: Enumerate concrete alternatives.
- Decision: Chosen option.
- Rationale: Why the chosen option is better.
- Impact on PLAN/README/configs: Reference exact files and sections changed.
- Follow-up actions: Concrete next steps.

## Decision Log

### DEC-0016 Full-Dataset Promotion for Paper-Grade Evidence
- Date (UTC): 2026-03-18 21:35 UTC
- Decision ID: DEC-0016
- Area: docs
- Context: The repo had drifted into treating the archived 10 percent subset as the de facto paper target, even though the current project goal is stronger, top-paper-style evidence. At the same time, the user explicitly approved auto-proceed behavior, allowed long Quest jobs, requested that established experiments move to full data, and asked that genuinely new ideas be gated on small or tiny pilots first.
- Options considered: 1) Keep the paper centered on the 10 percent subset and only use full-data artifacts opportunistically. 2) Move every idea, including new ones, directly to full data. 3) Promote established experiment families to the canonical full split, keep new ideas on tiny or small pilots until they earn scale, and treat any count-mismatched historical full artifact as provisional until refreshed.
- Decision: Adopt option 3. The active paper track now targets the canonical full prepared split (`31,463 / 6,743 / 6,742`), established experiment families may run at full scale immediately, new ideas must first pass a pilot gate, human-label collection remains skipped for this cycle, and stale or split-mismatched full-data artifacts cannot support final paper claims.
- Rationale: This best matches the stated quality bar. It strengthens reviewer-facing evidence without wasting large-GPU budget on speculative variants, and it avoids overstating results from artifacts that do not cleanly align with the current canonical manifest.
- Impact on PLAN/README/configs: Updated `PLAN.md`, `PIPELINE.md`, `REPORT.md`, and created fresh state documents for the auto-proceed paper track. Follow-on config additions define canonical full-data learned runs for the established experiment family.
- Follow-up actions: 1) Refresh full-data threshold tuning and non-learned baselines on the current prepared split. 2) Launch full-data `Cascade`, `Flat Hidden`, and distributional learned runs in parallel. 3) Rewrite the manuscript around refreshed full-data evidence if those runs preserve the core claims.

### DEC-0015 Local-Cache-First Multimodal Preflight and Non-Ablation 10 Percent Closeout
- Date (UTC): 2026-03-11 05:38 UTC
- Decision ID: DEC-0015
- Area: infra
- Context: After `RUN-EXP-0004` failed, the stronger inspection showed the real blocker was unauthenticated HF Hub metadata rate limiting on the GPU node rather than a general loader incompatibility. `NEW_PLAN.MD` still required a valid structured 10 percent rerun, an old-head control on the same subset, and the non-ablation figure bundle.
- Options considered: 1) Keep relaunching the full 10 percent run and hope the transient HF requests succeed. 2) Fall back to CPU or a fake-debug path and accept weaker evidence. 3) Patch the Qwen loader to prefer local cache first, require a short real multimodal GPU preflight, relaunch the structured run and old-head control only after that preflight succeeds, then close the non-ablation stage with a frozen analysis bundle while explicitly deferring ablations.
- Decision: Adopt option 3. Update the Qwen loader to prefer local cached artifacts, strengthen the opt-in multimodal smoke test, require `RUN-EXP-0006` as the real multimodal GPU preflight, relaunch the structured four-head run as `RUN-EXP-0007`, run the old action-only control as `RUN-CTRL-0001`, and generate the non-ablation analysis bundle `outputs/analysis/RUN-ANALYSIS-0001_10pct_protocol/`. Hold ablations until after this analysis pass.
- Rationale: This preserved the real GPU/Qwen path, avoided wasting another full allocation on an avoidable HF `429` failure mode, and produced a clean apples-to-apples comparison on the corrected protocol-category-stratified subset before any ablation expansion.
- Impact on PLAN/README/configs: Updated `carm/models/backbone.py`, `carm/models/registry.py`, `tests/test_qwen_inference_optin.py`, `scripts/run_experimental_small_data.py`, `scripts/run_carm_quest.sh`, `scripts/submit_carm_quest.sh`, `scripts/plots/plot_new_plan_10pct_analysis.py`, `REPORT.md`, `README.md`, and `WRITEUP.md`.
- Follow-up actions: 1) Inspect the generated figures and failure examples in `outputs/analysis/RUN-ANALYSIS-0001_10pct_protocol/`. 2) Decide whether to improve features, loss weighting, or supervision before running any ablations. 3) Resume the held ablation panel only after that decision.

### DEC-0014 Explicit Four-Head Experimental Model and Protocol-Category-Stratified 10 Percent Subset
- Date (UTC): 2026-03-11 03:14 UTC
- Decision ID: DEC-0014
- Area: eval
- Context: While advancing `NEW_PLAN.MD`, the first 10 percent experimental submission exposed two protocol mismatches: the learned path was still using the older `joint_info + reliability + action` structure instead of explicit `vision_info_state`, `text_info_state`, `pairwise_relation`, and `joint_action` heads; and the initial 10 percent subset had been stratified on the older joint-info buckets, which under-sampled C1/C2 relative to the revised benchmark design.
- Options considered: 1) Let `RUN-EXP-0003` continue and interpret it as an approximate run. 2) Patch only the model heads but keep the skewed 10 percent subset. 3) Patch the experimental labels/model/training/evaluation path to the explicit four-head contract, regenerate a protocol-category-stratified 10 percent subset, invalidate the earlier run, and resubmit with a fresh frozen spec.
- Decision: Adopt option 3. Replace the experimental path with explicit vision-info, text-info, relation, and action heads; keep joint-info only as a derived convenience field for audit/sampling compatibility; add `protocol_category_family` sampling; regenerate the corrected 10 percent subset `carm_vqa_5way_10pct_protocol_family_seed7.*`; invalidate and cancel `RUN-EXP-0003`; and submit `RUN-EXP-0004` as job `2390788` using `configs/experimental_10pct_qwen_protocol_family.yaml`.
- Rationale: This is the first point where the code actually matches the revised semantics rather than just the prepared data. Canceling the earlier run avoids spending GPU time on an invalid spec and keeps the result ledger honest.
- Impact on PLAN/README/configs: Updated `carm/experimental/labels.py`, `carm/experimental/model.py`, `carm/experimental/training.py`, `carm/experimental/evaluation.py`, `carm/experimental/sampling.py`, `scripts/run_experimental_small_data.py`, `scripts/sample_prepared_dataset.py`, `configs/experimental_small_debug.yaml`, `configs/experimental_small_qwen.yaml`, `configs/experimental_10pct_qwen.yaml`; added `configs/hf_5way_qwen_caption_derived_10pct_protocol_family.yaml`, `configs/experimental_10pct_qwen_protocol_family.yaml`, and updated tests plus `REPORT.md`.
- Follow-up actions: 1) Monitor `RUN-EXP-0004` job `2390788`. 2) If it completes cleanly, backfill metrics and acceptance status in `REPORT.md`. 3) If runtime or memory still regress on Quest, precompute backbone features for the 10 percent subset rather than relaxing the semantic contract again.

### DEC-0013 Frozen 10 Percent Experimental GPU Spec and Quest Submission Path
- Date (UTC): 2026-03-11 02:54 UTC
- Decision ID: DEC-0013
- Area: infra
- Context: `NEW_PLAN.MD` moved the next stage to the 10 percent prepared subset with revised modality-target fields, and `AGENT.md` requires a frozen run spec plus recorded artifacts before a GPU launch. The existing Quest submission wrapper was hardcoded to the older 15-example small-data Qwen run.
- Options considered: 1) Reuse the old hardcoded Quest script and manually edit paths for the 10 percent run. 2) Stop for another CPU preflight even though the repo already has `RUN-EXP-0001` debug preflight and `RUN-EXP-0002` small real-Qwen execution artifacts. 3) Freeze the 10 percent dataset manifest, add a dedicated semantics-audit artifact for Step 1 of the plan, parameterize the Quest submission wrapper, and submit the 10 percent Qwen run directly.
- Decision: Adopt option 3. Add `scripts/audit_prepared_dataset_semantics.py`, update `configs/experimental_10pct_qwen.yaml` with bounded backbone cache settings, parameterize `scripts/submit_experimental_quest.sh`, generate the 10 percent audit artifacts under `outputs/experimental/RUN-EXP-0003_10pct_qwen/`, validate the Slurm command with `--test-only`, and submit `RUN-EXP-0003` on Quest as job `2390150`.
- Rationale: This satisfies the repo’s GPU governance without another redundant CPU loop, makes the 10 percent label contract explicit before training, and removes one-off shell hardcoding from the experimental Quest path.
- Impact on PLAN/README/configs: Added `scripts/audit_prepared_dataset_semantics.py`; added `tests/test_audit_prepared_dataset_semantics.py`; updated `scripts/submit_experimental_quest.sh`; updated `configs/experimental_10pct_qwen.yaml`; updated `REPORT.md` with the frozen 10 percent experimental run row and artifact paths.
- Follow-up actions: 1) Monitor `RUN-EXP-0003` job `2390150` until completion. 2) Backfill `REPORT.md` with the finished metrics and acceptance status. 3) Decide whether the next code step is upgrading the experimental head from `joint_info + reliability` to the fully explicit four-head formulation in `NEW_PLAN.MD`.

### DEC-0012 Opt-In Small-Data Experimental Path for Revised Evidence Semantics
- Date (UTC): 2026-03-10 06:02 UTC
- Decision ID: DEC-0012
- Area: eval
- Context: `NEW_PLAN.MD` requested a fast-iteration path that audits whether the current HF-backed repo can support revised semantics separating supportive, irrelevant, and contradictory evidence, without disturbing the locked Phase A HF-first defaults.
- Options considered: 1) Rewire the existing baseline/CARM scripts in place and risk breaking the locked Phase A path. 2) Add a separate opt-in experimental path with its own centralized label derivation, small-run sampling, masked supervision, and reporting artifacts while leaving the default pipeline untouched.
- Decision: Adopt option 2. Add `carm/experimental/*`, `scripts/run_experimental_small_data.py`, `configs/experimental_small_debug.yaml`, and a deterministic debug backbone for CPU-safe preflight. The experimental derivation uses a conservative reduced label space of five observed joint modality states, masks revised action supervision on contradiction rows, writes full-schema audit artifacts before training, and emits a small-run preflight under `outputs/experimental/RUN-EXP-0001_small_debug_preflight/`.
- Rationale: The current dataset does not support the full 3x3 modality-state grid or a defensible revised action target for contradiction rows. A separate experimental path makes those limitations explicit, preserves backward compatibility for the locked scripts, and still validates the revised logging/training/eval path end to end.
- Impact on PLAN/README/configs: Adds `carm/experimental/`, `carm/models/debug_backbone.py`, `scripts/run_experimental_small_data.py`, `configs/experimental_small_debug.yaml`, and tests for the new path. `PLAN.md` remains unchanged because this is an opt-in experimental preflight rather than a lock change.
- Follow-up actions: 1) Decide the revised contradiction action semantics explicitly instead of inheriting legacy `C2 -> abstain`. 2) Run the same script with a real GPU-backed backbone once hardware is available. 3) If template-family leakage analysis is required, restore template IDs into the prepared export.

### DEC-0011 Bounded Backbone Cache for the CARM Rerun
- Date (UTC): 2026-03-07 18:11 UTC
- Decision ID: DEC-0011
- Area: infra
- Context: The first full CARM rerun (`RUN-0008-train`) was OOM-killed under the runtime-normalized config while using unlimited Qwen backbone result caches. A pure `cache_results=false` fallback would be safer, but it would also discard the short-range reuse that helps repeated multimodal/probe calls during evaluation and training.
- Options considered: 1) Disable caching entirely for the rerun. 2) Keep unlimited caches and only increase job memory. 3) Bound the backbone caches, clear them between train and validation phases, keep `batch_size=1`, and rerun with higher host memory.
- Decision: Add `cache_max_entries` support to the Qwen backbone adapter, expose it through `carm/models/registry.py`, clear caches between train/val phases in `carm/train/trainer.py`, emit per-epoch JSON progress lines, and use `configs/hf_5way_qwen_caption_derived_lowmem.yaml` with `cache_results=true`, `cache_max_entries=512`, `batch_size=1` for `RUN-0011`.
- Rationale: This keeps the runtime bounded and observable without reverting to the earlier unbounded-cache failure mode. The capped cache preserves the useful local reuse while preventing the job from accumulating a full-corpus cache in memory.
- Impact on PLAN/README/configs: Updated `carm/models/backbone.py`, `carm/models/registry.py`, `carm/train/trainer.py`, `configs/hf_5way_qwen_caption_derived_lowmem.yaml`, and the README train/eval instructions for the refreshed rerun path.
- Follow-up actions: 1) Monitor `RUN-0011-train` for epoch-level progress and memory behavior. 2) Keep `RUN-0008` marked failed/superseded in `REPORT.md`. 3) If `RUN-0011` still OOMs or stalls, move to explicit feature precomputation rather than re-expanding cache size.

### DEC-0010 Caption-Derived Partial C2 Text Targets With Explicit Coverage Accounting
- Date (UTC): 2026-03-07 18:02 UTC
- Decision ID: DEC-0010
- Area: data
- Context: The live `nbso/carm-vqa-5way` revision still does not ship an explicit C2 `text_supported_target`, but the previous runtime-normalized fallback left all C2 text diagnostics unavailable. The user approved deriving this field from the contradictory caption itself rather than from a model prediction.
- Options considered: 1) Keep waiting for HF to publish explicit C2 text targets and leave all C2 text diagnostics unavailable. 2) Force exact targets for every C2 row via aggressive heuristics or hand-curated overrides. 3) Derive `text_supported_target` conservatively from the perturbed caption with family-specific rules, store it only when the caption supports an answer, leave it null otherwise, and report the exact evaluable denominator in manifests and C2 diagnostics tables.
- Decision: Adopt option 3. `scripts/prepare_hf_5way_dataset.py` now derives C2 text targets from the contradictory caption when possible, writes `metadata.text_supported_target_source` as `derived_from_caption_rule` or `missing_after_caption_rule`, and emits `data/cache/hf_5way/prepared/carm_vqa_5way_caption_derived_20260307.jsonl` plus `...manifest.json`. On the current HF revision, this yields `8,931 / 8,992` C2 text targets (`99.32%`) and leaves `61` rows explicitly missing rather than fabricated.
- Rationale: This restores real C2 text-only diagnostics for almost the entire contradiction split without circular labeling or model-leakage. The remaining misses are concentrated in captions that do not specify an exact count or color, so preserving nulls is more honest than forcing guesses.
- Impact on PLAN/README/configs: Added `configs/hf_5way_qwen_caption_derived.yaml` and `configs/hf_5way_qwen_caption_derived_lowmem.yaml`; updated `scripts/prepare_hf_5way_dataset.py`, `carm/eval/evaluator.py`, `carm/eval/metrics.py`, `scripts/summarize_baselines_report.py`, and the related tests so C2 reports now include evaluable denominators instead of all-or-nothing `n/a`.
- Follow-up actions: 1) Rerun threshold tuning and final baselines on the caption-derived manifest. 2) Rerun CARM with the low-memory config (`cache_results=false`, `batch_size=1`). 3) Record the refreshed run IDs, metrics, and remaining 61-row C2 text-target gap in `REPORT.md`.

### DEC-0009 Runtime-Normalized HF Fallback for Execution While Upstream C2 Targets Are Missing
- Date (UTC): 2026-03-07 08:21 UTC
- Decision ID: DEC-0009
- Area: data
- Context: The active `nbso/carm-vqa-5way` HF revision still does not expose an explicit C2 text-supported target field. The finalized protocol forbids fabricating that text-only truth from captions or other proxies, but the requested train/eval reruns still need protocol-correct action labels and reproducible input artifacts.
- Options considered: 1) Stop execution entirely until the HF dataset is republished. 2) Fabricate `text_supported_target` from perturbed-caption heuristics. 3) Create a separate local fallback export that rewrites stale oracle actions from protocol category, sets only the non-controversial C2 `vision_supported_target = gold_answer`, leaves `text_supported_target` null, and marks the artifact as interim/non-final.
- Decision: Create `data/cache/hf_5way/prepared/carm_vqa_5way_runtime_normalized_20260307.jsonl` with companion manifest `data/cache/hf_5way/prepared/carm_vqa_5way_runtime_normalized_20260307.manifest.json`, and run the new GPU jobs against `configs/hf_5way_qwen_runtime_normalized.yaml`. Do not fabricate the missing explicit C2 text target.
- Rationale: This preserves protocol-correct C2 action supervision for training/evaluation, keeps the stale `RUN-0005` export untouched, and makes the interim limitation explicit instead of hiding it behind derived text labels.
- Impact on PLAN/README/configs: Added `configs/hf_5way_qwen_runtime_normalized.yaml`; created the runtime-normalized local JSONL/manifest pair; pending README/WRITEUP/REPORT updates describe this artifact as an interim execution fallback rather than the final protocol-complete release.
- Follow-up actions: 1) Use the runtime-normalized manifest for `RUN-0006`/`RUN-0007`/`RUN-0008`. 2) Mark C2 text diagnostics as unavailable for these fallback runs. 3) Replace this artifact once HF is republished with explicit `text_supported_target`.

### DEC-0008 Locked Validation Threshold Tuning and Frozen Threshold Artifacts
- Date (UTC): 2026-03-07 08:05 UTC
- Decision ID: DEC-0008
- Area: baselines
- Context: The thresholded baselines (`confidence_threshold`, `probe_heuristic`) were still using fixed config values at report time, which left the actual validation-to-test locking workflow undocumented and unenforced.
- Options considered: 1) Keep fixed thresholds in config and describe tuning only in prose. 2) Tune thresholds on `val` as a separate mandatory step, write a frozen artifact, and force final `test_id` runs to consume that artifact explicitly.
- Decision: Add `scripts/tune_baseline_thresholds.py`, select thresholds on `val` by maximizing `task_success` with tie-breakers of higher coverage then least aggressive abstention (`lowest confidence_threshold`, `highest probe_both_uncertain_threshold`), and add `scripts/run_baselines.py --tuned-thresholds-json` to override config thresholds and copy the frozen artifact into the run root.
- Rationale: This makes the locked workflow auditable, prevents silent drift between tuned and reported settings, and keeps threshold selection reproducible at the artifact level rather than relying on undocumented config edits.
- Impact on PLAN/README/configs: Added `scripts/tune_baseline_thresholds.py`; updated `scripts/run_baselines.py` to apply frozen overrides and auto-write report tables; pending README/WRITEUP/REPORT updates now describe `RUN-0006` as the validation tuning stage and `RUN-0007` as the frozen-threshold `test_id` stage.
- Follow-up actions: 1) Finish `RUN-0006` threshold tuning on GPU. 2) Launch `RUN-0007` with the resulting `tuned_thresholds.json`. 3) Record the chosen thresholds and validation metrics in `REPORT.md`.

### DEC-0007 Finalized C2 Protocol Storage and Runtime Semantics
- Date (UTC): 2026-03-07 07:58 UTC
- Decision ID: DEC-0007
- Area: eval
- Context: The revised five-category protocol required more than the earlier metric fix. The repo still needed an explicit dataset contract for C2 support targets, protocol-first evaluation semantics, and a safe way to keep stale stored `oracle_action` labels from corrupting train/eval targets.
- Options considered: 1) Leave C2 support truth implicit in metadata and trust stored labels. 2) Add explicit top-level support-target fields, make HF prep fail when the text-supported target is absent, prefer the new fields in evaluation, and normalize stale stored oracle actions from protocol category on load.
- Decision: Extend `ConflictExample` with optional top-level `vision_supported_target` and `text_supported_target`, make HF prep rewrite C2 oracle actions to `abstain` and hard-fail on missing explicit C2 text targets, prefer top-level targets in evaluation with legacy metadata fallback only for old rows, and normalize runtime `oracle_action` from `protocol_category` when loading examples.
- Rationale: This keeps headline `task_success` category-first and outcome-based, preserves backward readability for legacy rows, and prevents stale `C2=require_agreement` artifacts from leaking into action accuracy or CARM supervision.
- Impact on PLAN/README/configs: Updated `carm/data/schema.py`, `scripts/prepare_hf_5way_dataset.py`, `carm/eval/evaluator.py`, and the relevant tests; pending README/WRITEUP text now reflects the explicit C2 target contract and the read-only metadata fallback.
- Follow-up actions: 1) Regenerate prepared data once HF provides explicit C2 text targets. 2) Rerun baselines and CARM against the fully refreshed manifest. 3) Keep stale `RUN-0005` outputs marked as superseded.

### DEC-0005 HF-First Quest Qwen Baseline Submission Path
- Date (UTC): 2026-03-06 07:46 UTC
- Decision ID: DEC-0005
- Area: infra
- Context: The active HF-first baseline workflow needed a reproducible Quest submission path with fixed run IDs, captured commands, and governance updates. The originally requested `short` and `normal` queues reject `--gres=gpu:1` for this account on the current cluster configuration.
- Options considered: 1) Submit ad hoc `sbatch --wrap` commands manually and log results afterward. 2) Add checked-in Quest wrappers with fixed defaults, repo-local logging, and cluster-valid GPU partition settings.
- Decision: Add `scripts/submit_baselines_quest.sh` and `scripts/run_baselines_quest.sh`, use `gengpu` with `--gres=gpu:1` for both preflight and final runs, fix run roots to `RUN-0004` and `RUN-0005`, and align `PLAN.md`, `README.md`, and `REPORT.md` with the active HF-first Qwen baseline path.
- Rationale: The checked-in wrapper path reduces submission drift, guarantees command/env capture, and keeps the GPU request generic while staying valid for the current Quest partition layout.
- Impact on PLAN/README/configs: Updated `PLAN.md` baseline names and Quest note; updated `README.md` with Quest bootstrap and submission usage; updated `REPORT.md` with HF-first Quest run-ledger rows; added Quest submission scripts under `scripts/`.
- Follow-up actions: 1) Bootstrap `.venv` with `torch`, `torchvision`, `transformers`, `accelerate`, and `pytest`. 2) Submit `RUN-0004` via `bash scripts/submit_baselines_quest.sh preflight`. 3) Review preflight artifacts, then submit `RUN-0005`. 4) Backfill `job_id`, metrics, and acceptance status in `REPORT.md` after completion.

### DEC-0006 Quest Compute Runtime Compatibility Fixes
- Date (UTC): 2026-03-06 09:50 UTC
- Decision ID: DEC-0006
- Area: infra
- Context: The initial Quest preflight failed even after package bootstrap because the repo `.venv` had been created against a login-node-only `/usr/bin/python3.12`, Quest compute nodes defaulted to `python3.6`, some GPU nodes had no `git` on `PATH`, and the opt-in Qwen inference test was forcing CPU execution on a GPU allocation.
- Options considered: 1) Keep the original `.venv` and chase per-node path differences ad hoc. 2) Rebuild `.venv` against a shared compute-visible Python 3.12 binary, invoke the venv interpreter directly in the job runner, make git metadata capture best-effort, and let the real Qwen opt-in test use CUDA when available.
- Decision: Rebuild `.venv` with `/gpfs/software/bowtie2/2.5.4/bin/python3.12`, update `scripts/run_baselines_quest.sh` to call `./.venv/bin/python` directly and tolerate missing `git`, update `README.md` to document the compute-valid bootstrap path, and update `tests/test_qwen_inference_optin.py` to prefer CUDA.
- Rationale: This removes compute-node interpreter drift, prevents metadata capture from crashing the job, and turns the preflight into a real GPU-path validation instead of a slow CPU-only smoke.
- Impact on PLAN/README/configs: Updated `README.md` Quest setup section; updated `scripts/run_baselines_quest.sh`; updated `tests/test_qwen_inference_optin.py`; produced successful Quest runs `RUN-0004` and `RUN-0005`.
- Follow-up actions: 1) Keep using the shared Python 3.12 bootstrap path for Quest reruns. 2) Consider suppressing duplicate logger lines in `run.log`. 3) Optionally provide `HF_TOKEN` for faster authenticated Hub access on future runs.

### DEC-0004 Reproducible Vision Materialization and Subset Image Acquisition
- Date (UTC): 2026-02-28 19:27 UTC
- Decision ID: DEC-0004
- Area: data
- Context: Final-level runs require pixel-level vision corruption and reproducible data manipulations. Existing pilot data used metadata-only vision payloads, and local COCO image files were missing for many referenced examples.
- Options considered: 1) Download full COCO train/val image archives before materialization. 2) Add deterministic pilot-only materialization with optional subset image download for missing COCO files.
- Decision: Implement `scripts/materialize_vision_corrupt.py` with deterministic occlusion (`seed_key=example_id`), stable output naming, input/output hashing, optional image-directory fingerprinting, and optional missing-COCO subset download from official URLs; produce `pilot_3k_class_medium_real_vision` artifacts and reproducibility manifests.
- Rationale: This preserves practical class-project scale while making vision corruption concrete, reproducible, and auditable without requiring full COCO image archive downloads.
- Impact on PLAN/README/configs: Added `configs/class_medium_final.yaml`; updated README.md with materialization and release workflow; updated PLAN.md execution profile and priorities; recorded artifacts in REPORT.md RUN-0003.
- Follow-up actions: 1) Run baselines using `configs/class_medium_final.yaml` + `pilot_3k_class_medium_real_vision.jsonl`. 2) Run CARM and ablations on the same frozen final pilot. 3) Publish final artifacts/manifests to a versioned HF dataset repo if releasing externally.

### DEC-0003 Class-Medium Profile and Construction Throughput Patch
- Date (UTC): 2026-02-28 18:53 UTC
- Decision ID: DEC-0003
- Area: data
- Context: Full-scale Conflict Suite generation from the complete base set was too heavy for CS396 iteration speed and storage/runtime budget, and hard-swap donor selection was the dominant construction bottleneck.
- Options considered: 1) Keep full-scale-only workflow and tolerate long rebuild cycles. 2) Add a bounded profile for course-scale execution while preserving protocol logic and optimize hard-swap donor search implementation.
- Decision: Add `configs/class_medium.yaml` with `max_per_family=5000` and dedicated `*_class_medium` output paths; optimize `carm/data/construction.py` hard-swap candidate lookup via donor bucketing and token-cache reuse.
- Rationale: The bounded profile preserves dataset construction semantics and OOD structure while enabling practical iteration. The throughput patch preserves behavior and reduces construction time for medium/full suites.
- Impact on PLAN/README/configs: Added `configs/class_medium.yaml`; updated README.md with class-medium workflow and artifact counts; updated PLAN.md snapshot/priorities to include class-medium execution profile and construction optimization; recorded artifacts in REPORT.md RUN-0002.
- Follow-up actions: 1) Run baselines on `pilot_3k_class_medium`. 2) Run CARM ablation training on `pilot_3k_class_medium`. 3) Decide whether to backfill default-path full-suite rebuild or keep class-medium as primary class-project track.

### DEC-0002 Phase A Rebuild Implementation
- Date (UTC): 2026-02-24 19:31 UTC
- Decision ID: DEC-0002
- Area: infra
- Context: Phase A required execution of the rebuild plan with restored scaffold, config-first dataset pipeline, split integrity controls, pilot sampling, and baseline evaluation wiring.
- Options considered: 1) Continue with old scaffold behavior and patch minimally. 2) Restore scaffold and refactor modules/scripts/configs to match PLAN.md defaults and contracts.
- Decision: Restore scaffold and refactor to Phase A v2 protocol, including downloader, base ingestion, canonical suite generation, pilot sampling, required baselines, and adapter stubs.
- Rationale: Full alignment with PLAN.md reduces protocol drift, supports reproducible benchmark construction, and prepares clean interfaces for next-wave real model integration.
- Impact on PLAN/README/configs: README.md rewritten as runtime runbook; configs/default.yaml and cpu_local/cloud profiles updated; scripts and data/eval modules refactored to config-first contracts; smoke run recorded in REPORT.md RUN-0001.
- Follow-up actions: 1) Run full dataset build from official artifacts. 2) Populate REPORT.md with per-run baseline results at larger scale. 3) Start next-wave real adapter implementation under fixed interfaces.

### DEC-0001 Governance Artifacts Bootstrap
- Date (UTC): 2026-02-24 18:45 UTC
- Decision ID: DEC-0001
- Area: docs
- Context: The repository was reset to planning docs, and Phase A required enforceable governance records for results and decisions.
- Options considered: 1) Keep only ad-hoc notes in README. 2) Add structured REPORT and LOG artifacts with machine-checkable format.
- Decision: Add REPORT.md and LOG.md with fixed templates and automated contract validation.
- Rationale: Structured artifacts reduce drift, support reproducibility audits, and improve traceability for reviewer rebuttal.
- Impact on PLAN/README/configs: Adds REPORT.md and LOG.md contracts, updates README with documentation discipline, aligns with PLAN.md Section 8 and Section 13.
- Follow-up actions: 1) Populate one REPORT row after each run with artifacts. 2) Add new DEC entry for each protocol or design change. 3) Link result-motivated decisions to REPORT run IDs.
