# CARM Execution Plan (HF-First 5-Way)

## 1) Scope (Current Cycle)

- Canonical dataset: `nbso/carm-vqa-5way`
- Backbone A: `Qwen/Qwen2.5-VL-7B-Instruct`
- Backbone B: `llava-hf/llava-v1.6-8b` (still stub in this cycle)
- Task focus: five-category multimodal action supervision

## 2) Five-Category Protocol (Locked)

- `C1`: clean image + clean caption -> `require_agreement`
- `C2`: clean image + different caption -> `abstain`
- `C3`: clean image + irrelevant caption -> `trust_vision`
- `C4`: irrelevant image + clean caption -> `trust_text`
- `C5`: irrelevant image + irrelevant caption -> `abstain`

Category mix target is balanced (`1/5` each) at protocol level.

## 3) Count Policy (Locked)

- Current realized release used in this cycle: `44,982` rows.
- Historical larger targets (`45,000`, `90,000`) remain planning references only.

Count disambiguation:
- Upstream retained clean-base stats (`150,582 / 15,207 / 18,801`) describe a prior official-data filtering stage.
- They are not the same artifact as the current HF release size.

## 4) Data Preparation and Splits

Primary prep CLI:
- `scripts/prepare_hf_5way_dataset.py`

Outputs:
- baseline-ready JSONL
- local materialized images for model runtime
- manifest with repo/revision/SHA, counts, split stats, drop reasons

Split policy in prepared outputs:
- deterministic base-level split assignment
- `train / val / test_id`
- seeded reproducibility via prep CLI arguments

## 5) Runtime Architecture

- Mock backbone path is removed from runtime code.
- Backbones are created through registry (`create_backbone`).
- Default runtime backbone is Qwen.
- LLaVA remains a declared but non-runnable stub in this cycle.

## 6) Evaluation and Reporting

- Baseline runner interface remains stable: `scripts/run_baselines.py`.
- Active baseline set is locked to:
  - `backbone_direct`
  - `agreement_check`
  - `confidence_threshold`
  - `probe_heuristic`
- Two-pass self-consistency is excluded from baseline runner outputs.
- Primary baseline comparison metric is `task_success` (category-first and outcome-based), not raw `accuracy`.
- Required baseline reporting artifacts are produced with `scripts/summarize_baselines_report.py` and include:
  - baseline table (`csv` + `md`)
  - `C2` diagnostic table (`csv` + `md`)
  - `risk_coverage_task_success` curve JSON
- Reported metrics must come from real-model runs.
- Lightweight default tests stay inference-free.
- Real Qwen inference tests are opt-in, and required before release reporting.
- Quest HF-first Qwen baseline reporting should request `gengpu` with `--gres=gpu:1`.

## 7) Defaults (Locked)

- Canonical HF dataset repo: `nbso/carm-vqa-5way`
- Canonical current release size: `44,982`
- Default cache root: `data/cache/hf_5way`
- Default backbone: `qwen2_5_vl_7b`
- Primary config: `configs/hf_5way_qwen.yaml`

## 8) Legacy Compatibility

Legacy local build/release scripts are retained as deprecated compatibility paths and are no longer the primary workflow.

## 9) Anti-Drift Rule

`PLAN.md` is the source of truth for active defaults and protocol locks.
Any change to dataset source, count lock, backbone defaults, or evaluation gates must be reflected here in the same change set.
