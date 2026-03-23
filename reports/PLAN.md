# CARM Execution Plan (HF-First 5-Way)

## 1) Scope (Current Cycle)

- Primary objective: produce top-paper-grade evidence for modality-conflict arbitration in MLLMs, not just a course-project snapshot.
- Established experiment families may scale to the full prepared dataset immediately.
- New experiment ideas must first clear a tiny or small pilot before any full-data launch.
- Canonical dataset: `nbso/carm-vqa-5way`
- Canonical current prepared release: `44,948` rows
- Canonical current full split: `31,463` train / `6,743` val / `6,742` test_id
- Backbone A: `Qwen/Qwen2.5-VL-7B-Instruct`
- Backbone B (`llava-hf/llava-v1.6-8b`) remains non-runnable because the adapter is still a stub.
- Human-label collection is explicitly skipped for this cycle by user decision.

## 2) Five-Category Protocol (Locked)

- `C1`: clean image + clean caption -> `require_agreement`
- `C2`: clean image + different caption -> `trust_vision`
- `C3`: clean image + irrelevant caption -> `trust_text`
- `C4`: irrelevant image + clean caption -> `abstain`
- `C5`: irrelevant image + irrelevant caption -> `abstain`

Category mix target is balanced (`1/5` each) at protocol level.

## 3) Count Policy (Locked)

- Current canonical release used for the main paper track: `44,948` rows.
- Full paper tables should prefer the full split over the archived 10% subset whenever refreshed runs are available.
- The archived 10% protocol-family subset remains valid only for rapid iteration, smoke checks, and new-idea pilots.

## 4) Data Preparation and Splits

Primary prep CLI:
- `scripts/prepare_hf_5way_dataset.py`

Canonical prepared artifacts:
- `data/cache/hf_5way/prepared/carm_vqa_5way.jsonl`
- `data/cache/hf_5way/prepared/carm_vqa_5way.manifest.json`
- `data/cache/hf_5way/images`

Split policy in prepared outputs:
- deterministic base-level split assignment
- canonical split counts are `train=31,463`, `val=6,743`, `test_id=6,742`
- full-data learned runs should use the canonical prepared JSONL and explicit split caps equal to those manifest counts
- archived 10% subset artifacts are not the source of truth for the main paper after 2026-03-18

## 5) Runtime Architecture

- Backbones are created through registry (`create_backbone`).
- Default runtime backbone is Qwen.
- LLaVA remains declared in config but is not an active experiment target until a real adapter exists.
- Current backbone caching is in-memory only; there is no persistent on-disk feature cache yet.

## 6) Evaluation and Reporting

- Main paper tables must come from the current canonical prepared split and refreshed artifacts only.
- Historical full-baseline artifacts with example-count mismatches against the current manifest are provisional and must be rerun before they support paper claims.
- Non-learned baseline reporting should follow:
  - retune thresholds on the current `val` split
  - evaluate on the current `test_id` split with frozen thresholds
  - summarize with `scripts/summarize_baselines_report.py`
- Full-data learned comparison matrix for the current cycle:
  - `Distribution CARM v1`
  - `Distribution CARM v2 (+wt)` if it fits within cluster wall-time constraints
  - `Flat Hidden`
  - `Cascade CARM`
- Prompt-only abstain can be sourced from the learned-run baseline bundle on the same current test split.
- New ideas or architectural extensions (for example new feature families or new backbones) must first run on tiny or 10% pilot subsets.
- After the first full-data main runs complete, the next stability priority is additional same-split `Cascade CARM` seeds, not a second-backbone claim.

## 7) Defaults (Locked)

- Canonical HF dataset repo: `nbso/carm-vqa-5way`
- Canonical current release size: `44,948`
- Canonical full split sizes: `31,463 / 6,743 / 6,742`
- Default cache root: `data/cache/hf_5way`
- Default backbone: `qwen2_5_vl_7b`
- Default full-data baseline config: `configs/hf_5way_qwen_caption_derived.yaml`

## 8) Active Execution Order

1. Refresh full-data threshold tuning and non-learned baselines on the current prepared split.
2. Launch full-data established learned runs in parallel:
   - `Cascade CARM`
   - `Flat Hidden`
   - `Distribution CARM v1`
   - `Distribution CARM v2 (+wt)` if the 48-hour Quest ceiling is acceptable
3. Once the main full-data runs finish, decide whether the full paper should be written entirely around the full split or keep the 10% subset as an auxiliary pilot table.
4. If `Cascade CARM` remains strongest on full data, launch extra same-split cascade seeds.
5. Update the manuscript, rebuild the PDF, and run an external review pass.

## 9) Quality Bar

- Aim for reviewer-grade evidence suitable for a strong workshop, conference, or journal submission.
- Prefer fewer defensible claims over more speculative ones.
- Do not treat stale, split-mismatched, or partially comparable artifacts as paper-ready evidence.

## 10) Anti-Drift Rule

`PLAN.md` is the source of truth for active defaults and evidence policy.
Any change to dataset source, split scope, backbone defaults, quality bar, or evaluation gates must be reflected here in the same change set.
