# Research Pipeline Report

**Direction**: modality conflict handling for MLLMs with frozen-backbone arbitration  
**Current objective**: upgrade the paper from archived 10% evidence to refreshed full-dataset evidence  
**Date**: 2026-03-18  
**Mode**: auto proceed

## Quality Bar

- Aim for reviewer-grade evidence suitable for a strong paper, not just an old-plan completion.
- Prefer refreshed, same-manifest comparisons over convenient but stale artifacts.
- Treat the archived 10% results as pilot-scale evidence unless the full-data reruns fail or materially contradict the story.

## Current State

- The manuscript is still written around the 10% split and must be updated after the full-data runs complete.
- The canonical prepared dataset currently contains `44,948` rows with split counts `31,463 / 6,743 / 6,742`.
- The strongest completed learned result is still `RUN-EXP-cascade_10pct` with `task_success_revised=0.7344` on the archived 674-example test subset.
- Historical full-baseline artifacts exist, but at least one logged a test-example count inconsistent with the current canonical manifest and is therefore provisional.
- `llava-hf/llava-v1.6-8b` is not a reportable target yet because the adapter remains a stub.

## Active Policy

- Established experiments go to full data:
  - `Cascade CARM`
  - `Flat Hidden`
  - `Distribution CARM v1`
  - `Distribution CARM v2 (+wt)` if cluster wall time is acceptable
- New experiments stay on tiny or small pilots first.
- Human-label collection is skipped in this cycle by explicit user choice.

## Immediate Run Plan

1. Refresh full-data threshold tuning on `val`.
2. Refresh full-data non-learned baselines on `test_id` using frozen thresholds from step 1.
3. Launch full-data established learned runs in parallel on Quest.
4. Update the paper only after refreshed full-data artifacts land.

## Submitted Jobs

- `RUN-0014` -> job `3108359`: full-data val threshold tuning refresh
- `RUN-0015` -> job `3108364`: dependent full-data baseline eval after `RUN-0014`
- `RUN-EXP-cascade_full` -> job `3108360`
- `RUN-EXP-flat_hidden_full` -> job `3108362`
- `RUN-EXP-dist_full` -> job `3108363`
- `RUN-EXP-dist_full_v2` -> job `3108361`

Current queue state at submission time:

- all five main jobs are in `gengpu`
- `RUN-0015` is intentionally waiting on `afterok:3108359`

## Paper Risks To Watch

- full-baseline count mismatch vs current canonical manifest
- archived `Flat Hidden` split mismatch (`3,146` vs `3,148`)
- no second-backbone evidence yet
- no persistent disk feature cache, so long learned runs may approach the Quest 48-hour ceiling
