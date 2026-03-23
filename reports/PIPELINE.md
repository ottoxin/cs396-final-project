# Codex Research Pipeline Quickstart

This repo is now running in an auto-proceed, quality-first mode aimed at paper-grade evidence.

## Prerequisites

From the project root:

```bash
cd /projects/p33196/kym9881/cs396-final-project
export PATH="/projects/p33196/kym9881/cs396-final-project/.venv/bin:$PATH"
source /projects/p33196/kym9881/cs396-final-project/scripts/runtime_env.sh
setup_hf_runtime_env /projects/p33196/kym9881/cs396-final-project
```

Quick environment checks:

```bash
codex mcp list
claude auth status
latexmk -v
scontrol show partition gengpu
```

Expected state:

- `claude-review` is enabled in Codex MCP
- Claude Code authentication is live
- `gengpu` is available with `MaxTime=2-00:00:00`

## Active 2026-03-18 Workflow

### Established experiments: go to full data

Use full data for the already-defined experiment family:

- refresh non-learned baselines on the current full split
- launch full-data `Cascade`, `Flat Hidden`, and `Distribution` variants in parallel
- update the paper only from refreshed full-data artifacts

Primary launch paths:

```bash
bash scripts/submit_experimental_quest.sh ...
bash scripts/submit_carm_quest.sh ...
```

### New experiments: tiny or small first

Any genuinely new idea should first use:

- `configs/experimental_small_debug.yaml`
- `configs/experimental_small_qwen.yaml`
- archived 10% protocol-family configs only as pilot-scale evidence

Only after a new idea shows signal on a small run should it be promoted to a full-data Quest submission.

## Project-specific execution notes

- This repo is configured for Northwestern Quest via SLURM, not local GPU runs.
- The current quality bar is top-paper-style evidence, not old-plan completion.
- Treat stale or count-mismatched full-data artifacts as provisional until refreshed.
- Use multiple parallel single-GPU jobs rather than a single multi-GPU training job; the current scripts are single-process and do not implement multi-GPU training.
- LLaVA remains out of scope for reportable runs until `LlavaNextAdapter` is implemented.

## Paper compilation

The writeup compiles locally with:

```bash
cd /projects/p33196/kym9881/cs396-final-project/writeup
latexmk -pdf -interaction=nonstopmode -outdir=.latex-build-acl carm_acl.tex
```

Current paper target:

- replace the manuscript's 10% main-results framing with full-data evidence if the refreshed full runs support the same claims

## Monitoring

Use:

```bash
squeue -u "$USER"
```

Inspect:

- `outputs/.../slurm-*.out`
- `outputs/.../slurm-*.err`
- `outputs/.../run.log`

## Notes

- The login node is not a reliable local-GPU target.
- Auto proceed is enabled: long-running jobs may be launched without waiting for another checkpoint.
- Long experiments are acceptable when they materially improve claim quality.
