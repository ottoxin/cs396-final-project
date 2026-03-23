# Paper Improvement Log

## 2026-03-18: Full-Dataset Promotion

### Decision

- The paper should now target full-dataset evidence for its main claims whenever refreshed runs complete successfully.
- The archived 10% results remain useful for pilot analysis and fast debugging, but they are no longer the preferred main table for a high-quality submission.

### Why

- The quality target is now paper-grade evidence rather than old-plan completion.
- Several full-data artifacts exist historically, but at least one baseline artifact does not match the current canonical manifest counts, so refreshes are required.
- The manuscript currently hard-codes the 10% split throughout the abstract, setup, results, appendix tables, and limitations.

### Immediate Writing Consequences

- Rewrite the abstract and introduction metrics paragraphs from full-data results once refreshed runs land.
- Replace the current `674`-example main results framing in Sections 4 and 5.
- Update dataset statistics and any appendix tables that foreground the 10% subset.
- Keep the 10% subset only as pilot or archival context unless the full-data evidence fails or materially diverges.

### Current Experimental Priorities

1. Refresh full-data threshold tuning and baseline evaluation on the current canonical manifest.
2. Launch full-data `Cascade`, `Flat Hidden`, and distributional learned variants.
3. Keep genuinely new experiments on tiny or small pilots first.

### Known Risks

- full-baseline example-count mismatch vs current canonical manifest
- archived `Flat Hidden` split mismatch
- Quest `gengpu` partition max wall time is 48 hours
- no second-backbone evidence until `LlavaNextAdapter` is implemented
