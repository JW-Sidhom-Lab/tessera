# Changelog

## 0.1.7

- `lift_snv` / `lift_cna` (and thus `featurize` / `reconstruct` on non-GRCh37 input):
  coordinates that lift onto a non-canonical contig absent from the bundled GRCh37
  reference (e.g. `chrUn_gl000211`) are now dropped — like unliftable coordinates —
  instead of crashing the downstream sequence-context lookup. The liftover stats gain an
  `n_noncanonical` field reporting how many were dropped for this reason. Fixes a crash on
  GRCh38/hg38 inputs.

## 0.1.6

- `tessera.featurize` / `tessera.reconstruct`: new `batch_size` parameter (default 24),
  forwarded to the underlying model methods, to bound peak memory on large cohorts or
  samples with very large variant/segment bags. Results are identical regardless of
  batching (the bag is padded to the cohort maximum either way).
- `tessera.data.preprocessing.subsample_snv` / `subsample_cna`: new supported helpers —
  the per-sample variant/segment caps used to build the TESSERA pretraining data.
  `subsample_snv` keeps variants recurrent across the cohort (a driver/hotspot proxy)
  then random-fills the budget; `subsample_cna` keeps the largest-magnitude segments
  (with an optional `LOH` bonus). Both use the same column schema as `featurize`.

## 0.1.5

- `reconstruct()` reports the manuscript Ref/Alt joint SNV reconstruction accuracy
  (`snv_joint_accuracy`) alongside the alt-only accuracy.
