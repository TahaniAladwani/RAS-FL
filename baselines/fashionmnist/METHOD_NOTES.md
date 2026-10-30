# FIARSE and FjORD

These implementations adapt FIARSE and FjORD to the dataset architecture and evaluation protocol of this repository. Reference URLs and source hashes are recorded in REFERENCE_PROVENANCE.json.

FIARSE uses magnitude thresholding, the threshold-dependent gradient, and coordinate-wise aggregation of nonzero updates. Inactive coordinates are protected from optimizer updates. Active parameters are logical weights in dense tensors; they do not establish physical memory or FLOP savings.

FjORD uses nested prefix widths, shared batch-normalization affine parameters, separate running statistics at each width, and capacity-aware aggregation. Only locally visited statistic banks are aggregated.

Twenty clients occupy five fixed capacity tiers, with four clients per tier. FIARSE receives the corresponding FjORD active-parameter budget. This capacity assignment differs from the core comparison protocol. Reference removal therefore does not imply equal active parameters across all methods.

Checkpoint selection weights full validation accuracy and mean masked validation accuracy equally. Deployment clients are excluded from selection. The result files contain per-client counts, predictions, active parameters, and selected rounds; structure counts and FLOPs are not estimated here.
