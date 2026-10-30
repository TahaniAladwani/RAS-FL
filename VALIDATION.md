# Release validation

Checked on 2026-10-30 using CPU execution with Python 3.13, PyTorch 2.11.0, torchvision 0.26.0, NumPy 2.3.5, Pillow 12.0.0, and matplotlib 3.10.6.

## Automated tests

`python check.py`: **51 passed, 3 skipped**. The skipped checks require CUDA.

- Core: 25 passed.
- Fashion-MNIST baselines: 6 passed, 1 skipped.
- CIFAR-10 baselines: 6 passed, 1 skipped.
- CIFAR-100 baselines: 7 passed, 1 skipped.
- Release planning and matched-budget selection: 7 passed.

## Synthetic integration checks

- RAS-FL training and deployment evaluation on all three datasets at alpha 0.1.
- Fashion-MNIST at alpha 0.4, including global/personalized compaction profiling and dynamic replay.
- All four training ablations, eight non-default sensitivity settings, the main reference, and four training-level sets on Fashion-MNIST at alpha 0.1: 17 runs completed.
- FIARSE and FjORD training, evaluation, and export on Fashion-MNIST.
- Adaptive and fixed-full-model replay on the same 320-request trace. Fixed selection used only the full model, left infeasible requests unserved, and recorded no budget violations.
- Resume skipped all 17 completed control runs and both completed baseline runs.
- Summary export preserved missing uncertainty for single-seed checks.
- Complete default plan: three datasets, two concentrations, three seeds, 450 training seed runs, with distinct output paths.

These checks use synthetic data. Full dataset training, CUDA execution, and reproduction of the paper's numerical results were not performed for this release. The code exports measurements from completed runs; the package supplies no pretrained checkpoints.

## Shared-patience update

All methods now use patience 5 by default, with one shared launcher override (`--patience`). The 25 core tests and 7 release/control tests passed after this change. The default 450-run plan was checked without training. The earlier synthetic integration checks used the preceding configuration; full experiments were not rerun.
