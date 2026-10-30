# Implementation references

The core code implements RAS-FL and protocol adaptations of the comparison methods using PyTorch and torchvision. Dataset architectures, optimizer settings, capacity definitions, and checkpoint selection are recorded in the experiment configurations.

The FIARSE and FjORD implementations are local adaptations. Their reviewed reference files and source hashes are retained in each dataset's `REFERENCE_PROVENANCE.json`.

- FIARSE: Wu et al., NeurIPS 2024. Reference implementation: https://github.com/HarliWu/FIARSE
- FjORD: Horvath et al., NeurIPS 2021. Reference implementation: https://github.com/flwrlabs/flower/tree/main/baselines/fjord

The framework figure is supplied with this project as `assets/framework.pdf`; the README displays a raster rendering of that file. Dependencies retain their respective licenses.
