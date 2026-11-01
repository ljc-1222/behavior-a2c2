# A2C2 implementation

The core implementation for the **BEHAVIOR-1K Task 1 (`picking_up_trash`)** correction project lives here. Start at the [top-level README](../README.md) for the research result and pipeline diagram.

- [Model and losses](../docs/method.md): `src/model.py`, `src/loss.py`
- [Dataset format and usage](../docs/usage.md): `src/dataset.py`, `scripts/create_dataset.py`, `scripts/mp4_to_parquet.py`
- [Offline evaluation and online wrapper](../docs/usage.md): `scripts/train.py`, `scripts/eval.py`, `src/online.py`, `scripts/serve_a2c2_b1k.py`
- [Task 1 results and diagnostics](../docs/results.md)
- [Upstream fork integration](../docs/integration.md)

The frozen Comet policy remains a separate submodule. Trained A2C2 weights and full simulator assets are not committed to this repository.
