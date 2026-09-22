# Decision Log

## D001: Use ACAC as the codebase

Reason:
ACAC already implements agent-centric asynchronous history encoding,
asynchronous GAE and PPO.

## D002: Do not modify asynchronous GAE initially

Reason:
The proposed contribution is task-graph-conditioned allocation,
not a new asynchronous return estimator.

## D003: Start with Overcooked-Rand

Reason:
It is already supported by ACAC and minimizes integration cost.

## D004: Preserve the original macro executor

Reason:
ACAC and TG-ACAC must differ primarily in task representation and selection.

## D005: Implement graph construction before neural graph encoding

Reason:
Graph semantics must be independently testable.

## D006: Keep research outputs under the current project

The training entry point imports its adjacent ACAC source tree first. The
official code/config directory remains `PROJECT_DIR`; logs, checkpoints,
policies, snapshots, and optional W&B files use `DATA_DIR`, which defaults to
this research project's `data/` directory. `ACAC_DATA_DIR` may override the
output location. A standalone official checkout still defaults to its own
`data/` directory.

Reason:
The active Conda environment also has an editable ACAC installation at
`E:\myrepo\acac`, which caused the reported smoke run to write there.

## D007: Rotate checkpoint copies with Python

Replace shell `cp -rf` calls with `shutil.copy2` while preserving the existing
newest-to-oldest rotation order. This supports Windows and makes copy failures
visible to Python instead of silently continuing after a shell error.
