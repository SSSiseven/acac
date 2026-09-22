# Current Status

## Current phase

ACAC baseline smoke verification and reproduction preparation.

## Completed

- ACAC repository cloned
- Current conda environment created
- ovcA7 seed 1, 16-episode smoke run completed (reported by user; local artifacts verified)

## Not completed

- Three-seed ACAC reproduction
- Task graph builder
- TG-ACAC
- Pilot comparison

## Current blocker

The smoke run used Unix `cp` for checkpoint history rotation on Windows. The
latest checkpoint is readable, but older revisions were not saved. The code
now uses Python `copy2`; the change has passed a rotation check but has not
yet been exercised in a new training run. Seed 0 reproducibility also needs
attention before the three-seed baseline.

## Next action

Inspect the repository and produce:
1. execution call graph;
2. environment observation/action specification;
3. minimal modification plan.
