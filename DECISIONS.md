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

## D008: Keep a standalone research repository

The modified ACAC source is included under `acac/acac/` in the project Git
repository. The official source is attributed in `upstream/ACAC_SOURCE.md`.
Experiment outputs, bundled official results, paper PDFs, and the original
checkout Git metadata remain local and outside version control. The GitHub
remote is `https://github.com/SSSiseven/acac.git`; no push was performed.

Reason:
A single clone can run the modified ACAC source without depending on an
unpublished submodule commit.

## D009: Apply seed 0 in the ACAC entry point

Treat a configured seed as present when it is not `None`, including integer 0.
This changes only the truthiness guard in `scripts/acac_main.py`; it leaves
nonzero seed behavior and the existing worker seed offsets unchanged.

Reason:
The official README documents `--seed=0`, and released ACAC result files
include `ovcA7/acac/0.csv` and `ovcAR7/acac/0.csv`. The old guard skipped
Python, NumPy, and PyTorch main-process seeding for zero.

## D010: Interpret the original ACAC budget in environment timesteps

Preserve the original calculation `max_timesteps = env_terminate_step *
total_epi` for ACAC and all later comparison methods. Compare learning curves
on environment timesteps. Treat the logged `Episodes` value as a diagnostic,
not the training budget.

Reason:
Successful policies terminate `ovcA7` episodes much earlier than the 200-step
limit, so `total_epi=100000` produced roughly 343k to 374k completed episodes
before reaching 20M steps. The released official result CSVs also terminate at
approximately 20M timesteps. Changing the stop condition to 100k completed
episodes would shorten strong-policy runs and break reproduction and fairness.

## D011: Build task graphs from egocentric observable state

The pure builder accepts decoded immutable observations and static metadata,
not an Overcooked environment instance. Hidden entities carry IDs and types but
no position or dynamic state. A V1 adapter reconstructs only relationships that
are visible through co-located observation slots. Teammate readiness and current
macro action are unknown unless they belong to the perspective robot.

Reason:
Reading `env.agent`, `env.knife`, or other live environment objects would expose
state outside the actor's local macro observation and invalidate decentralized
execution.

## D012: Pause neural integration at the exploration-task conflict

Do not connect the current six-type task graph to ACAC training yet. With hard
visibility masks and no movement task, real initial observations can leave an
agent with Wait as its only action, and Wait cannot reveal new targets.

Reason:
The original V1 policy has four primitive movement macros in addition to the
semantic macros. Removing those actions changes reachability and can deadlock
the task policy. The task vocabulary or visibility rule needs an explicit
project decision before TG-ACAC-NoDependency is implemented.
