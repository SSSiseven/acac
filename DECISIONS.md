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

The modified ACAC source is included under `acac/` in the project Git
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

## D013: Flatten the vendored ACAC source to one `acac` directory

Move the upstream project files and Python package contents from the former
three-level repeated hierarchy into the repository's single `acac/` directory.
Keep the public `acac.*` import namespace by mapping that directory as the
`acac` package in `setup.py`, and make the entry point add the research
repository root to `sys.path`.

Reason:
The repeated directory name obscured source and working-directory locations.
One directory makes paths unambiguous while retaining the original import API,
training entry point, environment dynamics, rewards, macro actions, and
algorithm behavior.


## D014: Add one Explore task type with four directional instances

User approved the recommended resolution of D012. Create four nodes sharing
TaskType.EXPLORE, mapped to original right/down/left/up macros. Encode
direction with a four-way one-hot feature. Use robot position and static
walkable_positions metadata for feasibility. Absent topology masks Explore.
Do not read positions of entities outside local observation.

Reason: directional movement allows the robot to reveal invisible targets
without changing the original environment or macro executor. Explore is the
task type; each direction is a concrete task-node instance. No exploration
reward or new low-level navigation is introduced. The initial action-coverage
blocker is resolved; success of a learned policy remains to be tested.


## D015: Pause Stage 3 on missing retrieval and assembly operations

Original V1 get-ingredient macros also retrieve chopped food from a knife.
Get-plate macros can add chopped food to a partial plate. Current builder
completion rules mask both operations. Fixed-state execution verified
original-macro success against a false robot-task mask for both cases.

Following the project stop-on-semantic-conflict rule, pause neural integration.
Proposed resolution, not implemented: distinguish initial acquisition from
post-chop retrieval and plate assembly, retain original macro mapping, and
separate milestone completion from reusable interaction feasibility.
Add complete-chain regression coverage before training integration.


## D016: Resolve retrieval and assembly with two explicit task types

User approved CollectChoppedIngredient and AddToPlate. Add three collection
instances and six ingredient/plate assembly instances for the V1 recipe.
Use original get-ingredient and get-plate macros; retain raw-acquisition tasks
as separate milestones. A plate with contents is not considered unavailable
for empty-hand acquisition. Chop requires empty hands.

Original get-plate execution deposits a chopped ingredient and then retrieves
the plate. The graph therefore supports subsequent direct collection of
other chopped food into a held partial plate, with another robot cooking.
Dependency edges describe alternative knives and plates rather than mandatory
AND prerequisites. Feasibility remains based on locally observable state.

Validation: 24 unit tests pass, including a full original-environment
cooperative delivery chain. This resolves D015; policy learning remains
unverified. Original environment, rewards, navigation, and ACAC are unchanged.


## D017: Store selected task indices for the unchanged PPO learner

The TG controller samples a task node and maps it to an original macro action.
The TG runner validates this mapping and replaces only the actor action index
in stored experience with the task index. Busy robots retain the last selected
task while the original worker continues the macro. PPO therefore optimizes
the exact selected-task likelihood, including tasks sharing one macro action.
The original buffer schema, critic inputs, asynchronous GAE, start/end filters,
and PPO update formulas are reused.

Actor graphs are deterministically rebuilt from recorded local observations.
Unknown current-action and duration graph features retain contract defaults;
they are not filled from live environment state. The original GRU retains its
input dimension and ready-only update rules. Padding selects only Wait and
uses finite masked log-probabilities to avoid inf-times-zero in original PPO.

The implementation uses separate TG actor/controller/runner classes and an
ACAC construction factory hook whose defaults remain MAC and EnvsRunner.
Stage 3 supports V1 map A only; unsupported maps and obs_last_action=True
fail explicitly. V0 scale and map B coverage remain subsequent work.


## D018: Add one task-dependency message step after NoDependency validation

NoDependency completed its shape/mask/gradient tests and a 100-episode
(20,000-step) ovcA7 smoke before dependency propagation was enabled.
Full TG uses one predecessor-to-successor message step. Each message combines
the predecessor embedding with observable dependency-edge features. Active
messages are degree-normalized and added to the successor embedding before
robot-task scoring. The feasibility mask is identical for both TG variants.

tg_dependency_steps=0 is required for tg_acac_no_dependency;
tg_dependency_steps=1 is required for tg_acac. Both share tg_embed_dim=32
by default. A removed/disabled message produces the exact original task
embedding. Tests verify dependence of feasible-task probabilities on messages
and finite nonzero gradients to the dependency network.

Smoke tests use CPU, four environments, train_freq=4, and evaluation every
20 episodes with four evaluation episodes. These are short debugging settings
only; official pilot settings remain unchanged. total_epi continues to mean
env_terminate_step times total_epi environment-step budget.


## D019: Match ACAC-MLP using only its post-GRU decision head (2026-10-08)

Preserve original local observations, GRU, time embedding, 14-way macro policy,
availability mask, epsilon policy, runner and learner. Enlarge only fc3,
LayerNorm and fc4. MLP forward never accesses task features or graph masks.

Automatically choose the closest trainable Actor count to full TG using the same
encoder and tg_embed_dim. The reference uses static architecture metadata only,
with CPU RNG state preserved; it is never evaluated for MLP action selection.
mlp_head_width=0 means automatic. Manual widths must satisfy
mlp_match_tolerance=0.01. Current ovcA7/ovcAR7 width is 126.

Per Actor: MLP 14,700 vs full TG 14,688. Three-Actor difference is 36.
Target copies and frozen embeddings are excluded. No dummy parameters.
NoDependency retains its naturally smaller count without the dependency module.
The new algorithm/controller reuse the existing construction hook; original
ACAC models, critic, environment, macros, buffer, PPO and GAE remain unchanged.
Matching currently uses the static V1 map-A contract.

## D020: Profile common inputs and explicit cache/timing scopes (2026-10-08)

Inference: batch=1, three ready robots, 64 original random-macro rollout
observations (seed 17), five repeats, actor seed 0, time index 1, epsilon 0.05.
CPU and synchronized GTX 1050 Ti GPU timings include device transfers,
graph/cache work, GRU, head, mask and sampling. Cold/warm timings are separate.
Cache clearing is excluded; warm cache is primed with the entire input pool.

CPU training: identical seed 0, 3,200-step target, n_env=4, train_freq=4,
eval_freq=8, eval_num_epi=2, and original checkpoint saving. All methods finish
16 episodes / 3,200 steps. Rate is steps / sum(SamplingTime+TrainingTime).
Wall rate includes setup, evaluation and saves. This is one short cost run per
method; it is not a learning comparison or precise long-run forecast.
GPU training throughput/memory remain unmeasured.

The initial profiling command used an unsupported max_checkpoint CLI option;
it was rejected before training. Remove the profiling option and retain logs,
rather than changing original parser/saving defaults. The initial hand estimate
of MLP width 131 was corrected to measured nearest width 126 in the test.


## D021: Opt-in evaluator diagnostics and delivery repair (2026-10-09)

Use PilotMetricsWrapper outside the existing macro environment. It returns the
same observations, rewards and terminal flags, adds delivery metadata, and
keeps scalar diagnostic counters separate from the learner buffer.
The V1 environment and macro executors remain byte-for-byte unchanged.
The worker receives shelf_deliveries as expected; get_info transports optional
episode metrics. Runner records new arrays. Fix logger classification to match
Agent0/... prefixes, not every key containing Agent, so Train/AgentSteps logs.

Success is the existing correct-dish termination plus matching delivery state;
neither rewards nor navigation are altered. Historical statistics remain untouched.
Replay existing seed-0 baseline weights: delivery [0,0] becomes [1,1], while
returns stay [226.6,226.6] and episode lengths/discounted returns are identical.

Use common local task feasibility for selected-macro invalid rates. TG alone
also exposes detached raw pre-mask probability mass before epsilon mixing.
No gradient or policy changes; neural parameter counts are unchanged.
Resource contention uses centralized state only in the observer.
Wait ratio is explicit macro waiting, not all nonproductive physical behavior.
Detailed definitions are in EXPERIMENT_PROTOCOL.md.

Preserve failed diagnostics: empty plate contents are None and caused the first
observer membership check to fail; fixed observer and added a regression test.
The next run completed but logger dropped Agent-named counters; fixed logger
prefix matching. Failed/incomplete logs and CSVs remain in their original paths.

## D022: Common AUC support and short CPU/GPU resource calibration (2026-10-09)

Choose the protocol-permitted midpoint pilot budget of 25%: 10M env steps
(total_epi=50000). This is a proposed plan; no long pilot is started.
Use unchanged n_env=8/train_freq=8/eval_freq=200/eval_num_epi=20 for planned runs.
Short calibration uses target 32*200 steps and eval_freq=32; preserve actual
step counts if original budget accounting overshoots.

AUC uses a shared recorded interval, with no step-zero invention. Pooled rates
use raw numerator/denominator counters weighted by actual eval episode counts.
Pre-mask mass is TG-only and is not directly compared to a 14-macro policy.

Resource calibration includes diagnostics for all methods, CPU/CUDA on the same
machine, measured training vs process wall time, process-tree RSS, output size,
and CUDA peak allocated/reserved memory. Extrapolations are planning estimates,
not confidence bounds; long-run readiness and episode length can change costs.


## D023: Shared pilot container and durable outputs (2026-10-10)

Pin official Python 3.9.25 slim-bookworm digest and PyTorch 2.1.0+cu121.
Use validated direct dependencies instead of the unrelated legacy setup.py bundle.
Run source from /workspace with headless SDL and one OMP/MKL thread.
All four algorithms share one image; default CMD prints help.
ACAC_DATA_DIR=/data will use the n705/fcc RWX PVC. Build context excludes outputs,
checkpoints, Git metadata and credentials. Direct pins are not a full transitive
or apt lock: store resolved versions and reuse a single immutable image digest.
Build-time unit/import gates require no GPU; actual GPU access/timing need runtime
checks. No image build is claimed because the docker CLI is unavailable.
