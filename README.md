# OPD — TV-Regulated On-Policy Distillation

Reference implementation for **TV-Regulated OPD (TV-OPD)**: *Direction Matters in
On-Policy Distillation*.

The student generates rollouts on-policy; a **remote teacher vLLM server** returns
per-token log-probs used as the supervision signal. The codebase is a fork of
[verl](https://github.com/volcengine/verl) — the OPD pieces are additive, everything
else (Ray HybridFlow controller, FSDP workers, vLLM rollout, Hydra config) is upstream.

## Method

The loss is a PPO clip loss over the teacher–student log-prob gap

```
Δ_t = log π_T(ŷ_t) − log π_θ_old(ŷ_t)
```

Raw OPD uses `A_t = Δ_t` directly. **TV-OPD discards the magnitude and keeps only the
direction**, rescaled by a single global coefficient:

```
A_t^TV = c_k · sign(Δ_t)
```

`c_k` tracks the total-variation distance between teacher and student, so updates
shrink as the two converge. The TV distance is estimated per token as
`d̂_t = [1 − exp(Δ_t)]_+`, averaged over active tokens into `D̂_k`, and smoothed by an
EMA `D̄_k = β·D̄_{k−1} + (1−β)·D̂_k`.

> **Naming:** `dctv` in the config and metric names is the development name for
> TV-OPD. They are the same method.

Two controllers for `c_k` are implemented:

| Controller | `c_k` | Config |
|---|---|---|
| **TV schedule** (paper Eq. 12, used for the reported results) | `clip( ((D̄_{k−1}+ε)/(D_ref+ε))^α , c_min, 1 )`, `D_ref` anchored at the first step | `opd_adv_mode=sign` + `opd_tv_sched_enabled=True` + `opd_tv_sched_target=adv` |
| **Gradient-norm variant** | `min(c_cap, D̄/M̄)` with `M̂ = ½·η·(G^TV)²` | `opd_adv_mode=dctv` |

Defaults match the paper: `β=0.95`, `α=0.5`, `c_min=0.1`, `ε=1e-5`, `target=adv`.

## Layout

```
recipe/gkd/teacher/       Teacher server (proxy.py + worker.py, vLLM backend, ZMQ REQ/REP)
verl/workers/actor/       OPD loss + TV controllers (dp_actor.py)
verl/trainer/ppo/         core_algos.py: compute_policy_loss_opd, advantage transforms
verl/trainer/config/      Hydra config; actor/actor.yaml documents every opd_* knob
train_script/             Training + fleet launch scripts
```

## Setup

- Python 3.12, `pip install -e .`, plus vLLM and Ray.
- 8 GPUs for one student run (tested on 8×H100 80GB and 8×B200 180GB).
- One additional node hosting the teacher server.
- Under `$DATA_ROOT`: the training parquet (DeepMath-103K or DAPO-17K) and
  `val_all/aime24.parquet`, `val_all/aime25.parquet`.

## Teacher server

The student never loads teacher weights — it makes one ZMQ round-trip per batch.
Start the server on a dedicated node with `recipe/gkd/teacher/start_server.sh`; the
student reaches it at `TEACHER_SERVER_IP:TEACHER_SERVER_PORT` (default 15555) with
`TEACHER_N_WORKERS` slots. Retries are handled in `recipe/gkd/teacher/client.py`.

## Running

```bash
export WANDB_API_KEY=<key>
export REPO_ROOT=/path/to/OPD
export DATA_ROOT=/path/to/data
export MODEL_PATH=/path/to/student/ckpt
export TEACHER_CKPT_PATH=/path/to/teacher/ckpt
export TEACHER_SERVER_IP=<teacher-host>
export SEED=42

bash train_script/ds_ds_dapo_tvopd_exp.sh      # TV-OPD, DS-1.5B pair, DAPO-17K
bash train_script/qwen_qwen_rawadv_exp.sh # raw-OPD baseline, Qwen3-8B pair
```

Every cluster-specific value comes in through an env var; the scripts contain no
absolute paths.

To fan out over several nodes sharing one teacher, copy
`train_script/fleet.example.txt` to `train_script/fleet.txt` (one line per node,
gitignored) and use the `launch_*.sh` drivers. They default to a dry run and need
`--go` to dispatch.

## Baselines

Only raw OPD and the TV-OPD variants are implemented here. The other baselines in the
paper are run from their own public repos and are not reimplemented in this tree.

## Config knobs

All under `actor_rollout_ref.actor.policy_loss`; documented inline in
`verl/trainer/config/actor/actor.yaml`.

| Knob | Effect |
|---|---|
| `loss_mode=opd` | Enables the OPD loss (required). |
| `opd_adv_mode` | Magnitude transform: `raw` (default), `sign`, `dctv`, plus ablation modes (`power`, `group_const`, `alloc_power`, `strength_interp`). |
| `opd_tv_sched_enabled` | TV schedule for `c_k`. Mutually exclusive with `opd_adv_mode=dctv`. |
| `opd_tv_sched_beta / alpha / c_min` | EMA smoothing, annealing exponent, floor. |
| `opd_tv_sched_target` | `adv` scales the loss coefficient (paper default); `lr` scales the optimizer LR instead. |
| `opd_dctv_beta_d / beta_m / c_cap` | Estimator smoothing and cap for the gradient-norm variant. |

## License

Apache 2.0 (see `LICENSE`), inherited from verl.
