# OPD — On-Policy Distillation with TV-Calibrated Advantages

**OPD** trains a small student model against a large teacher model with a PPO-style
online loss, where the student generates rollouts on-policy and a **remote teacher
vLLM server** returns per-token log-probs used as the supervision signal. The
codebase is a fork of [verl](https://github.com/volcengine/verl) and lives in the
same repo layout; the OPD-specific pieces are additive.

The distillation loss is a PPO clip loss over
```
Δ_i = teacher_logp_i − student_logp_i,     A_i = sign(Δ_i) · |Δ_i|.
```
On top of that, `opd_adv_mode` selects **how the magnitude `|A|` is shaped** (raw,
sign, power, group_const, alloc_power, strength_interp, or DCTV), and
`opd_tv_sched_*` optionally applies a **TV-calibrated LR schedule** that anneals
the optimizer step by the estimated sequence-level TV distance (`D̂_t`) between
teacher and student.

The current tree contains the raw-advantage baseline (`opd_adv_mode=raw`,
`opd_tv_sched_enabled=False`) plus the DCTV + TV-scheduler code paths as
opt-in switches. Nothing is turned on by default beyond the raw baseline.

---

## Repository layout

```
recipe/gkd/teacher/       Remote teacher server (proxy.py + worker.py, vLLM backend, ZMQ REQ/REP)
verl/workers/actor/       DataParallelPPOActor with the OPD loss + DCTV / TV-scheduler hooks
verl/workers/config/      PolicyLossConfig (all opd_* knobs)
verl/trainer/config/      Hydra config; actor/actor.yaml documents every OPD knob
verl/utils/reward_score/  Reward scorers, incl. GPQA / LiveCodeBench extractors used by val
tests/                    Standalone unit tests (credit conservation, etc.)
train_script/             Training + fleet launch scripts (see below)
```

Everything else is stock verl.

---

## Prerequisites

- Python 3.12 venv, verl installed editable (`pip install -e .`), vLLM, Ray.
- Enough GPUs on a single node for one student run (tested on 8×B200 180GB and
  8×H100 80GB).
- One extra node (or pod) reachable by TCP to host the teacher vLLM server.
- Access to:
  - Student init ckpt (e.g. Qwen3-8B),
  - Teacher ckpt (e.g. Qwen3-8B) staged on the teacher node,
  - `deepmath_messages.parquet` and/or `dapo_17k_messages.parquet` under `$DATA_ROOT`,
  - `val_all/aime24.parquet` and `val_all/aime25.parquet` under `$DATA_ROOT`.

---

## Teacher server

The student never loads the teacher weights — it makes a ZMQ REQ/REP call
into the remote server for every batch. Start the teacher on a dedicated 8-GPU
node with vLLM TP=8 hosting the teacher model, e.g. Qwen3-8B. The server is
`recipe/gkd/teacher/{proxy.py, worker.py}`; example start-scripts and a
runbook live under `recipe/gkd/teacher/` (checked in locally per fleet, not in
this commit — copy from a working node).

The student expects:

- `TEACHER_SERVER_IP:TEACHER_SERVER_PORT` reachable (default port 15555)
- `TEACHER_N_WORKERS` slots ready (default 8)

The student's `teacher_client.py` handles retries; a single ZMQ round-trip
per batch is the only teacher dependency at train time.

---

## Training scripts

Two scripts drive the current baseline:

| Script | Role |
|---|---|
| `train_script/qwen_qwen_rawadv_b200_exp.sh`     | Single-node student run (8 GPUs). Reads env vars, launches `verl.trainer.main_ppo`. |
| `train_script/launch_qwen_qwen_rawadv_b200.sh`  | Fleet driver — fans out N student runs over SSH, all sharing one remote teacher. |

Both scripts are **path-free** — every cluster-specific value comes in through an
env var. The old fixed-path variants are deliberately kept out of git (see
`.gitignore`) because they baked in cephfs paths and cluster IPs.

### Single-node run

```bash
# One-time
export WANDB_API_KEY=<your-wandb-key>

# Per-run
export REPO_ROOT=/path/to/OPD
export VENV_DIR=$REPO_ROOT/.venv        # optional; defaults to $REPO_ROOT/.venv
export DATA_ROOT=/path/to/opd_data      # contains *_messages.parquet + val_all/
export MODEL_PATH=/path/to/student/ckpt
export TEACHER_SERVER_IP=10.0.0.99      # remote teacher

# Optional
export DATASET=deepmath          # or 'dapo'
export SEED=42
export HTTP_PROXY_URL=http://proxy:3128    # only if you need one for wandb

bash train_script/qwen_qwen_rawadv_b200_exp.sh
```

### Fleet run

Copy `train_script/fleet.example.txt` to `train_script/fleet.txt` and put one
line per node (`<host_or_ip> <label> <dataset> <seed>`). `fleet.txt` is
gitignored (it contains cluster hostnames).

```bash
export WANDB_API_KEY=...
export TEACHER_SERVER_IP=...
export DATA_ROOT=...
export MODEL_PATH=...

bash train_script/launch_qwen_qwen_rawadv_b200.sh          # launch all
bash train_script/launch_qwen_qwen_rawadv_b200.sh check    # snapshot
bash train_script/launch_qwen_qwen_rawadv_b200.sh kill     # stop all
```

`STAGGER=<seconds>` controls how long to wait between successive student
launches (default 120s) so `val_before_train` isn't hammered simultaneously.

### Perf-tune knobs (tune sweep on B200)

Passed either as env or via the launcher defaults. `tune5` is the current
production preset:

| Preset | SP | FSDP | GEN_TP | OFFLOAD | GPU_MEM_UTIL | val_before | step (s) | peak GPU |
|---|---|---|---|---|---|---|---|---|
| legacy H100 | 2 | 8 | 2 | True  | 0.90 | True | ~185 | ~166 GiB |
| tune3       | 1 | 2 | 1 | False | 0.50 | True | 171.5 | ~145 GiB |
| tune4       | 1 | 2 | 4 | False | 0.50 | True | **157.3** | ~150 GiB |
| **tune5**   | 1 | 2 | 4 | False | 0.65 | True | **157.5** | **~168 GiB** |

OOM fallback: drop `GPU_MEM_UTIL` from 0.65 → 0.50 (reverts to tune4 KV pool),
then `GEN_TP` from 4 → 1, then `ppo_max_token_len_per_gpu` from 34816 → 24576.

Verl's reported `perf/max_memory_allocated_gb` is a **cross-rank** aggregate
and overstates single-GPU pressure; the ground truth for OOM is
`nvidia-smi memory.used`.

---

## OPD-specific config knobs

All knobs live under `actor_rollout_ref.actor.policy_loss` in the trainer
config. Every knob is documented inline in
`verl/trainer/config/actor/actor.yaml`; the summary:

| Knob | Effect |
|---|---|
| `loss_mode=opd` | Enables the OPD loss (required for everything below). |
| `opd_adv_mode` | Magnitude transform on `A_i`. `raw` (default), `sign`, `power`, `group_const`, `alloc_power`, `strength_interp`, `dctv`. |
| `opd_adv_power_alpha` | Exponent α when `opd_adv_mode=power`. α=0 ⇔ sign, α=1 ⇔ raw. |
| `opd_adv_alloc_alpha` | Within-sign-group allocation exponent for `alloc_power`. |
| `opd_adv_strength_beta` | Strength interpolation for `strength_interp`. |
| `opd_dctv_beta_d / beta_m` | EMA smoothing for DCTV's `D̂` and `M̂` estimators. |
| `opd_dctv_c_cap` | Upper bound on the DCTV controller `c_t` (1.0 is the natural cap). |
| `opd_tv_sched_enabled` | TV-guided LR scheduler; **mutually exclusive** with `opd_adv_mode=dctv`. |
| `opd_tv_sched_beta / alpha / c_min` | EMA smoothing, annealing exponent, and floor for the scheduler. |
| `opd_tv_sched_target` | `lr` (multiplies optimizer LR) or `adv` (multiplies loss coefficient). |

The `raw` baseline sets `opd_adv_mode=raw` and `opd_tv_sched_enabled=False`.
The current 4-run sweep uses exactly that; DCTV and the TV-scheduler are the
next-round sweeps.

---

## Verl / upstream

The upstream verl README is at [`docs/`](docs/) and on
[github.com/volcengine/verl](https://github.com/volcengine/verl). This fork
adds the OPD loss, the DCTV controller, the TV-guided LR scheduler, and the
teacher-client integration. Everything else — Ray HybridFlow controller, FSDP
workers, vLLM rollout, Hydra config plumbing — is upstream verl.

## License

Apache 2.0 (see `LICENSE`), inherited from verl.
