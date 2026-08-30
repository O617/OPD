# CLUSTER_QUICKSTART.md — 集群重启后快速起训

> 每次 pod 重启（`/root/` overlay-fs 被清空、hostname 变更、Ray 状态丢失）后照这个文档跑一遍即可从零到训练起飞。

## 0. 先确认三件事

1. **集群 prefix**：登陆节点 hostname 前缀，例如 `$CLUSTER_PREFIX`（前缀变了就更新脚本，见 §4）
2. **可用节点数**：README 里写 16 台不代表都能 resolve，用下面这条命令探
3. **teacher server**：`<REDACTED_IP>:15555` 是**另一套 pod** 上的 vLLM 服务，本集群重启不影响它，但要 tcp probe 一下确认还活着

```bash
CLUSTER=$CLUSTER_PREFIX    # ← 改成当前集群 prefix
# 探节点存活 + 编号
for n in launcher worker-{0..14}; do
  timeout 4 ssh -o StrictHostKeyChecking=no -o BatchMode=yes "${CLUSTER}-${n}" \
    "echo ${n} OK" 2>/dev/null || echo "${n} MISSING"
done
# 探 teacher
ssh "${CLUSTER}-launcher" \
  "timeout 5 bash -c 'echo > /dev/tcp/<REDACTED_IP>/15555' && echo TEACHER_ALIVE || echo TEACHER_DEAD"
```

如果拿到的能 SSH 的节点数 < 你想跑的 run 数，先跟运维要机器再走后面。

## 1. 起占卡脚本

占卡脚本在 `$DATA_ROOT/CLUSTER_README.md` 里，重启后一般已经在跑，通常不需要人工再起。**判据**：

```bash
for n in launcher worker-{0..6}; do
  ssh "${CLUSTER}-${n}" "pgrep -f 'occupy.py' | wc -l | xargs -I{} echo ${n}: occupy={}" &
done; wait
```

- 数字非 0 → 已在跑，跳过
- 数字为 0 → 手动起：`ssh "${CLUSTER}-${n}" "cd ~ && nohup python3 ./occupy.py --all > tmp/occupy_\$(hostname).log 2>&1 &"`

**关键性质**（不用手动释放）：占卡脚本一旦检测到真实任务把 GPU 利用率打到 >60% 就自动让出对应卡，任务结束后自动占回。所以起训练前**不需要** kill occupy.py。

## 2. Prewarm 学生模型到每台的 `/root/`

`/root/` 是 overlay fs，pod 重启会丢。学生 3.4 G 从共享盘读会有 8× per-node 冷启动开销，一定要 prewarm。

```bash
cd $DATA_ROOT/develop/OPD
bash train_script/prewarm_ds_distill_qwen15b.sh          # 缺才拷（幂等）
bash train_script/prewarm_ds_distill_qwen15b.sh --check  # 只看状态不动
bash train_script/prewarm_ds_distill_qwen15b.sh --force  # 强制重拷
```

期望输出：每台一条 `COPIED 3.4G`（首次）或 `OK 3.4G`（已在）。

**注意**：该脚本内部 hardcode 了 `CLUSTER_PREFIX`。集群 prefix 变更时同步改，见 §4。

## 3. Wandb key（每次新 pod 都要写一次 `<netrc-path>`）

```bash
cat > <netrc-path> <<'EOF'
machine api.wandb.ai
  login user
  password <你的 wandb_v1_... key>
EOF
chmod 600 <netrc-path>
```

各 dispatcher 会用 `awk /machine api.wandb.ai/{...}` 从这里读，然后 `env WANDB_API_KEY=...` 传到 worker 端。

**验证**：`awk '/api.wandb.ai/{f=1;next} f&&/password/{print $2;exit}' <netrc-path>` 应打印出 key。

## 4. 换集群 prefix 时要改的地方

集群重启偶尔会换 hostname 前缀（如 `ts-69aDaB...` → `ts-3bfFC7...`）。用 `grep -rn` 找一下：

```bash
grep -rln "^CLUSTER_PREFIX=" $DATA_ROOT/develop/OPD/train_script/
```

至少要改：
- `train_script/prewarm_ds_distill_qwen15b.sh`
- `train_script/launch_ds_ds_dapo_dctv.sh`
- `train_script/launch_ds_ds_dapo_replicates.sh`（如果还用）
- `train_script/launch_ds_ds_dapo_phase2.sh`（如果还用）

顺手也把 `ALL_NODES` 数组截到当前实际可用节点数（README 写 16 台不代表都能 resolve）。

## 5. 起训练任务

### DCTV 主实验（当前）

```bash
cd $DATA_ROOT/develop/OPD
bash train_script/launch_ds_ds_dapo_dctv.sh         # dry-run，看 4 台节点/seed 映射对不对
bash train_script/launch_ds_ds_dapo_dctv.sh --go    # 真起
```

默认拓扑：
- 4 seeds (42/63/7/21) × 单机 8 GPU × 4 nodes = 32 GPU 一次
- Trainer 脚本：`train_script/ds_ds_dapo_dctv_exp.sh`
- 每节点 log：`/tmp/ds_ds_dapo_dctv_seed{SEED}.log`
- Wandb: project `ON_POLICY_DISTILL`, run `OPD_DS_DS_DAPO_adv_dctv_seed{SEED}`
- 用剩下 4 台的话，就手动 dispatch 别的方法 (raw/sign/…) 补 seeds，dispatcher 都在 `train_script/`

### 起飞后监控

```bash
CLUSTER=$CLUSTER_PREFIX
declare -A TAGS=(
  [worker-0]=dctv_seed42
  [worker-1]=dctv_seed63
  [worker-2]=dctv_seed7
  [worker-3]=dctv_seed21
)
for n in "${!TAGS[@]}"; do
  tag=${TAGS[$n]}
  ssh -o ConnectTimeout=4 "${CLUSTER}-${n}" "
    proc=\$(pgrep -f 'verl.trainer.main_ppo' | wc -l)
    err=\$(grep -cE 'CUDA out of memory|zmq.*Timeout|TimeoutError.*teacher|Traceback' /tmp/ds_ds_dapo_${tag}.log 2>/dev/null || echo 0)
    step=\$(grep -oE 'Training Progress: *[0-9]+%[^\"]*[0-9]+/2000' /tmp/ds_ds_dapo_${tag}.log 2>/dev/null | tail -1)
    gpu=\$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader | awk '{s+=\$1}END{printf \"%.0f\", s/NR}')
    printf '${n}/${tag}: procs=%s err=%s gpu=%s%% step=\"%s\"\n' \$proc \$err \$gpu \"\$step\"
  " &
done; wait
```

**判活优先级**：`nvidia-smi util` > log mtime。val_before_train 阶段 log 会 buffered 静默 3-5 min 但 GPU 满载 86-89%，属正常。

**判死的信号**：`err > 0`（有 traceback / zmq timeout / OOM）+ log mtime 卡住 > 10 min + GPU util 掉到 0。

### Kill 单个 run

```bash
ssh "${CLUSTER}-worker-N" 'pkill -9 -f verl.trainer.main_ppo; ray stop --force 2>&1 | tail -3'
```

### Kill 全部当前 DCTV runs

```bash
CLUSTER=$CLUSTER_PREFIX
for n in worker-0 worker-1 worker-2 worker-3; do
  ssh "${CLUSTER}-${n}" 'pkill -9 -f "python3 -m verl.trainer.main_ppo"' >/dev/null 2>&1 &
done; wait
```

**⚠️ 绝对不要** `pkill -f 'raylet|gcs_server|ray::'` 一把梭 —— `ray::` 会匹配到远端 teacher server 的 vLLM ray workers，误杀。**只在训练节点分开清 ray**，或用 `ray stop --force`（需 activate venv）。

## 6. 常见坑

1. **首步 `Ray connect` 卡死**：极偶尔（08-25 handoff 出过一次 worker-2）。修：`pkill main_ppo` + `ray stop --force` + `pkill raylet/gcs_server/ray::` **只在该节点**，然后 dispatcher 重派该 tag。
2. **`_inductor fx graph cache` W-level traceback**：起步阶段全部 run 都有，是 warning-level 非致命；判致命 traceback 用 `grep -E 'CUDA out of memory|RuntimeError|zmq.*Timeout|TimeoutError.*teacher'`。
3. **log 短暂不 flush**：Ray 有 buffering，log mtime 卡 3-5 min 但 GPU 满载属正常。判活以 `nvidia-smi util` 为准。
4. **step:0 ≠ 第一步训练完**：`step:0` 是 val_before_train 打的，真正第一步的判据是 `Training Progress: 1/2000` 或 `step:1`。DCTV 首步预计 c_0 = 1（等价 sign），wandb 上 `actor/opd_dctv_c_next` 出现 <1 才是 controller 开始工作。
5. **wandb 需要 proxy**：脚本里已 `export HTTPS_PROXY=$HTTP_PROXY_URL`；`NO_PROXY` 保 teacher(<REDACTED_IP>) / ray 直连。忘了改 prefix → 会看到 `wandb.errors.CommError` 报错，log 里 grep 一下就知道。
6. **teacher server 一旦挂**：reward manager `zmq.RCVTIMEO = 30 min` 才 timeout，log 里 30 分钟没进度且 GPU 掉到低利用率是特征。手动 `ssh "${CLUSTER}-launcher" "timeout 5 bash -c 'echo > /dev/tcp/<REDACTED_IP>/15555'"` 再确认。

## 7. 一图流：pod 重启后到训练起飞

```
① CLUSTER=<current prefix>                           # 更新变量
② 探节点 + teacher 存活                                # 见 §0
③ occupy.py 检查（几乎总在）                            # §1
④ prewarm 学生模型                                     # §2
⑤ <netrc-path> 写 wandb key                            # §3
⑥ 需要时改各 dispatcher 里的 CLUSTER_PREFIX + ALL_NODES  # §4
⑦ dispatcher --go                                     # §5
⑧ 90s 后 nvidia-smi util + log err 双通道监控           # §5
```

预期时间：③④⑤⑥ 全串行 ~5 min，⑦起完 30 s，⑧ val_before_train 到第一个 train step ~15 min，第一次 val (step 25) ~2 h，完整 2000 step ~7.5 天。

## 8. 相关文档

- `CLUSTER_QUICKSTART.md` — 本文档
- `DCTV_OPD_method.md` — DCTV 方法论
- `HANDOFF-2026-08-27.md` — 08-27 最新 handoff（原 08-27 集群）
- `HANDOFF-2026-08-25.md` — Phase-2 launch handoff（含 occupy/ray 坑详解）
- `$DATA_ROOT/CLUSTER_README.md` — 集群硬件/占卡脚本参考
