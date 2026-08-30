对，我同意你的判断。**方法应该定义在 gradient space，而不是 parameter-update space。** TVOPD 本身给出的是一个 policy-gradient estimator，我们研究的是如何对这个 gradient field 做 principled scaling；Adam/AdamW 是所有方法共同套在外面的优化器。除非我们论文研究的是 optimizer dynamics，否则没必要为了让 \(c_t\) 精确对应最终 \(\Delta\theta\) 再去 modifier optimizer update。

而且顺着这个思路，我觉得我们可以把刚才那个“估计一步 TV motion”的方案再改得更漂亮：**甚至不需要通过相邻两步 \(D_t-D_{t-1}\) 来估 motion，直接从 TV gradient norm 推出来。** 这样完全是 gradient-space 的。

---

# 方法：Adaptive Distance-Calibrated TVOPD

定义

$$
\Delta_{i,t}
=
\log\pi_T(a_i|s_i)
-
\log\pi_{\theta_t}(a_i|s_i),
$$

以及 pure-TV token direction

$$
z_{i,t}
=
\operatorname{sign}(\Delta_{i,t}).
$$

原始 TVOPD gradient 为

$$
g_t^{\rm TV}
=
-\mathbb E_i
\left[
z_{i,t}\nabla_\theta\log\pi_\theta(a_i|s_i)
\right].
$$

忽略符号约定，本质上有

$$
\boxed{
g_t^{\rm TV}
=
2\nabla_\theta D_t
}
$$

其中

$$
D_t
=
\bar D_{\rm TV}^{\rm on}(\pi_T,\pi_{\theta_t}).
$$

---

## 1. Sequence-level TV distance estimator

每个 sampled token：

$$
d_{i,t}
=
\left[
1-\exp(\Delta_{i,t})
\right]_+,
$$

满足

$$
0\le d_{i,t}\le1,
$$

且在 student sampling 下：

$$
\mathbb E[d_{i,t}\mid s_i]
=
D_{\rm TV}
(\pi_T(\cdot|s_i),\pi_S(\cdot|s_i)).
$$

对 sequence：

$$
\hat D_b
=
\frac1{L_b}\sum_l d_{b,l},
$$

再对 batch：

$$
\hat D_t
=
\frac1B\sum_b \hat D_b.
$$

维护：

$$
\bar D_t
=
\beta_D\bar D_{t-1}
+
(1-\beta_D)\hat D_t.
$$

建议第一轮固定

$$
\beta_D=0.95.
$$

---

# 2. 从 gradient norm 推导“一次 full TV step 能走多远”

这是现在最漂亮的一步。

如果暂时考虑普通 gradient descent：

$$
\theta_{t+1}
=
\theta_t
-
\eta_t c_t g_t^{\rm TV},
$$

由于

$$
g_t^{\rm TV}=2\nabla D_t,
$$

一阶 Taylor expansion：

$$
\begin{aligned}
D_{t+1}
&\approx
D_t
+
\nabla D_t^\top
(\theta_{t+1}-\theta_t)
\\
&=
D_t
-
2\eta_tc_t
\|\nabla D_t\|^2.
\end{aligned}
$$

因为

$$
\|g_t^{\rm TV}\|^2
=
4\|\nabla D_t\|^2,
$$

所以

$$
\boxed{
D_{t+1}
\approx
D_t
-
c_t
\underbrace{
\frac{\eta_t}{2}
\|g_t^{\rm TV}\|^2
}_{M_t}
}
$$

定义

$$
\boxed{
M_t
=
\frac{\eta_t}{2}
\|g_t^{\rm TV}\|^2
}
$$

作为 **full-TV step 的一阶 TV motion scale**。

它有非常明确的意义：

> 如果这一 step 不做任何 damping，即 \(c_t=1\)，按照局部一阶模型预计会减少多少 TV distance？

---

# 3. 自动得到 \(c_t\)，不再需要 \(\tau\)

为了避免预测的一步运动超过剩余 TV distance：

$$
c_tM_t\le D_t.
$$

所以最自然的 maximal full-direction step 是

$$
\boxed{
c_t
=
\min
\left(
1,
\frac{D_t}{M_t}
\right).
}
$$

于是：

### 距离远

$$
D_t\gg M_t
\Rightarrow
c_t=1.
$$

完全使用 pure TV。

### 进入 optimizer resolution region

$$
D_t<M_t
$$

则

$$
c_t=\frac{D_t}{M_t}<1.
$$

### 接近 teacher

$$
D_t\rightarrow0
\Rightarrow
c_t\rightarrow0.
$$

所以根本不需要定义：

$$
D<\tau
$$

什么时候算“足够近”。

临界条件由训练动力学自己产生：

$$
\boxed{
\text{remaining TV distance}
\approx
\text{one-full-gradient-step TV motion}.
}
$$

我觉得这比固定 \(\tau\) 好很多。

---

# 4. 它还有一个很漂亮的 first-order interpretation

把

$$
c_t=\min(1,D_t/M_t)
$$

代回去：

如果

$$
D_t<M_t,
$$

则

$$
D_{t+1}
\approx
D_t-\frac{D_t}{M_t}M_t
=0.
$$

也就是说 controller 做的是：

$$
\boxed{
\text{take the largest TV-gradient step that does not
first-order overshoot the teacher}.
}
$$

这有一点类似 Polyak step / trust-region 的思想，但又是直接从 TV geometry 推出来的。

论文里我会叫：

**distance-to-motion calibration**。

---

# 5. 实际实现不要用当前 batch 直接控制当前 batch

为了理论干净，\(c_t\) 最好由历史决定。

所以实际上：

step \(t\) 使用

$$
\boxed{
c_t
=
\min
\left(
1,
\frac{\bar D_{t-1}}
{\bar M_{t-1}+\epsilon}
\right)
}
$$

其中：

$$
\bar M_t
=
\beta_M\bar M_{t-1}
+
(1-\beta_M)\hat M_t.
$$

第一轮直接：

$$
\beta_M=\beta_D=0.95.
$$

这样 conditioned on previous training history：

$$
c_t
$$

是一个 deterministic positive scalar。

所以当前 stochastic TV gradient：

$$
\hat g_t^{\rm adaptive}
=
c_t\hat g_t^{\rm TV}
$$

仍满足

$$
\mathbb E[
\hat g_t^{\rm adaptive}
\mid\mathcal F_{t-1}
]
=
2c_t\nabla D_t.
$$

因此：

$$
\boxed{
\text{TV direction is exactly preserved in expectation.}
}
$$

---

# 6. 实现时怎么拿到 \(M_t\)

最终 loss 直接还是：

$$
\boxed{
A_{i,t}
=
c_t\,\operatorname{sign}(\Delta_{i,t}).
}
$$

假设当前 backward 后看到的 OPD gradient norm 是

$$
G_t^{\rm scaled}.
$$

因为 global scalar 对所有 loss gradient 等比例作用：

$$
G_t^{\rm scaled}
=
c_t G_t^{\rm TV}.
$$

因此可以恢复：

$$
G_t^{\rm TV}
=
\frac{G_t^{\rm scaled}}
{\max(c_t,\epsilon)}.
$$

然后：

$$
\boxed{
\hat M_t
=
\frac{\eta_t}{2}
\left(
G_t^{\rm TV}
\right)^2.
}
$$

最好使用：

> **gradient clipping 之前的 OPD grad norm**。

如果你本来就有 `grad_norm` logging，这个实现改动应该很小。

---

## 一个实现伪流程

```text
Initialize:
    c = 1
    D_ema = None
    M_ema = None

For iteration t:

    # rollout
    delta = teacher_logp - student_logp

    # TV direction
    adv = c * sign(delta)

    # TV distance estimator
    tv_token = relu(1 - exp(delta))
    tv_seq   = mean_per_sequence(tv_token)
    D_hat    = mean(tv_seq)

    # normal TVOPD backward
    loss = -mean(adv * student_logp)
    backward(loss)

    # pre-clip grad norm
    G_scaled = grad_norm()

    # recover full-TV gradient scale
    G_tv = G_scaled / max(c, eps)
    M_hat = 0.5 * lr * G_tv^2

    optimizer.step()

    # update statistics
    D_ema = EMA(D_hat)
    M_ema = EMA(M_hat)

    # controller for NEXT iteration
    c = min(1, D_ema / (M_ema + eps))
```

第一步没有统计量时：

$$
c_0=1.
$$

完全不需要额外 warmup hyperparameter。

---

# 7. 理论性质现在可以总结得非常干净

### Token-wise magnitude erasure

任何两个 token：

$$
|A_i|=|A_j|=c_t.
$$

所以：

$$
\boxed{
\text{No token-wise magnitude shaping.}
}
$$

---

### TV descent direction preserved

$$
g_t^{\rm adaptive}
=
c_tg_t^{\rm TV},
\qquad c_t>0.
$$

因此 gradient direction 完全一致。

---

### Teacher-guided policy improvement preserved

若原来

$$
\mathbb E[A^{\pi_S}z_T]>0,
$$

那么：

$$
c_t
\mathbb E[A^{\pi_S}z_T]>0.
$$

所以 first-order improvement direction 不变。

---

### TV metric interpretation preserved

我们依然是在减小：

$$
D_{\rm TV}(\pi_T,\pi_S).
$$

因此之前 TV triangle inequality / teacher-target closeness 的论证完全保留。

---

### Automatic late-stage damping

controller 只在

$$
D_t
\lesssim M_t
$$

时激活。

也就是说不是：

> “训练到后期所以我要 decay。”

而是：

> **“剩余误差已经小于当前 stochastic TV gradient 的一步运动尺度，所以必须减速。”**

这个解释强很多。

---

# 实验 Plan：4 seeds

这轮我建议不要再铺 ablation，直接验证这一条 hypothesis。

## 主比较

同一个 pair、完全相同训练配置：

| Method                |        Seeds |
| --------------------- | -----------: |
| RKL-OPD               | existing / 4 |
| TVOPD                 | existing / 4 |
| **Adaptive DC-TVOPD** |        **4** |

新方法不 sweep 参数：

$$
\beta_D=\beta_M=0.95
$$

固定即可。

这两个 \(\beta\) 我甚至会定位成 estimator smoothing constants，而不是 method hyperparameters。

---

## 必须新增 logging

每 step：

$$
\hat D_t,\quad
\bar D_t
$$

$$
G_t^{\rm TV}
$$

$$
\hat M_t,\quad
\bar M_t
$$

$$
\boxed{
R_t=\frac{\bar D_t}{\bar M_t}
}
$$

以及

$$
c_t=\min(1,R_t).
$$

其中 **\(R_t\)** 是这一轮最关键的诊断指标。

---

# 核心实验预测

我们现在的 hypothesis 给出了非常强的预测。

### Early / middle stage

应该有：

$$
R_t=
\frac{D_t}{M_t}
\gg1.
$$

于是：

$$
c_t=1.
$$

所以 Adaptive DC-TV 和 pure TV 的曲线应该几乎完全重合。

如果前期就大量出现

$$
c_t<1,
$$

反而说明我们的 motion estimator 或理论假设有问题。

---

### Pure TV 开始 late instability 时

我们预测应该观察到：

$$
\boxed{
R_t\rightarrow O(1).
}
$$

即剩余 TV distance 已经下降到“一次 full TV gradient 能移动的尺度”。

这本身就是一个非常重要的 validation。

---

### Adaptive DC-TV

当：

$$
R_t<1
$$

之后：

$$
c_t=R_t<1.
$$

于是应该看到：

* TV oscillation 下降；
* grad magnitude 自动下降；
* eval variance / late-stage fluctuation 下降；
* 前中期性能不受损；
* 最终性能追平或超过 RKL；
* 同时保留 TVOPD 的前中期优势。

---

# 最关键的一张图

我会把四条量放在训练 step 上对齐：

$$
\text{Eval Performance},
\qquad
\bar D_t,
\qquad
R_t=D_t/M_t,
\qquad
c_t.
$$

然后标出 pure TV 开始 unstable 的区域。

如果最后看到：

$$
\text{TV instability onset}
\approx
R_t\approx1
$$

并且 Adaptive DC-TV 在这里自动：

$$
c_t<1
$$

随后 stability 恢复，这个结果就非常完整了。

---

我觉得现在这版比固定 \(\tau\) 的版本明显更好。它不再是 **“Huber-TV with a threshold”**，而变成了一个真正自适应的：

$$
\boxed{
\textbf{TV direction + TV distance + TV gradient motion}
}
$$

三者全部来自同一个 optimization geometry。

如果要给它一句最简洁的方法描述，我会写：

> **We remove token-wise magnitude entirely, retain only the sampled TV descent direction, and globally scale this direction by the ratio between the remaining policy-level TV distance and the predicted one-step TV motion.**

这个已经很像一个完整 method 了。