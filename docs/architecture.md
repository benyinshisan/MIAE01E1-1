# DQN 复现工程：架构与接口契约

- 状态：草案（模块划分、接口契约、环境版本已定；配置命名、评测口径、包结构待定）
- 接口代码：[`prototype/interfaces.py`](../prototype/interfaces.py)（只有签名，无说明）
- 论文规格来源：Mnih et al., *Human-level control through deep reinforcement learning*, Nature 518:529–533, 2015
- 决策记录：ADR-0007（主线改为 DQN 并完全自写）

## 1. 总览

八个模块，单向依赖：

```text
runner ──→ trainer ──→ agent ──→ network
  │            │          └───→ buffer
  │            └──→ envs ──→ config
  └──→ evaluator ──→ agent, envs
```

| 模块 | 职责 | 明确不做 |
|---|---|---|
| `config` | 配置数据类，五个子规格 | 不读环境变量、不解析 CLI |
| `envs` | `EnvFactory`：造训练/评测环境，含 wrapper 链 | 不长期持有 env 实例 |
| `buffer` | `ReplayBuffer`：numpy 环形数组 | **不 import torch** |
| `network` | `QNetwork`：论文 CNN | 不知道 env、不知道动作语义 |
| `agent` | 选择动作、梯度更新、同步 target | 不写日志、不建环境、不控制循环 |
| `trainer` | 步驱动主循环 | 不定义网络、不碰 buffer 内部数组 |
| `evaluator` | 独立环境上的评测 | 不复用训练环境实例 |
| `runner` | 组装、跑训练、跑评测、落盘 | 不含算法逻辑 |

## 2. 依赖规则

1. 只允许向下依赖，禁止反向 import 与循环依赖。
2. `config` 被所有人 import，但它不 import 任何内部模块。
3. `buffer` 保持纯 numpy；设备搬运与归一化在 agent / network 侧。
4. `trainer` 只能通过 buffer 的公开方法取数据。
5. `evaluator` 必须自己造环境，否则训练期 ε 与评测 ε 会互相污染。
6. 日志只写在 `trainer / evaluator / runner`；agent 与 buffer 不知道 writer 存在。

## 3. 三套计数单位

论文里 frame / step / update 是三套不同的计数，混用会静默出错：

| 量 | 含义 | 换算 |
|---|---|---|
| frame | Atari 原始帧 | `frame = agent_step × action_repeat`（默认 ×4） |
| agent_step | 智能体做出一次决策 | 主循环的计数单位 |
| update | 一次梯度更新 | `update = agent_step ÷ learn_every_steps`（默认 ÷4） |

由此得到两个最容易搞错的参数：

- `eps_decay_frames = 1,000,000` 是**帧**，转成步是 250,000 步。
- `target_update_every_updates = 10,000` 是**参数更新次数**，转成帧是
  `10,000 × 4 × 4 = 160,000` 帧（引自论文 Extended Data Table 1 表注）。

## 4. 各模块契约

### config

`EnvSpec / OptimSpec / AlgoSpec / EvalSpec / RunSpec` 由 `RunConfig` 聚合。

- `with_overrides(**flat)` 接受 `algo.gamma=…` 或 `gamma=…` 两种写法；
  **未知字段必须抛错**，不得静默忽略——否则 3 种子 × 3 档 buffer 的批量实验
  会跑错配置而不自知。
- `AlgoSpec.action_repeat` 与 `EnvSpec.action_repeat` 必须一致，由 `build_config` 校验。

### envs

`EnvFactory.make_train(seed)` / `make_eval(seed)` 各自返回独立 env。

- 训练环境包装链：NoopReset → MaxAndSkip(4) → EpisodicLife → FireReset（仅含 FIRE 的游戏）
  → ClipReward → ResizeObservation(84) → GrayscaleObservation → FrameStackObservation(4)。
- 评测环境与训练相同，**但不启用 EpisodicLife 的生命截断**——评测回合按真 game over
  结束，否则与论文"每局最长 5 分钟、30 局取平均"的口径不符。
- 构造 ALE 时必须显式传 `frameskip=1`（`ALE/Pong-v5` 默认 4）与
  `repeat_action_probability=0.0`。

### buffer

- 容量固定，写满覆盖最旧；dtype 固定 `uint8`。
- **只分配一份观测数组**：`next_obs` 不另存，取 `obs_buf[(i + 1) % capacity]`。
  因此写指针那一格不可被采样（它的下一格还没写入）。
- `sample()` 在 `len(self) < batch_size` 时抛 `ValueError`，不得静默缩小 batch。
- `push(done=...)` 只接收**真终止**；时间上限截断不传进来。

### network

论文结构：`conv 32@8×8/s4 → ReLU → conv 64@4×4/s2 → ReLU → conv 64@3×3/s1 → ReLU
→ flatten → FC 512 → ReLU → FC n_actions`（线性输出）。

- 输入 `(B, 4, 84, 84)`，dtype 通常为 `uint8`；**`x / 255.0` 必须发生在 `forward` 里**。
- 这不是风格问题：buffer 存 uint8 是内存设计的一部分。若改成 buffer 存 float32，
  200k 容量会从 5.6 GB 变成 22 GB，本机与 Colab 都装不下。

### agent

- `act(obs, epsilon)` 训练期用，带 ε 探索；`act_greedy(obs)` 评测期用，纯贪心。
- `epsilon_at(agent_step, spec)` 线性退火且**按帧换算**：
  `frames = agent_step × action_repeat`，
  `eps = max(eps_start − (eps_start − eps_end) × frames / eps_decay_frames, eps_end)`。
- `update()` 返回 TD loss；样本不足时返回 `None` 且不更新。
- `sync_target()` 由调用方按更新次数触发，agent 自己不计时。
- device 由 agent 持有；buffer 给的是 numpy，agent 负责搬到 device。

### trainer

步驱动主循环：

```text
for step in range(total_agent_steps):
    eps    = agent.epsilon_at(step, algo)
    action = agent.act(obs, eps)
    next_obs, r, term, trunc, info = env.step(action)
    done   = term                        # 不含 trunc
    real_next = info["final_observation"] if trunc else next_obs
    buffer.push(obs, action, r, real_next, done)
    obs = next_obs
    if step % learn_every_steps == 0 and step >= replay_start_steps:
        if agent.update() is not None:
            updates += 1
            if updates % target_update_every_updates == 0:
                agent.sync_target()
    if episode_end:
        log 未裁剪的原始回合得分
```

不变量：

- 环境步数与 buffer 写入数一一对应，预热期不得丢样本。
- `real_next_obs` 在 trunc 时必须用 `final_observation` 覆盖，否则引导项跨回合泄漏。
- 日志里存**原始得分**（未裁剪、未归一化），归一化在报告层做。

### evaluator

- 跑 `EvalSpec.episodes` 局，每局不同随机 no-op 起始，ε 固定 0.05。
- 回合结束按真 game over；返回**原始**得分向量 + 均值 + 标准差。
- 人类归一化 `100 × (ours − random) / (human − random)`；Pong 常量
  random `−20.7`、human `9.3`（论文 Extended Data Table 2）由 runner 注入，
  不硬编码在 evaluator 内。

### runner

- 唯一解析 CLI/YAML、唯一创建 run 目录的地方。
- 组装顺序：config → env factory → buffer → network → agent → trainer → evaluator。
- `run_name` 必须能从名字反推 (env_id, seed, 关键超参, 时间戳)。

## 5. 冻结顺序与开工起点

1. `config` 先冻结（被每个构造函数引用，改一个字段牵动全仓）——**唯一串行点**。
2. 之后 `buffer` 与 `envs` 的接口并行冻结。
3. `agent` 依赖前两者；`network` 只依赖 torch 与 config，任何时候都能开工。

三人起点：A = config → buffer；B = envs；C = network。

## 6. 环境与版本

- 统一 `ALE/Pong-v5`，必须 `frameskip=1`（默认 4，与 MaxAndSkip(4) 叠加会变成
  每动作 16 帧，且不报错）。
- `gymnasium 1.0` 的 wrapper 名：`GrayscaleObservation`、`FrameStackObservation`
  （0.29 的 `GrayScaleObservation` / `FrameStack` 会 `AttributeError`）。
- `ale-py ≥ 0.10` 随包分发 Atari ROM，不需要 AutoROM，也不需要额外的 ROM 下载脚本。
- 依赖见 [`../environment.yml`](../environment.yml)（conda）与
  [`../requirements.txt`](../requirements.txt)（Colab / 纯 pip）。
- Colab 免费层：内存 12.7 GB，200k 容量 buffer 占 5.6 GB；device 不可写死；
  GPU 上数字与 CPU 不完全可比，需在报告中说明。

## 7. 默认值即论文规格

| 字段 | 值 | 出处 |
|---|---|---|
| `OptimSpec.kind` | `rmsprop` | Methods |
| `OptimSpec.lr / momentum / alpha / eps` | 2.5e-4 / 0.95 / 0.95 / 0.01 | EDT1 |
| `OptimSpec.td_error_clip` | 1.0（裁 TD 误差到 [−1,1]） | Methods |
| `OptimSpec.grad_clip_norm` | `None`（论文无 clipnorm） | Methods |
| `AlgoSpec.gamma / batch_size` | 0.99 / 32 | EDT1 |
| `AlgoSpec.replay_start_frames` | 50,000（帧） | EDT1 |
| `AlgoSpec.learn_every_steps` | 4 | EDT1 update frequency |
| `AlgoSpec.target_update_every_updates` | 10,000（参数更新次数） | EDT1 |
| `AlgoSpec.eps_start / eps_end / eps_decay_frames` | 1.0 / 0.1 / 1,000,000 | EDT1 |
| `AlgoSpec.replay_capacity` | **200,000**（论文 1e6） | 内存上限，属 deviation |
| `AlgoSpec.total_agent_steps` | **2,500,000**（= 10M 帧；论文 50M 帧） | 实验预算，属 deviation |
| `EvalSpec.episodes / epsilon` | 30 / 0.05 | Methods |

## 8. 未纳入契约的部分

- **训练续跑（resume）**：单次约 2.3 小时，不做续跑；若验收流程需要再补。
- `EvalSpec.every_frames = 1M` 是**我们的假设**——论文正文未写明主结果的评测间隔，
  需在报告 Limitations 中声明。
- run 命名、日志 tag、seed 贯穿点；包形态、入口命令、测试位置。
