# IL→RL（CPU 方案）：模仿学习暖启动 + 自对弈强化学习

- **日期**: 2026-07-06
- **状态**: 待执行
- **路线**: IL（行为克隆）暖启动 → RL（自对弈）微调 → （可选）IS-MCTS 包装
- **算力**: 纯 CPU（本机 6 核，无 GPU）——已验证可行
- **前置**: 本地 cabt 环境已搭好（`specs/000-ptcg-env-setup/plan.md`）；S1 Dragapult sample agent 已提交（`specs/001-dragapult-sample-agent/plan.md`）

---

## 1. 背景与思路

### 1.1 现状
- random baseline publicScore = 332.9；S1 Dragapult sample 已提交待出分。
- 现有 agent 均为**纯规则**。

### 1.2 为什么 IL→RL
这是 **AlphaGo 配方**：先用人类高手对局做监督学习（IL）拿到不随机的强力策略，再用 RL 自对弈在其基础上迭代超越。针对 PTCG：
- **解决 RL 最大痛点**：状态空间巨大+隐藏信息，从零随机起步的 RL 探索不动；IL 暖启动让自对弈从 "competent play" 开始，样本效率天差地别。
- **工作复用**：IL 阶段建的"状态编码器 + 动作头"网络，RL 阶段**直接接着用同一个网络**继续训（policy gradient 在自对弈返回上更新）。IL 是 RL 的初始化，不是前置任务。
- **每阶段可提交验证**：IL 出货（~高手水平）→ RL 微调出货（争取超高手）→ 可选 MCTS 包装（上限再提）。

### 1.3 为什么 CPU 可行（已实测）
本机 6 核，引擎（`cg.game` ctypes 直跑，随机动作）实测：
- **39.5 局/秒/核，25ms/局，~120 步/局**（含 Python+ctypes 开销）。
- 6 核多进程：引擎 **~240 局/秒**。全局 `battle_ptr` 限单进程单局，但每进程独立加载 `libcg.so`、内存隔离，多进程天然并行。
- 加每步小 MLP 前向（~0.1-0.5ms）约翻倍 → **~120 局/秒**。
- IL：~10⁶ 样本 × 小 MLP × 几 epoch = **几分钟~1 小时**。
- RL：10⁵-10⁶ 局自对弈 = **~15 分钟~2.5 小时**。

> 无 GPU 这个 blocker 基本消除。代价：网络必须是小 MLP（CPU 训练快 + 提交环境无 torch 时能 freeze 成纯 numpy 前向——两约束一致）。

### 1.4 目标与非目标
- **目标**：Phase 1 出 IL agent（ladder ≥800，接近高手）；Phase 2 RL 微调超越 IL（ladder ≥ IL 分）。
- **非目标**：策略报告撰写；大网络/transformer；云 GPU（除非后期 6 核不够）。

---

## 2. 共享架构（IL/RL 复用）

```
obs_dict ──[状态编码器]──▶ tensor ──[小 MLP trunk]──▶ embedding ──[动作头]──▶ 每个 option 的得分
                                                                          │
                                                          选 top-maxCount（mask 非法）= action
```

### 2.1 状态编码器（obs_dict → 张量）
obs 顶层字段（已验证）：`select / logs / current / search_begin_input`。

编码分量（均归一化/one-hot）：
- **我方手牌**：60 槽 one-hot(1267) + count。
- **我方场面**：主动 + 替补（最多 5），每只：卡 ID emb / 当前HP·maxHP / 能量列表 / 工具 / 伤害指示物 / 状态。
- **我方牌库/奖品**：剩余数；奖品可由 decklist − 可见牌 推断（wmh 的 PrizeTracker 思路，~83% 决策可推断）。
- **对方可见场面**：主动 + 替补（同上，但能量/手牌为可见部分）+ 已拿奖品数 + 牌库剩余数 + 手牌数。
- **当前决策上下文**：`select.context`(SelectContext 枚举 one-hot) + `minCount/maxCount` + `select.type` + `contextCard` + `effect`。
- **合法 option 列表**：每个 option 编码为小向量（`type` + 引用的卡/目标），与 trunk embedding 拼接后过动作头。
- **全局**：`current.turn/yourIndex/firstPlayer/supporterPlayed/stadiumPlayed/energyAttached/stadium`。
- **logs**：最近 K 条 Log 事件（类型 one-hot + 涉及卡）。

> 1267 卡池：用 one-hot 过一个 embedding 层（128-256 维）学卡牌表示，比裸 one-hot 紧凑。

### 2.2 动作头（可变动作空间）
- 每个合法 option → 标量得分 `s_i`。
- 选 `maxCount` 个：训练时多标签交叉熵（玩家选的 option 为正类）；部署时取 top-maxCount。
- 处理 `minCount < maxCount`（可选选少）：训练目标为玩家实际选的数量；部署按得分降序、负分可跳（若规则允许）。

### 2.3 模型
- **小 MLP**：trunk = embedding 层 + 3-5 层 MLP（宽 256-512），ReLU/GELU。动作头 = option embedding 与 trunk concat → 小 MLP → 标量。
- **不用 transformer/大 CNN**（CPU + numpy freeze 双约束）。
- 框架：训练用 PyTorch（CPU）；部署 freeze 成纯 numpy（权重字面量 + 矩阵乘）。

---

## 3. Phase 1：IL（行为克隆）

### P1.1 数据采集与解析
- **源**：每日 episode 数据集 `kaggle/pokemon-tcg-ai-battle-episodes-YYYY-MM-DD`（~750MB zip/天，**直接 zip 内读，不解压**——21GB 解压太大）。
- **格式**（wmh 已验证，执行时第一步再核对）：
  - 一局一个 JSON：`rewards=[p0,p1]`（高者胜）；`info.Agents[i].Name` = 队名（join leaderboard CSV 得 Elo）。
  - `steps[1][pi]['action']` = 60 张 deck；`steps[t][pi]['observation']` = obs；`steps[t+1][pi]['action']` = 对 obs[t] 的回答（**off-by-one**；子决策是连续同 pi 步）。
- **筛选**：Elo ≥ 阈值（先 1000，可调）+ 目标牌组（Dragapult：`steps[1][pi]['action']` 命中 Dragapult 卡 ID 集合）。
- **产物**：`(obs_dict, picked_option_indices, context)` 元组流 → 序列化到 `.npz`（编码后张量 + 目标）。

### P1.2 训练
- 损失：多标签交叉熵（按 `maxCount` 归一）。
- 优化器：Adam，lr 1e-3，batch 256-1024。
- 验证集：留出若干天/若干局。
- 算力：6 核 CPU，~10⁶ 样本几 epoch ≈ 几十 min~1hr。

### P1.3 部署
- freeze：PyTorch state_dict → numpy 权重字面量，写入 `agents/il/main.py`（纯 numpy 前向）。
- `agent(obs_dict)`：编码 obs → numpy 前向 → 取 top-maxCount option → 返回；外层 `_legal_fallback`（异常返回前 maxCount 索引，永不崩溃）。
- deck：固定 Dragapult（`agents/il/deck.csv` = S1 的 deck）。
- **torch 可用性探针**：提交一个 `import torch; ...` 的探针 agent 看梯子是否报错；若梯子有 torch，可省 numpy freeze（但仍建议 freeze 以保速度/稳）。

### P1.4 验证与验收
- 本地：`run_match.py agents/il agents/baseline 40`（应 ≥70% vs 随机）；`agents/il vs agents/dragapult`（与 sample 互角或胜）。
- 梯子：提交，publicScore ≥ 800（接近高手）。
- 不达标：检查编码漏字段 / off-by-one / 数据筛选门槛。

---

## 4. Phase 2：RL 自对弈微调

### P2.1 自对弈循环（多进程）
- N 个 worker 进程，每个独立加载 `cg` + 当前网络权重快照，用 `cg.game.battle_start/select` 跑完整局：
  - 当前玩家用网络（随机采样，加温度）选 action；
  - 收集轨迹 `(obs, action, logp, player)`；
  - 终局 `result` → reward（胜+1/负-1，可加 shaping）。
- learner 进程：聚合 batch → PPO 更新 → 广播新权重给 workers（同步 A2C/PPO）。
- 吞吐：~120 局/秒（6 核含网络），10⁵ 局 ~15min，10⁶ 局 ~2.5hr。

### P2.2 Reward
- 主：终局胜负（`obs.current.result`，胜+1/负-1）。
- shaping（可选，谨慎）：prize 领先、本回合 KO 数、场面 HP 差、能量附着进度——**全部归一化、小权重**，避免 reward hacking。shaping 是 IL→RL 里唯一需要"设计"的部分，但远不如规则路线那般需要逐情境手写。

### P2.3 League / 对手多样性（最关键的软件设计）
只跟自己打会过拟合自己牌组/风格。对手池：
- **历史自身版本**：reservoir sampling 保留过往 checkpoint 当对手（防 cyclic / 防 overfit 当前版本）。
- **meta 牌组 bot**：把当前 top 牌组（每日 episode 挖 consensus decklist）配上简单规则 pilot（或 IL 早期版本）当对手——**注入真实 meta，对抗每日漂移**。
- **Dragapult sample agent**（S1）：作为固定参照对手。
- 采样对手时按 meta 出现频率加权（wmh 的 gauntlet 思路）。

### P2.4 训练
- 算法：PPO（clip ratio 0.1-0.3，GAE λ），on-policy。
- 从 IL 权重暖启动（只续训，不从零）。
- 周期性 checkpoint + 本地 cabt 评估（vs 固定参照池）选最佳。

### P2.5 部署与验收
- 同 P1.3 numpy freeze。
- 本地：`agents/rl vs agents/il` ≥55%；`agents/rl vs agents/dragapult` 胜率提升。
- 梯子：publicScore > IL 分数。

---

## 5. Phase 3（可选）：IS-MCTS 包装
- IL/RL 网络当**叶子估值器 + rollout 策略**（AlphaGo 式），套 IS-MCTS，用引擎 `SearchBegin/SearchStep/SearchEnd` API 规划。
- 信念采样：从公开信息 + 我方 decklist 采样对手手牌/牌序/奖品（K 个确定化）。
- 每步时限内跑 N 次 rollout，选根节点期望胜率最高的 action。
- 目标：上限 1100+。详见后续 `specs/003-...`（若做）。

---

## 6. 风险与回退

| 风险 | 影响 | 回退/缓解 |
|---|---|---|
| episode steps 结构 off-by-one 对错 | IL 学成废物 | P1.1 先下 1 个 JSON 人工核对 steps 索引与 action 对应 |
| 梯子无 torch | 不能直接部署 torch | numpy freeze（P1.3 主路径，不依赖 torch 可用性） |
| 状态编码漏关键字段 | 学不会 | 逐分量消融（去掉某分量看本地胜率掉多少） |
| 可变动作空间处理错 | 动作头失效 | mask 非法 option；多标签目标核对 maxCount |
| 自对弈 league 设计差 | RL 过拟合自己、梯子不涨 | meta bot 注入 + 历史版本 reservoir + 定期重暖 IL |
| meta 每日漂移 | RL 模型滞后 | 周期性用新 episode 数据重暖 IL；league 注入新 meta 牌组 |
| 小 MLP 表达力上限 | 上不去 | 加宽/深（仍 numpy 可 freeze）；或转 Phase 3 搜索补足 |
| 6 核不够（10⁶+ 局慢） | 迭代慢 | 临时借 Kaggle/Colab CPU（更多核）；或减 rollout/降频 |
| 全局 battle_ptr | 单进程单局 | 多进程（已验证每进程独立 .so，无冲突） |

---

## 7. 验收标准

### Phase 1（IL）
1. `agents/il/{main.py, deck.csv}` 就位，main.py 含 numpy 前向 + `_legal_fallback`，不依赖 torch。
2. `run_match.py agents/il agents/baseline 40` ≥70%，无崩溃。
3. 提交 ladder，publicScore ≥ 800。

### Phase 2（RL）
1. `agents/rl/` 就位（同 numpy freeze 模式）。
2. `agents/rl vs agents/il`（40 局）≥55%。
3. 提交 ladder，publicScore > IL 分数。

---

## 8. 执行前需先只读验证的 loose ends
1. **下一个 episode JSON**，核对 `steps[t][pi]['observation']` ↔ `steps[t+1][pi]['action']` 的 off-by-one 与子决策步（P1.1 第一步）。
2. **torch 梯子可用性**：提交 `import torch` 探针 OR 查 kaggle-environments 的 cabt 镜像依赖（决定是否必须 numpy freeze——建议无论如何 freeze）。
3. **leaderboard CSV 字段**：`kaggle competitions leaderboard --show` 已能看（teamName+score，用于 Elo 筛选）。
4. **引擎 Search API**（Phase 3）：读 `cg/sim.py` 的 Search* 签名 + `cg/api.py` 的 search_begin_input 用法。

---

## 9. 后续
- Phase 1/2 完成后，视梯子分数决定是否进 Phase 3（IS-MCTS）或转策略报告（基于 IL→RL 真实战果 + meta 分析写 ≤2000 字）。
- 若 RL 涨幅停滞，考虑：(a) 加大 league 多样性；(b) 重暖 IL 到新 meta；(c) 转 Phase 3 搜索补足表达力。
