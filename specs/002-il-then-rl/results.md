# IL→RL 执行结果

本文件记录 `specs/002-il-then-rl/plan.md` 的执行结果，与 plan 分离。plan 只描述方案，结果/数据/分数记录在此。

---

## Phase 1（IL 行为克隆）— 2026-07-06

Pipeline 全程打通：episode 数据 → 编码 → 训练 → numpy freeze → 部署 → 提交。

### 步骤记录
- **P1.1a 数据采集与结构核对**：下 06-16 episode 数据集（`kaggle/pokemon-tcg-ai-battle-episodes-2026-06-16`，102MB zip，**每局一个 JSON** 非整 zip）。核对结构：
  - `steps[t][pi].observation` ↔ `steps[t+1][pi].action`（**off-by-one 确认**）。
  - 选牌 obs `select is None`（→ step[1] 交 60 张 deck）；游戏中 `select` 非 None。
  - obs 顶层：`select/logs/current/search_begin_input/remainingOverageTime/step`。
  - player：`active(list)/bench(list)/hand(list)/discard/prize(6 槽多为 null)/deckCount/handCount`；卡牌字段是 `id`（cardId）。
  - option：`type(OptionType) + area/index/playerIndex` 或 `index`(手牌) 或 `attackId`。
- **P1.1b/c 解析与编码**：写 `tools/parse_episodes.py`（计数）+ `tools/il_encode.py`（编码）。
  - 06-16：1277 局 / 412 局含 Dragapult 玩家 / **32,169 个 Dragapult 决策对**。
  - 编码器输出：state_scalars(72) + slot_card_ids(12) + slot_scalars(48) + hand_card_ids(20) + stadium_id + option（type/area/card_id/target_id/attack_id/scalars）+ picked 标签。
  - 数据集 `data/il_decisions_06-16.pkl`（37MB，gitignored）。
- **P1.2 训练**：`tools/il_train.py`。
  - 架构：card Embedding(1268,64) + attack Embedding(1557,64) + state MLP(→256) + option MLP(→128) + score MLP(→1)。
  - 646,593 参数（小，numpy 可 freeze）。
  - 8 epoch CPU，lr 1e-3，bs 64，masked BCE（picked 权重 1/maxCount）。
  - val top-k 准确率：0.575 → **0.662**（随机基线 ~20%）；loss 0.397 → 0.314。
- **P1.3 部署**：`tools/il_freeze.py` 导出 `agents/il/model.npz`（2.6MB）；写 `agents/il/main.py`（编码器 + 纯 numpy 前向 + `_legal_fallback` + `_asset_path`，deck = Dragapult）；`build_submission.sh` 更新含 `*.npz`。
- **P1.4 验证与提交**：
  - 本地 `run_match il vs baseline` 40 局 = **52.5%**（21W/19L），0 崩溃。
  - `il vs dragapult sample` 20 局 = 55%（11W/9L）。
  - 动作分布健康（3 局探针）：PLAY 71 / EVOLVE 17 / ATTACH 16 / ATTACK 13 / ABILITY 29 / END 仅 9——真在打 Dragapult，非默认被动。
  - 打包 2.79MB（main.py + deck.csv + model.npz + cg/），干净目录验证通过。
  - 提交，`SubmissionStatus.PENDING`。

### 分析
- **本地 52.5% vs 随机偏低**，但动作分布正常（PLAY/EVOLVE/ATTACH/ATTACK 都有，END 仅 9）。疑为**分布偏移**：IL 在真实 meta 对局训练，vs 随机 sample 牌组（35 基础水能量）是 OOD，打分噪声大、近似随机选。梯子打真实 meta（in-distribution）应更准。
- **推理速度**：IL 是小 numpy MLP（~1ms/步），不会超时（对比 S1 Dragapult sample 854 行 Python 可能超时）。

### 梯子分数
| 提交 | 日期 | publicScore | 备注 |
|---|---|---|---|
| random baseline | 07-06 | 344.5 | 底分参考 |
| S1 Dragapult sample | 07-06 | 289.7 | 低于随机；sample 过时或超时 |
| **P1 IL (全玩家, 8ep)** | 07-06 | **400.4** | 全段位模仿，中等偏上 |
| **P1.1 IL (Elo≥800, 16ep)** | 07-06 | **待提交** | 本地 vs baseline 57.5% / vs dragapult 70% |

## Phase 1.1（Elo 筛选）— 2026-07-06

对 P1 IL 做 Elo 筛选改进：只从高分玩家（Elo ≥ 800）的 Dragapult 决策中学习。

### Leaderboard 分布
- 总队伍 4367，Elo ≥ 800 = 1103（前 25%），Elo ≥ 900 = 460，Elo ≥ 1000 = 81。
- 06-16 单天：原始 412 局 Dragapult / 32k 决策；Elo ≥ 800 → 139 局 / 10,769 决策（约 1/3 但高质量）。

### 训练与评估（Elo ≥ 800, 16 epoch）
- val top-k 准确率：**0.684**（原始全玩家模型 0.662 → +2.2pp，仅用 1/3 数据）
- `il vs baseline`（40 局）：**57.5%**（23W/17L；原始 52.5% → +5pp）
- `il vs dragapult sample`（20 局）：**70.0%**（14W/6L；原始 55% → +15pp）
- 0 崩溃，动作分布正常。

### 改进
| 指标 | 原始（全玩家） | Elo ≥ 800 |
|---|---|---|
| val top-k acc | 0.662 | **0.684** |
| vs baseline | 52.5% | **57.5%** |
| vs dragapult sample | 55% | **70.0%** |

### 待办（视 IL 分数定）
- ≥800 → Phase 1 达标，进 Phase 2 (RL)。
- 当前 400.4 → 继续改进方向：
  1. ~~**Elo 筛选**~~ ✅ 已做。下步可试 ≥ 900（88 局 / 7k 决策）或 ≥ 1000 but 需更多天数据。
  2. **更多数据**：下几天完整 episode（~750MB/天），叠加 Elo 筛选后可凑够数据量。
  3. **加入 logs 编码**（当前未编 logs，可能漏事件信息）。
  4. **编码消融**：逐分量（hand pool / slot / logs）看 val top-k 掉多少，补关键字段。

### 产物
- `tools/{parse_episodes.py, il_encode.py, il_train.py, il_freeze.py}`
- `agents/il/{main.py, deck.csv, model.npz, model.pt, model_elo800.npz, model_elo800.pt, model_allplayers.npz, submission.tar.gz}`
- `data/{leaderboard.csv, il_decisions_06-16.pkl, il_decisions_elo800_06-16.pkl}`
- `data/episodes/pokemon-tcg-ai-battle-episodes-2026-06-16.zip`（102MB，gitignored）
