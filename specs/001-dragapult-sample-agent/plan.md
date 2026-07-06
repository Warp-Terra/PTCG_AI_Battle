# S1：从官方 Dragapult ex sample agent 起步

- **日期**: 2026-07-06
- **状态**: 待执行
- **阶段**: S1（快速起跳，从 308 Elo 跃到 ~800–1000）
- **前置**: 本地 cabt 环境已搭好（见 `specs/000-ptcg-env-setup/plan.md`，P0–P7 完成）
- **方向**: 从官方 sample agent 起步（用户选定）
- **牌组**: Dragapult ex（当前 top meta，63% WR），用 sample notebook 自带牌组（用户选定）

---

## 1. 背景与目标

### 1.1 现状
- 唯一提交 = 随机 baseline（`random.sample` 选合法选项），publicScore = **308.9**，4354 队里垫底。
- 梯子 top1 = 1217.3（Majkel1337），前 20 在 1065–1217。严肃参赛者 ~1000+。
- 我们**没有任何真实 agent 策略**，308 是"什么都不做"的底分。

### 1.2 为什么从官方 sample 起步
社区研究（wmh/ptcg-abc）结论：
- top 梯队是**规则化逐牌策略 + meta 牌组 + divergence mining**，不是深度学习。
- **官方 sample agent 是写好的规则 pilot**，远胜随机；wmh 实测官方 Dragapult sample 击败从零写的策略 13:1。
- **牌组选择主导**分数；Dragapult ex 是当前 top-tier（63% WR，Phantom Dive 控场）。
- 性价比最高的跃迁：**官方 sample agent 直接打包提交，从 308 跳到 ~800–1000**。

### 1.3 S1 目标
拉取官方 `kiyotah/a-sample-rule-based-agent-dragapult-ex-deck` notebook，提取其 `agent()` + 自带 deck，加固后打包提交，验证 Elo 从 308 跃迁到 ≥800。

### 1.4 非目标（S1 不做）
- divergence mining / 逐牌打分调优（属 S2）。
- 换 consensus meta 牌组（S2）。
- IS-MCTS / RL（S3）。
- 策略报告撰写。

---

## 2. 关键资源（已定位）

| 资源 | slug/路径 | 用途 |
|---|---|---|
| **官方 Dragapult sample notebook** | `kiyotah/a-sample-rule-based-agent-dragapult-ex-deck`（Kiyota，289 票，2026-06-19） | S1 的 agent+deck 来源 |
| 官方 RL/MCTS sample | `kiyotah/reinforcement-learning-and-mcts-sample-code`（683 票） | S3 搜索参考 |
| 备选 strong baseline | `romanrozen/strong-start-baseline-agent-v10-lb-950`（公开 LB 950） | 若 sample 路线受阻的备选 |
| 本地环境 | `pokemon-tcg-ai-battle/`（venv + cg-lib + tools） | 已就绪 |

> wmh 的 `cabt_eval.py` 标注 dragapult sample 的 agent 源在 notebook 的 **cell index 3**（0-indexed），cell 可能以 `%%` magic 开头需剥离。

---

## 3. 实施步骤

### S1.1 拉取 notebook
```bash
cd pokemon-tcg-ai-battle
venv/bin/kaggle kernels pull -k kiyotah/a-sample-rule-based-agent-dragapult-ex-deck \
  -p data/notebooks/dragapult/
```
产物：`data/notebooks/dragapult/a-sample-rule-based-agent-dragapult-ex-deck.ipynb`（+ `kernel-metadata.json`）。

### S1.2 检视并提取 agent + deck
1. 解析 `.ipynb` JSON，列出各 code cell 的前几行，定位：
   - 含 `def agent` 或 `_policy` 的 cell → agent 源码。
   - 含 60 个卡 ID（list 或 deck.csv 读取）的 cell → deck。
2. 若 agent cell 以 `%%` 开头（cell magic），剥离首行。
3. 产物：
   - `agents/dragapult/main.py`（agent 源码 + 必要 import）。
   - `agents/dragapult/deck.csv`（60 个卡 ID，每行一个）。
4. 若 deck 是 Kaggle 数据集附件（不在 notebook 内），S1 先用 sample agent 内联的 deck list；若都没有，回退到 baseline 的 deck（仅用于跑通，提交前必须换合法 Dragapult 牌组）。

### S1.3 加固 agent
在提取出的 `main.py` 外层套用与 `agents/baseline` 相同的加固模式：
1. **deck 解析**：用 `_deck_path()`（cwd + sys.path + `/kaggle_simulations/agent/` 三级回退），**不依赖 `__file__`**（cabt exec 不设 `__file__`）。
2. **永不崩溃**：外层 `try/except` 包裹原 agent 逻辑，异常时 `_legal_fallback`（返回前 `maxCount` 个选项索引），deck 读取失败也兜底。
3. 保留 sample 原有的 `_policy`/打分逻辑不动（S1 不调策略，只加固）。

> 若 sample agent 已自带 deck 读取与 fallback，核对无误后保留其实现，仅补齐缺失的加固项。

### S1.4 本地验证
```bash
cd pokemon-tcg-ai-battle
PYTHONPATH=cg-lib venv/bin/python tools/run_match.py agents/dragapult agents/baseline 20
PYTHONPATH=cg-lib venv/bin/python tools/run_match.py agents/dragapult data/sample_submission 20
```
验收：
- 全部 `DONE`，无 `ERROR`/`INVALID`。
- dragapult vs baseline（随机）胜率应 ≥70%（规则 pilot 应明显压制随机）。
- 无崩溃/超时。

### S1.5 打包 + 干净目录验证
```bash
bash tools/build_submission.sh agents/dragapult
# 干净目录验证
TMP=$(mktemp -d) && tar -xzf agents/dragapult/submission.tar.gz -C "$TMP"
PYTHONPATH="$TMP" venv/bin/python -c "import cg, main; print('cards', len(__import__('cg.api', fromlist=['all_card_data']).all_card_data())); print('deck', len(main.read_deck_csv())); print('OK')"
rm -rf "$TMP"
```
验收：`import cg` + `import main` + deck 60 张无报错。

### S1.6 提交 + 看分
```bash
venv/bin/kaggle competitions submit pokemon-tcg-ai-battle \
  -f agents/dragapult/submission.tar.gz -m "S1: dragapult official sample agent"
venv/bin/kaggle competitions submissions pokemon-tcg-ai-battle
```
验收：提交被接受（`COMPLETE`/`PENDING`），等 UTC 00:00（北京 08:00）出分，publicScore 从 308 跃到 **≥800**（目标 ~900）。

---

## 4. 风险与回退

| 风险 | 影响 | 回退 |
|---|---|---|
| notebook agent cell 结构不符预期（多 cell 拼接、依赖 notebook 运行时） | 提取失败 | 手工检视全部 cell，必要时合并多 cell 源码；参考 wmh `cabt_eval.py` 的提取方式（cell 3） |
| deck 不在 notebook 内（是 Kaggle 数据集附件） | 无 deck | S1 临时用 baseline deck 跑通本地；提交前从 episode 数据挖 consensus Dragapult 牌组（提前进入 S2 的 deck 部分） |
| sample agent 在本地 cabt 崩溃/超时 | 本地验证不过 | 加 `_legal_fallback`；若仍超时，简化或排查时限（cabt actTimeout） |
| 本地胜率 vs 梯子不一致 | 误判 | 以梯子为准（wmh 警告 cabt 对部分对局系统性偏）；S1 只看"能否完赛+明显压制随机" |
| 提交后分数未达 800 | S1 目标未达 | 检视 submission 日志（`kaggle competitions submissions` 的 error 字段）；若 agent 在梯子崩，回退 baseline + 排查 |
| 每日 5 次提交配额 | 超额 | S1 仅 1 次提交；调优（S2）再用配额 |

---

## 5. 验收标准（Definition of Done）

1. `agents/dragapult/{main.py, deck.csv}` 就位，main.py 含 sample 的 `_policy`/`agent` 逻辑 + `_deck_path` + `_legal_fallback` 加固。
2. `run_match.py` 跑 dragapult vs baseline 20 局：全 `DONE`，胜率 ≥70%。
3. `build_submission.sh` 产出 `submission.tar.gz`，干净目录 `import cg/main` + deck 60 张通过。
4. 已提交到 `pokemon-tcg-ai-battle`，状态 `COMPLETE`，publicScore ≥ 800。
5. `specs/001/` 本文档状态更新为"已完成"+ 记录实际分数。

---

## 6. 后续

- **S2（调策略）**：拉每日 episode 数据集（`kaggle datasets download kaggle/pokemon-tcg-ai-battle-episodes-YYYY-MM-DD`），对 Elo≥1150 选手做 divergence mining（按 `SelectContext` 分桶找我们 vs top 的分歧），改逐牌打分；换 consensus meta 牌组。目标 1000–1100。届时另立 `specs/002/`。
- **S3（冲顶）**：自选/克隆 top 牌组 + 完整 per-card policy（BasePolicy 抽象），或上 IS-MCTS（参考 `kiyotah/reinforcement-learning-and-mcts-sample-code`、SebAustin 仓库）。目标 1100+。
- **策略报告**：基于 S1–S3 的真实战果 + meta 分析写 ≤2000 字报告（策略赛道 70% 方法 + 20% 牌组 + 10% 报告）。
