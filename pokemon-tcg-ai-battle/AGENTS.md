# AGENTS.md — PTCG AI Battle session 备忘

给未来 session 的快速上下文。先读 `README.md` 的搭建步骤，再读本文件。

## 赛道与身份
- 模拟赛道 `pokemon-tcg-ai-battle`（引擎+Elo 梯子，截止 2026-08-16）；策略赛道 `…-strategy`（报告，24 万美元，截止 2026-09-13）。
- 单人参赛，两赛道同一账号（队名一致性自动满足，规则 §2.1.c）。
- 提交配额：5/天，最新 2 个计分，UTC 00:00（北京 08:00）重置。

## 环境（已搭好）
- Python 3.11.15（uv 装）+ `kaggle-environments==1.30.1`（含 `cabt` env）于 `venv/`。
- 引擎：`cg-lib/cg/`（`libcg.so` x86_64 + `api.py/game.py/sim.py/utils.py`）。源码参考在 `data/engine/`。
- 运行：`PYTHONPATH=cg-lib venv/bin/python ...`；`from kaggle_environments import make; env = make('cabt'); env.run([a, b])`（~1s/局）。
- venv 的 kaggle CLI 是 2.2.3（2.x，处理新版 OAuth）；系统 CLI 1.7.4.5 仅能列文件。

## 关键 gotcha（踩过的坑）
1. **Python 版本**：`kaggle-environments==1.30.1` 需 ≥3.11；3.10 上最高 1.25.9（无 `cabt`）。必须用 3.11。
2. **cabt 不把 agent 目录加入 sys.path**：`get_last_callable` 仅在 exec 期间临时 append、exec 后 pop。所以 deck.csv 不能靠 sys.path 解析（跨目录时）。
3. **cabt exec 不设 `__file__`**：agent 内用 `__file__` 会 `NameError`。读 deck 用 cwd + sys.path + `/kaggle_simulations/agent/` 三级回退（见 `agents/baseline/main.py:_deck_path`）。
4. **`Agent.act` 按 `co_argcount` 截断参数**：传 callable 给 `env.run` 时，bare `*args` wrapper（co_argcount=0）会收到 **0 个参数**。wrapper 须显式 `(observation, configuration=None)`（co_argcount=2），再按内层 agent 的 argcount 转发。文件路径方式则由 `callable_agent` 外壳处理（无需关心）。
5. **单进程单 cwd**：cabt 两个 agent 同进程共享 cwd。跨目录对战靠 `run_match.py` 的 chdir-per-call（deck 只在选牌阶段读一次，turn-based 无竞争）。
6. **`env.run` 字符串参数**：传目录路径会被当原始 Python 代码 exec（`NameError: name 'data'`）。传 `main.py` **文件路径**或 callable。
7. **cabt 噪声**：±10pts@40 局，对比用 ≥80 局；且本地 cabt 仍可能误判梯子排名（真实梯子才是唯一裁判）。

## 引擎 API 要点
- 游戏结束：`obs["current"]["result"] >= 0`（胜者索引；0/1/其他=平）。选牌阶段 `obs["select"] is None`。
- `SelectContext`：MAIN=0, SETUP_ACTIVE=1, SETUP_BENCH=2, SWITCH=3, TO_ACTIVE=4, TO_BENCH=5, TO_HAND=7, DISCARD=8, DAMAGE_COUNTER=13/14, IS_FIRST=41, MULLIGAN=42。
- `OptionType`：NUMBER=0, YES=1, NO=2, CARD=3, ENERGY=6, PLAY=7, ATTACH=8, EVOLVE=9, ABILITY=10, RETREAT=12, ATTACK=13, END=14。
- 卡牌：`from cg.api import all_card_data, all_attack`。`card.cardId/hp/weakness/resistance/ex/megaEx/stage1/stage2/attacks/skills/evolvesFrom/retreatCost`。合法池 1267 张。
- 状态字段：`obs.current.turn/yourIndex/firstPlayer/supporterPlayed/energyAttached/stadium`；`pokemon.hp/maxHp/energies/energyCards/tools/id`。
- `EnergyType`：COLORLESS=0, GRASS=1, FIRE=2, WATER=3, LIGHTNING=4, PSYCHIC=5, FIGHTING=6…
- `cg/game.py` 提供直接 ctypes harness（`battle_start/battle_select/battle_finish`），但社区报告对某些对局不如 cabt 准确——优先用 `make('cabt')`。

## 提交纪律
- 只在用户明确说「提交/submit」时才 build + submit（不要自作主张）。
- build：`bash tools/build_submission.sh agents/<name>`；submit：`venv/bin/kaggle competitions submit pokemon-tcg-ai-battle -f agents/<name>/submission.tar.gz -m "..."`。
- 改动 agent 后本地跑 `tools/run_match.py` 确认无崩溃/INVALID 再提交。

## 卡牌图鉴 PDF
`../pokemon-tcg-ai-battle-challenge-strategy/Card_ID List_{EN,JP}.pdf` 是卡牌视觉参考（很大，137MB/182MB，gitignored；本模型不能直接读 PDF）。

## 设计文档
- `specs/000-ptcg-env-setup/plan.md` — 本环境的搭建设计（Approach B，已实施）。
- `specs/001-dragapult-sample-agent/plan.md` — S1：从官方 Dragapult ex sample agent 起步。
