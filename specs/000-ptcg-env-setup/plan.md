# PTCG AI Battle Challenge — 官方比赛环境搭建设计文档

- **日期**: 2026-07-06
- **状态**: 已实施（P0–P7 全部完成，2026-07-06）
- **赛道**: `pokemon-tcg-ai-battle`（模拟赛道，引擎所在）+ `pokemon-tcg-ai-battle-challenge-strategy`（策略报告赛道）
- **搭建深度**: Approach B（本地可跑：引擎 + sample agent + 自定义 harness + 打包提交）
- **参赛身份**: 单人参赛（两赛道同一账号，队名一致性自动满足）

---

## 1. 背景与目标

Kaggle × The Pokémon Company × 松尾研 × HEROZ 联合举办的 Pokémon TCG AI Battle Challenge 分两个关联赛道：

| 赛道 | 用途 | 截止 | 状态 |
|---|---|---|---|
| `pokemon-tcg-ai-battle`（模拟赛道，Knowledge） | Elo 梯子自动对战，**官方引擎在此** | 2026-08-16 | 已加入 |
| `pokemon-tcg-ai-battle-challenge-strategy`（策略赛道，24 万美元） | ≤2000 字策略报告（70% 方法 + 20% 牌组 + 10% 报告） | 2026-09-13 | 已加入 |

**目标**：在本地搭建可运行的官方对战环境，使 `agent(obs_dict)` 能在官方 `cabt` 引擎中对战、测试，并能打包成 `submission.tar.gz` 提交到模拟赛道梯子。该环境是后续开发对战 AI 与撰写策略报告的共同基础。

**非目标**（本阶段不做）：
- 实现高强度对战策略 / MCTS / 学习型评估器。
- 本地评估 harness 的胜率矩阵 / meta 分析 / episode 回放工具（属 Approach C）。
- 策略报告（WRITEUP.md）撰写。

---

## 2. 关键发现（探索结论）

### 2.1 引擎发行包
模拟赛道数据页 `sample_submission/sample_submission/` 提供完整可运行引擎：

```
sample_submission/sample_submission/
  main.py            # 官方最小 baseline agent（约 1.3KB）
  deck.csv           # 样例 60 张牌组（约 245B）
  cg/                # 引擎包
    __init__.py
    api.py           # 卡牌/招式数据接口（all_card_data / all_attack）
    game.py
    sim.py
    utils.py
    libcg.so         # Linux x86_64 预编译原生库（1.34MB）← 本机使用
    libcg-arm64.so   # Linux ARM64
    libcg.dylib      # macOS
    cg.dll           # Windows
```

`ptcg_engine/ptcgProgram 22/`（C++ header-only 源码：`Api.h`、`BattleData.h`、`Card.h`、`CardImpl.h` 878KB、`Export.cpp`、`GameProc.h`、`README.md` 等）仅作源码参考，**运行不需要编译**。

### 2.2 本地运行方式
```python
import cg  # 放到 sys.path 后导入，注册引擎
from kaggle_environments import make
env = make('cabt')              # 官方环境名
env.run([agent_a, agent_b])     # 约 1s/局
```
- 依赖：`kaggle-environments==1.30.1`（与梯子同版本）+ `cg/` 包在 `sys.path`。
- `cabt` 环境由 `kaggle-environments` 与 `cg` 包共同提供（P3 验证确切注册方式）。

### 2.3 Agent 契约
- 签名：`agent(obs_dict) -> list[int]`，返回**选项索引**列表。
- 选牌阶段（`obs.select is None`）：返回 60 张卡 ID。
- **永不崩溃**（必须总有合法 fallback）+ **遵守每步时限**。
- 提交物：`submission.tar.gz` 内含 `main.py` + `deck.csv` + `cg/`。
- 提交命令：`kaggle competitions submit pokemon-tcg-ai-battle -f submission.tar.gz -m "..."`。
- 配额：5 次/天，最新 2 个计分，UTC 00:00 重置。

### 2.4 官方样例 agent notebooks（Code 标签）
- `beating-the-day-1-1-crustle-bot.ipynb`
- `a-sample-rule-based-agent-mega-lucario-ex-deck.ipynb`
- `a-sample-rule-based-agent-mega-abomasnow-ex-deck.ipynb`
- `a-sample-rule-based-agent-dragapult-ex-deck.ipynb`
- `a-sample-rule-based-agent-iono-s-deck.ipynb`

### 2.5 引擎 API 要点（供 agent 开发参考，源自社区文档）
- 游戏结束：`obs.current.result != -1`（胜者索引）。
- `SelectContext`：MAIN=0, SETUP_ACTIVE=1, SETUP_BENCH=2, SWITCH=3, TO_ACTIVE=4, TO_BENCH=5, TO_HAND=7, DISCARD=8, DAMAGE_COUNTER=13/14, IS_FIRST=41, MULLIGAN=42。
- `OptionType`：NUMBER=0, YES=1, NO=2, CARD=3, ENERGY=6, PLAY=7, ATTACH=8, EVOLVE=9, ABILITY=10, RETREAT=12, ATTACK=13, END=14。
- 卡牌数据：`all_card_data()` → `card.cardId/hp/weakness/resistance/ex/megaEx/stage1/stage2/attacks/skills/evolvesFrom/retreatCost`；`all_attack()` → `attack.attackId/name/damage/energies/text`。
- cabt 评估有噪声（±10pts@40 局），对比需 ≥80 局。

---

## 3. 项目结构

```
PTCG_AI_Battle/
  pokemon-tcg-ai-battle-challenge-strategy/   # 已有卡牌数据（保留原位）
    EN_Card_Data.csv, JP_Card_Data.csv, Card_ID List_{EN,JP}.pdf
  specs/                                       # 设计文档（本文件所在）
  pokemon-tcg-ai-battle/                       # 模拟赛道工作区
    data/
      card/                # EN/JP_Card_Data.csv（复制自上级 + 重下）
      engine/              # ptcg_engine/ 源码 + README（参考）
      sample_submission/   # 官方 main.py + deck.csv + cg/
      notebooks/           # 官方样例 agent notebooks
    agents/
      baseline/            # main.py + deck.csv（复制自 sample_submission）
    cg-lib/cg/             # 可运行引擎（复制自 sample_submission/cg）
    tools/
      run_match.py         # 本地 cabt 对战 harness
      build_submission.sh  # 打包 submission.tar.gz
    venv/
    requirements.txt
    README.md
    AGENTS.md
    .gitignore
```

---

## 4. 实施步骤

### P0 预检（已完成，只读）
- 架构：x86_64 / glibc 2.35 / Python 3.10.12 / Docker 29.1.3（与 `libcg.so` 兼容）。
- 两赛道均已加入；CLI 1.7.4.5 能列文件。
- 无 GPU / 无 conda（本阶段不需要）。

### P1 脚手架
- `git init`（当前非 git 仓库）；建目录结构。
- `.gitignore`：`venv/`、`**/cg/`、`*.tar.gz`、`data/`、`.kaggle*/`、`__pycache__/`。
- `requirements.txt`：`kaggle-environments==1.30.1`、`kaggle`、`numpy>=1.26`、`pandas>=2.0`、`pytest>=8`。

### P2 下载引擎与样例
- `kaggle competitions download -c pokemon-tcg-ai-battle -f "<path>" -p data/<dest>` 逐文件下载：
  - `sample_submission/sample_submission/` 全部（`cg/`、`main.py`、`deck.csv`）→ `data/sample_submission/`。
  - `ptcg_engine/ptcgProgram 22/`（源码 + README）→ `data/engine/`（参考）。
  - 卡牌数据 `EN_Card_Data.csv` / `JP_Card_Data.csv` → `data/card/`（上级已有，复制统一管理）。
- 复制 `data/sample_submission/cg/` → `cg-lib/cg/`（本地运行用，sys.path 指向 `cg-lib`）。
- 官方样例 notebooks：`kaggle kernels pull -k <owner>/<slug> -p data/notebooks/`（先在 Code 标签查 slug，逐个拉取）。

### P3 Python 环境
- `python3 -m venv venv && venv/bin/pip install -U pip && venv/bin/pip install -r requirements.txt`。
- 验证脚本：
  ```python
  import sys; sys.path.insert(0, "cg-lib")
  import cg
  from kaggle_environments import make
  env = make("cabt")
  print("cabt OK", env)
  ```
- **最大风险点**：若 `make('cabt')` 失败 → 排查是否需先 `import cg` 注册（已在脚本里前置）；若 `libcg.so` 加载失败 → 检查 `ldd cg-lib/cg/libcg.so` 依赖；回退 Docker（Kaggle 官方基础镜像 `gcr.io/kaggle/python` 或 `kaggle-environments` 镜像）。

### P4 引擎跑通
- 写 `tools/run_match.py`：
  - 参数：`<agent_a.py> <agent_b.py> [games]`；可指定各自 `deck.csv`。
  - `make('cabt')`、轮换座位、跑 N 局、统计胜/负/平 + 崩溃/超时检测。
  - 参考 `sys.path` 注入 `cg-lib`，agent 文件路径加载。
- 跑 `data/sample_submission/main.py` vs 自身 → 确认一局完整结束、`rewards` 正常赋值。

### P5 最小 agent + deck
- 复制 `data/sample_submission/{main.py,deck.csv}` → `agents/baseline/`。
- 在 `main.py` 外层加 `_legal_fallback` 包装：捕获异常 → 返回第一个合法选项（索引 0 或空列表，按 obs 语义），保证永不崩溃。
- `run_match.py` 跑 baseline vs sample → 通过（有完整局、无崩溃）。

### P6 打包 + 端到端冒烟（已获用户批准）
- 写 `tools/build_submission.sh`：
  ```bash
  #!/usr/bin/env bash
  set -euo pipefail
  AGENT="${1:-agents/baseline}"
  TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
  cp "$AGENT/main.py" "$TMP/main.py"
  cp "$AGENT/deck.csv" "$TMP/deck.csv"
  cp -r cg-lib/cg "$TMP/cg"
  ( cd "$TMP" && tar -czf "$AGENT/submission.tar.gz" . )
  tar -tzf "$AGENT/submission.tar.gz" | grep -E "main|deck|cg/"
  ```
- 干净目录验证：解压 tarball 到临时目录，`import main; import cg` 不报错。
- **真实提交一次** baseline 到模拟赛道：
  ```bash
  venv/bin/kaggle competitions submit pokemon-tcg-ai-battle \
    -f agents/baseline/submission.tar.gz -m "env-setup smoke test"
  venv/bin/kaggle competitions submissions pokemon-tcg-ai-battle
  ```
  确认提交被接受、能拿到分数（即使很低）。

### P7 文档
- `README.md`：环境搭建步骤、命令清单、`cabt` 运行方式、agent 契约、提交流程、配额规则。
- `AGENTS.md`：给未来 session 的备忘——引擎 API 要点、SelectContext/OptionType 枚举、cabt 噪声特性、队名一致性要求、`.gitignore` 为何排除 `cg/`（体积大 + 许可）。

---

## 5. 风险与回退

| 风险 | 影响 | 回退 |
|---|---|---|
| `libcg.so` 本地加载失败 | 引擎无法本地跑 | `ldd` 查依赖；Docker 用 Kaggle 基础镜像 |
| `make('cabt')` 找不到环境 | 无法对战 | 先 `import cg` 注册；确认 kaggle-environments 版本=1.30.1 |
| CLI OAuth 下载/提交 403 | 无法拿数据/提交 | 升级 `kaggle>=2.2`（SebAustin 文档提到新版 OAuth 需 2.x） |
| 官方 notebooks slug 未知 | 拉不到样例 agent | 在 Kaggle Code 标签手工查；或先只用 `sample_submission/main.py` |
| cabt 噪声大 | 评估不准 | 本阶段仅冒烟，不评估强度；后续开发用 ≥80 局 |

---

## 6. 验收标准（Definition of Done）

1. `venv` 内 `import cg` + `make('cabt')` 成功，`libcg.so` 正常加载。
2. `tools/run_match.py` 能让两个 agent 完成一局对战，输出胜/负/平 + 无崩溃。
3. `agents/baseline`（带 `_legal_fallback`）能在 cabt 中跑通，永不崩溃。
4. `tools/build_submission.sh` 产出 `submission.tar.gz`，干净目录可加载。
5. baseline 已成功提交到 `pokemon-tcg-ai-battle` 并获得分数。
6. `README.md` + `AGENTS.md` 完成，未来 session 可据此复现环境。

---

## 7. 后续

本设计文档评审通过后：
1. 进入实施阶段，按 P1→P7 顺序执行；每步验证后再进下一步。
2. 环境搭好后，可启动 Approach C（评估 harness / meta 分析 / 策略报告脚手架）或直接开始 agent 策略开发——届时另立设计文档。
3. 单人参赛，每日 5 次提交配额需精打细算（最新 2 个计分）。

---

## 8. 实施结果与偏差（2026-07-06）

P0–P7 全部完成，验收标准全部满足。实施中的关键偏差与发现：

1. **Python 版本**（重大偏差）：`kaggle-environments==1.30.1` 需 Python ≥3.11，本机 3.10.12 不兼容（3.10 上最高 1.25.9，无 `cabt`）。改用已安装的 `uv` 装 Python 3.11.15 + 重建 venv。无 pyenv/conda、sudo 需密码，uv 是免 sudo 最优解。
2. **`cabt` 是 `kaggle-environments` 内置 env**（`envs/cabt/cabt.py`），env 自带 `cg` 引擎包（相对导入 `.cg.game`）；agent 的 `from cg.api import` 走绝对导入，由 `cg-lib` 提供。
3. **agent 调用约定**（关键发现）：`kaggle_environments.Agent.act` 按 `co_argcount` 截断参数——bare `*args` callable wrapper（co_argcount=0）收到 0 参。`run_match.py` 用显式 `(observation, configuration=None)` 签名（co_argcount=2）+ chdir-per-call 解决跨目录 deck 解析。
4. **`__file__` 不可用**：cabt 用 `exec` 加载 agent 不设 `__file__`。baseline 的 `_deck_path` 改用 cwd + sys.path + `/kaggle_simulations/agent/` 三级回退。
5. **CLI**：venv 装的是 kaggle 2.2.3（2.x，处理新版 OAuth），系统 CLI 1.7.4.5 仅能列文件。
6. **冒烟提交**：baseline（随机 agent）已提交，`SubmissionStatus.PENDING`，487KB 包，含 `main.py + deck.csv + cg/`（`libcg.so`）。
7. **合法池 1267 张**：`all_card_data()` 返回 1267 张（与社区文档一致）。
8. **notebooks 暂缓**：官方样例 agent notebooks（Crustle 等）未拉取（非环境验证必需），列为后续。

详细命令与 gotcha 见 `pokemon-tcg-ai-battle/README.md` 与 `pokemon-tcg-ai-battle/AGENTS.md`。
