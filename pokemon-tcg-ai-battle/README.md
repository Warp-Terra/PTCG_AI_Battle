# PTCG AI Battle — 本地环境

Pokémon TCG AI Battle Challenge（Kaggle × The Pokémon Company × 松尾研 × HEROZ）模拟赛道的本地开发环境。官方对战引擎 `cabt` 跑通，可开发/测试 `agent(obs_dict)`、本地对战、打包提交。

## 关联赛道
- `pokemon-tcg-ai-battle`（**模拟赛道**，引擎在此，Elo 梯子，截止 2026-08-16）
- `pokemon-tcg-ai-battle-challenge-strategy`（策略报告赛道，24 万美元，截止 2026-09-13）
- 两赛道**同一账号/队名**（规则 §2.1.c），单人参赛已满足。

## 一次性搭建

### 1. Python 3.11 环境
`kaggle-environments==1.30.1`（含 `cabt`）要求 Python ≥3.11。用 `uv`（已装于 `~/.local/bin/uv`）管理：
```bash
uv python install 3.11
uv venv --python 3.11 venv
uv pip install --python venv -r requirements.txt
```

### 2. 下载官方引擎与样例
引擎在 `sample_submission/sample_submission/cg/`（预编译 `libcg.so` + Python 封装），源码在 `ptcg_engine/ptcgProgram 22/`。需先在 Kaggle 加入 `pokemon-tcg-ai-battle` 并接受规则。
```bash
SS="sample_submission/sample_submission"
kaggle competitions download -c pokemon-tcg-ai-battle -f "$SS/main.py"     -p data/sample_submission
kaggle competitions download -c pokemon-tcg-ai-battle -f "$SS/deck.csv"     -p data/sample_submission
kaggle competitions download -c pokemon-tcg-ai-battle -f "$SS/cg/__init__.py" -p data/sample_submission/cg
kaggle competitions download -c pokemon-tcg-ai-battle -f "$SS/cg/api.py"    -p data/sample_submission/cg
kaggle competitions download -c pokemon-tcg-ai-battle -f "$SS/cg/game.py"   -p data/sample_submission/cg
kaggle competitions download -c pokemon-tcg-ai-battle -f "$SS/cg/sim.py"    -p data/sample_submission/cg
kaggle competitions download -c pokemon-tcg-ai-battle -f "$SS/cg/utils.py"  -p data/sample_submission/cg
kaggle competitions download -c pokemon-tcg-ai-battle -f "$SS/cg/libcg.so"  -p data/sample_submission/cg
kaggle competitions download -c pokemon-tcg-ai-battle -f "ptcg_engine/ptcgProgram 22/README.md" -p data/engine
cp -r data/sample_submission/cg cg-lib/cg                        # 本地运行用副本
cp ../pokemon-tcg-ai-battle-challenge-strategy/EN_Card_Data.csv data/card/   # 卡牌数据（已有）
```
> `libcg.so` 是 x86_64 Linux 预编译；本机 glibc 2.35 兼容（`ldd cg-lib/cg/libcg.so` 全部依赖解析）。

### 3. 验证引擎
```bash
PYTHONPATH=cg-lib venv/bin/python -c "import cg; from kaggle_environments import make; make('cabt'); print('OK')"
```

## 本地对战
```bash
PYTHONPATH=cg-lib venv/bin/python tools/run_match.py <agent_a_dir> <agent_b_dir> [games]
PYTHONPATH=cg-lib venv/bin/python tools/run_match.py agents/baseline data/sample_submission 10
```
- `run_match.py` 把每个 agent 加载为 callable，签名 `(observation, configuration=None)`（kaggle_environments `Agent.act` 按 `co_argcount` 截断参数——bare `*args` 会收到 0 参），并在调用时 chdir 到 agent 目录使 cwd-relative 的 `deck.csv` 解析（cabt 单进程单 cwd，否则跨目录 agent 互相抢 deck）。
- `cg` 由 `cg-lib` 全局提供（`from cg.api import` 在模块加载时解析）。
- cabt 有噪声（±10pts@40 局），对比用 ≥80 局。

## Agent 契约
- `agent(obs_dict: dict) -> list[int]`，返回**选项索引**（`[0, len(obs.select.option))` 内 `maxCount` 个不重复值）。
- 选牌阶段（`obs.select is None`）返回 60 张卡 ID。
- **永不崩溃**（须有合法 fallback）+ 守每步时限。
- 牌组：60 张，须符合 PTCG 规则；非法牌组 → `INVALID`。

## 打包与提交
```bash
bash tools/build_submission.sh agents/baseline      # 产 agents/baseline/submission.tar.gz
venv/bin/kaggle competitions submit pokemon-tcg-ai-battle -f agents/baseline/submission.tar.gz -m "msg"
venv/bin/kaggle competitions submissions pokemon-tcg-ai-battle     # 查状态/分数
```
- 提交包 = `main.py + deck.csv + cg/`（`cg/` 复制自 `cg-lib/cg`，含 `libcg.so`）。
- 配额：5 次/天，最新 2 个计分，UTC 00:00（=北京 08:00）重置。

## 目录约定
- `agents/<name>/`：每个 agent 自包含 `main.py` + `deck.csv`；读 deck 用 cwd/sys.path/Kaggle 三级回退（见 `agents/baseline/main.py:_deck_path`），不依赖 `__file__`（cabt exec 不设 `__file__`）。
- `cg-lib/cg/`：可运行引擎副本（gitignored，体积大+许可限制）。
- `data/`：官方素材（gitignored）。

## 后续（未在本阶段做）
- 拉取官方样例 agent notebooks（Crustle / Mega Lucario ex / Mega Abomasnow ex / Dragapult ex / Iono's Bellibolt）作参考：`kaggle kernels pull -k <owner>/<slug> -p data/notebooks/`。
- 评估 harness（胜率矩阵/soak/A/B）、meta/episode 回放、策略报告脚手架（见 `specs/` 设计文档的 Approach C）。
